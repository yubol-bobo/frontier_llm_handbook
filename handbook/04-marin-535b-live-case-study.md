# 04｜Marin 535B：一场仍在进行中的超大模型训练，怎样作出真实决策

[返回全流程](00-end-to-end.md) · [数据与配方](01-data-model-pretraining-design.md) · [分布式执行](02-distributed-pretraining-operations.md) · [后训练](03-posttraining-agent-rl-evaluation.md)

审读日期：2026-09-08。本章固定本地 Marin 快照 `5e2436d0f61462983003bd8b6eaef8235ecab78c`，追读 `experiments/grug/moe_hero_ep` 的模型、数据、launcher、训练步和恢复代码，并对照官方公告及公开 issue。**源码事实**表示这个快照的行为；**团队陈述／计划**保留发布时间；**工程解读**解释设计含义。没有运行上游程序、访问训练数据、提交作业或测量训练性能。

## 1. 先对齐时间：启动公告、未来承诺和当前代码是三种证据

9 月 2 日公告确认训练已经启动：535B 总参数、约 23B 激活参数，目标约 18T tokens，并把 12 月 1 日写成预计预训练结束时间，提前登记 Paloma 表现预测。它不是完成报告，也没有证明最终 agent 能力。[9 月 2 日官方公告](https://openathena.ai/blog/huang-foundation-marin-535b-training-run/)

9 月 3 日 David Hall 的文章描述当时运行情况，并报告 ragged all-to-all 减少 dropping、改善吞吐。这是作者对那个时期的总结。我们固定的代码却已把生产默认值临时退回 pooled-wave，原因是 ragged 在 11 rack、watch step 后会挂住。**性能优化曾经成功，与后来因稳定性问题回退，可以同时成立。**静态 checkout 也不能替代实时作业状态。[9 月 3 日 launch note](https://openathena.ai/blog/marin-535b-launch-note/)、[本地对应 README 的回退记录](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/README.md#L19-L37)

这个案例的主线是：先用小规模实验约束大跑预期，把数据与优化器变成可计算配置，再在真实硬件上处理通信、显存与恢复，最后通过独立 checkpoint 分支提前打通后训练。

## 2. Scaling ladder：预测损失，也预测什么时候该介入

团队在 8 月 18 日创建的 issue #8435 中解释，先前 ladder 暴露出梯度范数随 token horizon 增长的问题，促成 logit z-loss 修正；已有小跑轨迹也帮助区分正常梯度变化与异常。这里引用的是审读时可见的可编辑 issue 正文，不把所有段落都认定为创建当天的版本。[公开决策记录](https://github.com/marin-community/marin/issues/8435)

源码中的第一段链路是：

```text
build_ladder_run(size)
  → _ladder_model(size)
  → batch = 1024 × rack 数
  → steps ≈ 791 × active_parameters / (batch × 4096)
  → 按 token 预算、宽度、batch 生成优化器配置
  → ArtifactStep(run=run_grug, deps=数据 artifact + 验证集)
```

五档宽度为 d768、d1024、d1536、d2048、d6144；对应 1、2、6、11、11 rack。它固定每 rack 的 token 负载，尝试保留大跑的路由压力，而不是只缩参数。生产档配置 390,251 steps、每步 11,264 条 4K 序列。纯静态乘法得到约 **18.005T tokens**；这是计划预算，不能当成已处理的数据量。小档约每 5% 进度评估，大档每 3,000 steps 评估。[launcher 的模型和预算构造](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/launch_scaling_ladder.py#L105-L222)

学习率不是把小模型的一个数复制上来。`MoeHeuristic` 根据总 tokens、宽度及每步 tokens 计算 MuonH／Adam 学习率、epsilon 和 beta2；其注释明确这是先前 sweep 的经验拟合。它是待验证的迁移规则，不能外推成任意架构都适用的定律。[优化器预算映射](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/heuristic.py#L24-L83)

还有一个会影响复现的细节：README 概括 ladder 使用同一 transport，但此快照 `_ladder_model` 为小档显式指定 ragged，hero 则继承已经回退的 pooled-wave。因而今天运行 launcher 不自动等价于复现当时那组预测实验。**工程解读：**每条拟合曲线都应绑定生成它的 commit、backend、数据和 eval 语义；“同一个脚本名”不足以定义同一实验。

这也决定了预测的使用方法：把小档拟合出的整条验证损失轨迹作为参照，同时保存实验误差和配置差异。当大跑偏离轨迹时，先定位偏离是否紧随数据切换、重启、路由变化或吞吐变化，再决定回滚模型状态还是修复系统。仅仅看到某一步训练损失升高，并不足以判断架构选择失败。公开一个终点预测，能防止事后随意解释结果；保留中间轨迹，才能在计算仍在消耗时帮助决策。

## 3. 数据配方：候选库、采样预算、实际训练量必须分开

第二段链路是 `harrier_mix_2026_08_18_data_config → _two_phase_data_config → LmDataConfig`。它采用一个带版本的数据 artifact，按 40 个 cluster × 5 个质量层级组织成 200 个 cell。JSON 明确保存每个 cell 可用 tokens 及两阶段权重；代码检查 tokenizer、store URI、两组权重是否覆盖全部 cell、总和是否为 1，以及累计重复暴露是否超过 8 epochs。参数名之外，真正重要的是这些拒绝错误配置的条件。[Harrier 配方与校验](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.py#L36-L155)

这里同时存在三种规模：本地 JSON 的候选库存量加总约 **23.106T**；混合策略定义的目标预算是 **15T + 3.75T = 18.75T**；生产 launcher 的步数预算约 **18.005T**。这些不是互相替换的模型规格。小实验在不超过 `1e23` 解析训练 FLOPs 时启用 simulated epoching，用有限实验预算模拟大预算下的重复暴露；超过该阈值则使用原始 mixture，不能说 hero 也在重复模拟一个小数据集。

两阶段切换由总 steps 的约 80% 推导，并对 mixture block 对齐；它不是看到 `15T` 常量就机械认定生产第 15T token 才切换。验证集以零训练权重接入同一配置，同时检查命名碰撞。零权重保证这些组件不被采样，却不能单独证明训练文本中不存在评估题的近重复。[阶段边界与零权重验证集](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe/launch_datakit_moe_mix.py#L295-L360)

**工程解读：**训练时长改变，学习率 horizon、数据阶段边界和重复暴露都可能改变。修改 `num_steps` 是一次配方变更；因此要重新算清这些量，再判断原来的 scaling 预测是否仍适用。

## 4. EP 的选择：通信、显存、路由和数值状态一起设计

生产模型是 48 层、宽度 6144、384 个 routed experts、每 token 选 8 个，另有 2 个 shared experts；专家宽度和 latent width 都是 3072。每 rack 用 64 张 GB200 形成 expert mesh，即 16 个四卡 worker，每卡持有 6 个 routed experts；11 rack 通过 `replica_dcn` 做复制。NVL72 是硬件系统名，本配方使用的是其中 64 卡的 EP mesh，不能直接拿 72 乘 rack 数当训练进程数。

当前 pooled-wave 把发往每个目的端的数据放入固定池，分成 3 waves；超过容量的 expert assignments 会被丢弃，sender／receiver capacity factor 都是 1.15。这给通信和 buffer 一个有界形状，代价是改变部分 token 实际获得的专家计算。ragged 试图让传输适应实际分配，却增加运行时和集体通信路径的复杂性。[真实模型及 transport 默认值](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/heuristic.py#L86-L130)

第三段链路可以直接在训练步追到：

```text
上一 step 的 pending_qb_betas → 更新 router bias
  → BF16 compute forward + CE／logit z-loss → gradients
  → 调入 optimizer state 和 FP32 master → optimizer.update
  → 生成 BF16 device weights，master／optimizer state 回 pinned host
  → 保存下一 step 的 router balancing 信息
```

默认 `z_loss_weight=1e-4`；QB 用全局 histogram 提供下一步、停止梯度的路由偏置。`_drop_metrics` 的总 / sender / receiver drop-fraction 指标以 batch × sequence × top-k × 层数为分母；额外的 `receiver_drop_fraction_of_received` 则先从分母扣除 sender drops。因此这些 drop fraction 是**专家分配比例**，不是“多少完整 token 被从语料删除”。[训练步、统计口径和 state 更新](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/train.py#L713-L882)、[训练默认值](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/hero_recipe.py#L31-L97)

host offload 不是免费显存：训练步仍需处理状态迁移，吞吐和峰值内存要一起量。此配方为 pooled-wave 保留 FP32 host master；切换到另一种权重布局还会改变 checkpoint 结构。`template_for_candidate_layout` 检查 manifest：有 master 的 checkpoint 可以把权威 FP32 master 恢复为 device 参数；无 master 的 checkpoint 恢复到要求 master 的模式会直接拒绝，代码不支持临时合成 master。**工程解读：**换 kernel 的回退预案必须同时检查 checkpoint 是否能沿反方向恢复。[恢复布局的真实判定](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/train.py#L148-L190)

## 5. 启动以后：保存实验身份，区分故障与正常重试

投产前的诊断也有明确边界：一 rack 测试沿用生产的数据、watch 和进程布局，以每卡 16 条序列保留局部负载；可用合成数据单独观察计算和内存，但它不会覆盖真实存储读取。README 特别提醒，一 rack trace 不包含十一 rack 之间的复制通信和全局 histogram reduction。因此“单 rack 能跑”只通过一个局部门槛，不能证明全集群稳定；此次 watch 后挂住的问题正说明最后一层集成验证有独立价值。[诊断与 profiling 的覆盖边界](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/README.md#L101-L155)

`trigger_hero.sh` 会记录完整 commit、dirty 状态、run ID 和 coordinator job，然后才提交 Iris。当前脚本还明确从 step 58,014 的完整 checkpoint 派生新 run ID，以启用 gate／router weight decay 续训。它保留自己的输出目录，是可追踪的配方分支；这不是后面提到的 early-cooldown RL 分支，也不能证明后者已经完成。[启动与分支脚本](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/trigger_hero.sh#L14-L55)

launcher 为生产档保留每 6,000 steps 的永久 checkpoint，并配置每小时滚动临时 checkpoint。后者在区域附近保存、只保留一份，恢复会搜索多个目录；显式指定父 checkpoint 却无法加载时应失败，避免悄悄从零开始。小时级间隔是在多 TB 保存成本与重做训练之间取舍，不保证所有故障一小时内恢复，也不包含恢复本身的停机时间。与此同时，卡住一个 step 和进程长期无进展有不同 watchdog 超时。[checkpoint、watchdog 与 eval 配置](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/launch_scaling_ladder.py#L245-L330)

运维文档把信号拆成梯度过高、连续 skipped updates、token drops、router entropy／bias、MFU、eval 退化和遥测失联。例如当前 hero 规则使用 drop 超过 7%、15 分钟至少 3 次 skipped steps 等阈值；这些是具体实验的运维规则，不是全行业的训练标准。尤其值得学的是 `run_id` 与 `execution_uid` 分开：重试沿用逻辑 run ID，但新执行有新 UID。若只按 run 聚合，就可能把上次执行的 skipped steps 加进本次，或比较恢复前后重做的评估。告警也不能一律解释成模型变坏：数据切换、配置变化、恢复本身都可能移动 loss 基线。[告警条件与 attempt 隔离](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/docs/ops/hero-run-health-alerts.md#L38-L115)

评估还有一个容易遗漏的口径：当前 launcher 报告 dropless held-out eval，禁用 capacity-limited 的当前模型评估路径；配置注释指出后者会破坏此模型规模的 ragged train step。因而训练仍可能丢部分专家分配，而报告损失来自不丢弃的评估。**工程解读：**固定评估语义有助于比较模型权重，但不能据此推断生产推理的延迟和路由行为。交接下游时应另测目标推理 backend，保存导出前后的差异，也不能把语言建模损失预测直接转换成代码执行成功率或多轮任务完成率。

## 6. 长上下文与后训练：在主跑结束前购买信息

issue 的计划先用 4K，是为了增加每批序列多样性、改善 expert balance；长上下文可能显著增加 dropping。团队提出运行约 10–20 天后，另取分支做 1–2 天 early cooldown，提前试 RL 和上下文扩展；随后如何扩到 8K、65K、262K，要依试验结果决定。这里记录的是条件性计划，本章没有核实该分支已经执行或达标。[上下文与 early-cooldown 计划](https://github.com/marin-community/marin/issues/8435)

**工程解读：**上下文长度改变的不只是 attention 计算量和位置编码，也改变每批任务组成、专家负载及通信压力。独立分支让后训练团队先测试模型导出、推理一致性、rollout 和 reward 环境；即使结果不好，也可以把问题反馈给仍在训练的主干，而不必等待数月后才发现接口或能力缺口。9 月 3 日公告确认团队已在做 Grug MoE 的 vLLM 支持、checkpoint 导出及推理一致性工作，但这不等于完整 535B 后训练配方已经公开或成功。[推理交接工作的官方说明](https://openathena.ai/blog/marin-535b-launch-note/)

## 7. 这个案例目前能教什么，哪些结果仍然没有答案

可直接学习的是一条可审查的决策链：预测和实验身份绑定；预算驱动数据与优化器；路由质量同吞吐一起评估；通信优化要能恢复；同一逻辑训练允许受控分支；评估和告警理解每次执行的边界。它们比一张最终 benchmark 表更接近实际训练工作。

截至本章审读，不能据这些材料宣布完成 18T、达到预注册损失、实现目标长上下文，或获得 frontier agent 成绩；也没有验证实时 token 计数、总成本、最终安全门槛及发布日。源码当前默认值、实际被某个 job 加载的配置、历史预测实验与最终模型，需要靠各自的 provenance 和结果 artifact 接起来。阅读下一次更新时，应继续追这条证据链，而不是用新公告覆盖旧事实。
