# 后训练与 agent：M10–M15

2026-09-08。面向有基础编程能力、没有 frontier lab 经历的学习者，接续基础模型、数据、预训练和 serving 模块，训练解释、实现与审计一段完整链路的能力。

推荐 **M10 → M11 → M12 → M13 → M14 综合验收 → M15**。M14 的划分、基线和失败分类从 M03 后开始使用，每阶段回访。M12 的纯 harness 阅读可在 M01 后提前，但训练整合先修 M11；不能用“先运行 Pi，再接异步 RL”替代概率和同步闭环基础。

每节按序读三组核心材料，选读不影响通关。源码为固定快照，commit 与行号见笔记；新增外部入口已联网核实，未克隆或安装。**练习均待执行；工时统一指核心阅读＋CPU 练习的人力规划，非实测，GPU 升级另计。** CPU 优先用自有小程序／合成数据，不承诺上游完整导入链支持 CPU；GPU 需核对显存、平台、依赖锁，另计下载与训练墙钟时间。

<a id="m10"></a>

## M10｜SFT、偏好优化与 reward model：先理解学习信号

**先修与跳过诊断：** M02、M03、M04。标出多轮对话的 SFT token，解释输入／标签位移，以及“A 胜过 B”为何不证明 A 正确。不能独立完成则回访基础；能完成可压缩入门阅读，仍交审计和目标函数实验。

**细目：** 按能力缺口选择继续预训练、mid-training、SFT、偏好或 RL，不固定先 DPO 后 RL。学习 chat template、assistant-only mask、packing、截断；chosen／rejected 的相同任务条件、标注噪声、长度偏好；标量 reward model、规则 verifier、过程评分与监控 metric 的区别；reference policy，以及 pairwise accuracy／reward margin 的局限。

**核心阅读，依次完成：**

1. 读 [后训练手册](../handbook/03-posttraining-agent-rl-evaluation.md) 前两节、[InstructGPT 原论文](https://arxiv.org/abs/2203.02155) 的数据流程、[DPO 原论文](https://arxiv.org/abs/2305.18290) 的推导。画出独立 reward model 与直接偏好优化的分支，勿把历史配方当今日闭源内部事实。
2. 读 [Open Instruct 笔记](../notes/repositories/open-instruct.md)，追 [olmo_core_finetune.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/olmo_core_finetune.py#L71) 的 token／mask／metadata 缓存，再追 [dpo_utils.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/dpo_utils.py#L525) 的 `_get_batch_logps`、`dpo_loss`，比较求和与平均。
3. 读 [reward_modeling.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/reward_modeling.py#L351) 的单输出头和 351–357 行 loss。对照 DPO detached implicit rewards 的来源与用途。

**选读：** OLMo 3、Tülu 公开配方，比较阶段顺序。

**推导与图：** 写 `L_SFT=-Σmask·logpθ/Σmask`，注明 token 平均选择；从 `P(chosen>rejected)=sigmoid(r_chosen-r_rejected)` 推出 pairwise loss，再替换为 DPO policy／reference 对数概率差，标明 beta 和冻结量。

**练习 A：数据到标签。** CPU 自建 12 条短对话，含多轮、工具、超长和空答案；输出 token 标签、截断计数。交换一条偏好对的 prompt，检查能否发现。缺 tokenizer 时用整数 fixture。GPU 升级仅过拟合这批样本，观察目标概率，不计作泛化成绩。

**练习 B：目标函数对照。** CPU 手算／有限差分／自动微分计算三种 loss；固定概率与 mask，改变 padding、长度、标签和 beta，保存梯度。GPU 升级用相同划分比较 SFT 与一个偏好分支，记录预算、长度、评估和重复运行差异，不要求后者必胜。

**验收与误解：** 为什么 reward 全部加常数不改变上述 pairwise loss？把序列 logprob 求和改成平均是否只是数值缩放？低 SFT loss、较高偏好准确率为何仍可能伴随事实性下降？“DPO 没有显式 reward model”不等于没有偏好数据假设。

**交付物：** `m10/data-audit.md`、标签可视表、三个 loss 的独立小实现与对照记录。**工时：** CPU 主线 12–20 小时；GPU 升级另计 8–16 小时实验与排错。**下一步：** 能解释每个梯度来源后进入 M11，同时按 M14 固定评估划分。

<a id="m11"></a>

## M11｜RL 数学与同步最小闭环：先让一次更新正确

**先修与跳过诊断：** M10、M09 serving 概念。手算二选一策略的期望 reward 和梯度，解释不可微 verifier 为何不能直接反传。失败则补概率／链式法则；有 RL 经验仍需解释 response mask 和 prompt group。

**细目：** 状态、动作、策略、trajectory、回报；score-function gradient、baseline、advantage、critic／GAE；PPO 的 old policy、clipping、reference KL 与多次更新；GRPO 同题组、outcome reward、组内 baseline。RLVR 指奖励来源，不保证稳定。明确全组同分、单样本组、长度归一化和截断。

**核心阅读，依次完成：**

1. 读 [PPO 原论文](https://arxiv.org/abs/1707.06347) 的 surrogate objective，再读 [DeepSeekMath 原论文](https://arxiv.org/abs/2402.03300) 的 GRPO。先辨认每个期望由什么分布采样；公式里的 old policy 不等于固定 reference。
2. 读 [verl 笔记](../notes/repositories/verl.md)，对照 [core_algos.py](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/core_algos.py#L268) 的 `compute_grpo_outcome_advantage`：按 uid 分组、逐 token reward 求和、可选 std 归一化、特殊组处理。用真实函数行为修正自己的公式笔记。
3. 沿 [ray_trainer.py](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/ray_trainer.py#L1405) 的 `RayPPOTrainer.fit` 追 batch，再读 [slime/train.py](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/train.py)。标出生成、评分、logprob、更新、保存、同步的屏障；async 函数名不证明算法完全异步。

**选读：** [Prime RL 的 GRPO 实现](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/algo/grpo.py#L16) 在此快照使用减组均值，供比较归一化口径；暂不展开其异步调度。

**推导与图：** 推导 `∇E[R]=E[(R-b)Σ∇logπ]`，说明 baseline 条件。写 `Ai=(Ri-mean(R))/(std(R)+ε)`，注明 std 可关；令 `rit=exp(logπθ-logπold)`，画 `min(rit·Ai,clip(rit,1-ε,1+ε)·Ai)` 的正负优势曲线。loss 取负后按 mask／分母聚合，可另加 KL。画出 behavior、old、reference、current 四个角色。

**练习 A：组与 mask。** CPU 构造三题各四条，含全零奖励、不同长度；打乱行序保留 uid，再换错 uid。检查 advantage、padding、分母，补 K=1 与 verl 对照。GPU 升级比较 batch balancing 的有效 tokens 和更新，不把 CPU 结果外推为吞吐。

**练习 B：同步闭环。** CPU 写小词表策略，在短序列任务上执行：冻结采样策略→每题 K 条→verifier→advantage→更新→同步。固定种子，保存失败轨迹、KL、有效样本数，验证梯度方向。GPU 升级换小语言模型，保留协议，暂不接多轮工具、异步或 MoE。

**验收与误解：** 为什么重排 batch 不能丢 uid？优势为零是否代表模型已经解决问题？多次更新时为什么 old logprob 不能跟着当前参数移动？增加生成长度让 reward 上升，可能是能力、预算还是评分漏洞？clip 不是“严格保证策略没有漂移”。

**交付物：** `m11/one-update.md`、两张公式图、CPU 同步 trainer、逐 batch 审计日志。**工时：** 主线 18–30 小时；GPU 升级另计 12–24 小时。**下一步：** 能从一条回答追到一次权重更新后，再进入 M12。

<a id="m12"></a>

## M12｜任务、harness、verifier 与 trajectory：把互动变成可信数据

**先修与跳过诊断：** 阅读先修 M01，训练整合先修 M11。解释工具结果为何影响动作却通常不直接进 policy loss；能区分文本日志、请求、真实 token 和评分产物，才可略读概念。

**细目：** Task 目标、Harness 决策、Runtime 资源、Verifier 证据；生命周期、所有权、预算、超时／重试、工具权限、artifact 和离线评分。轨迹记录 task／group／branch ID、模型版本、实际 token IDs、logprobs、训练 mask、终止原因、采样参数／mask。压缩、fork、子 agent、重写可能打断前缀连续性。

**核心阅读，依次完成：**

1. 读 [ReAct 原论文](https://arxiv.org/abs/2210.03629) 建立交互概念，再读 [Verifiers 笔记](../notes/repositories/verifiers.md)；追 [v1/rollout.py](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/rollout.py#L178) 的生命周期及 [v1/task.py](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/task.py#L198) 的 `score`。这一层不是 optimizer。
2. 读 [slime 笔记](../notes/repositories/slime.md) 与 [trajectory.py](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/agent/trajectory.py#L283) 的 `record_turn`、`_split_chain_into_builders`、`to_sample`。共享前缀只训练一次，各 Sample 获得完整 reward；不是旧解释的 reward/K。
3. 对照 [APEX 笔记](../notes/repositories/apex-agents-skyrl-recipe.md) 与 [agents/tito.py](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/apex_agents_skyrl_recipe/agents/tito.py#L148)：追加 token、工具 mask=0、停止 token／模板边界。训练数据未公开、SkyRL 为外部依赖，读 glue code 不等于复现。

**选读：** [DeepSeek Harness 笔记](../notes/repositories/deepseek-harness.md) 的请求可重建约束，以及 [Pi 笔记](../notes/repositories/pi.md) 的上下文投影。此时比较 harness 才有训练数据正确性的判断依据。

**画图：** 画双泳道图：上层 Task→Harness→工具→Verifier，下层 request tokens→sampled tokens→mask/logprob→TrainingSample。给一次 compaction 标出哪些旧 token 仅作新上下文，哪些仍有可靠采样来源；解释为什么重新 tokenize 最终聊天文本不能恢复全部概率信息。

**练习 A：无模型任务。** CPU 脚本 agent 修改临时文本，评分器只读产物。注入正常、超时、缺失产物、评分故障，分别记状态，不统一写零分。关闭 runtime 重评分，列出缺失证据。GPU 升级仅将脚本换小模型，保持任务和预算。

**练习 B：轨迹边界。** CPU 整数 fixture 覆盖追加、工具、回答重写、早期前缀变化、共享分支；输出 tokens／mask／logprobs 和训练次数。模型升级检查真实 tokenizer 的两轮特殊 token；服务须提供匹配采样 token 的概率，普通文本输出不足以验证 RL。

**验收与误解：** response region 为何也能包含 mask=0？borrowed runtime 应由谁释放？offline judge 缺少现场文件该如何报告？文本可重放不等于 token 概率可重放；框架有 sandbox 选项不等于 verifier 不可被 agent 干扰。

**交付物：** `m12/task-contract.md`、状态分类表、轨迹 fixture、数据 schema 与可重评分 artifacts。**工时：** 主线 16–28 小时；真实模型／沙箱升级另计 10–20 小时。**下一步：** 先将一个任务接回 M11 同步循环，再研究 M13 的并发。


**Claude Code 进阶单元：** 在 Pi 初读之后进入 [完整讲解](../handbook/05-claude-code-harness.md) 和 [固定 SDK / 历史快照](../notes/repositories/claude-agent-sdk.md)，按执行循环 → 上下文 → 权限 → 恢复 → 评估学习。运行 [CPU 状态机实验](../experiments/harness-state-machine/README.md)，交付拒绝调用、预算耗尽和未知执行结果三种 trace，再讨论哪些字段必须进入 M13 的训练轨迹。

<a id="m13"></a>

## M13｜异步 agent RL 与一致性：吞吐增长不能掩盖训练语义变化

**先修与跳过诊断：** M11、M12、M08、M09。画 trainer 更新两次而慢任务未返回的时间线，指出行为／训练版本和缓存过期。若统称“浮点误差”，回访同步闭环；有通信经验仍需验收 loss ratio。

**细目：** straggler、流水线、异步；backpressure、group admission、取消／失败、排队老化、广播与恢复。分开三类差异：版本滞后；同版本 kernel／精度不一致；温度、top-p/top-k 改变采样分布。再学 MoE 离散路由、packing／CP／SP、routing replay，指标包括有效 tokens／秒和达到质量目标的时间。

**核心阅读，依次完成：**

1. 读 [Prime RL 笔记](../notes/repositories/prime-rl.md)，沿 [train_sink.py](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/train_sink.py#L182) 的组完成、`_drop_stale` 到 [orchestrator.py](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/orchestrator.py#L513) 的版本门控与 batch 发送。源码把 Episode、取消和 dispatch failure 区分处理。
2. 回读 [Open Instruct 笔记](../notes/repositories/open-instruct.md)，追 [grpo_utils.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_utils.py#L410) 的 `compute_rho_correction`、`compute_grpo_loss`、`perform_weight_sync`。分别标出 `new/old` 更新比率和 `old_train/infer` 修正；裁剪、拒绝和截断会带来偏差取舍。
3. 读 [Miles 笔记](../notes/repositories/miles.md)，追 [replay_base.py](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/utils/replay_base.py#L14)、[replay_data.py](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/training_utils/replay_data.py#L30) 和训练 actor 的 replay 阶段。它重放离散 index，当前 scores 仍参与计算；这不是冻结旧 logits。

**选读：** [IMPALA 原论文](https://arxiv.org/abs/1802.01561) 学 actor–learner 分离，不把其 V-trace 当上述框架的实现证明；DeepEP／DeepGEMM 笔记补通信与矩阵计算。

**推导与图：** 标明四种策略的来源，画版本推进与队列老化；给重要性权重写出分子／分母及 support 假设，解释 clipping 降低极端权重为何不保证无偏。

**练习 A：事件模拟器。** CPU 模拟 20 个任务的延迟、失败、取消、版本和队列；固定 reward 改调度，比较完成量、过期率、有效组和任务分布，检查慢任务是否被排除。GPU 升级用真实延迟，固定有效 token 预算比较时间与质量。

**练习 B：拆开不一致。** CPU 概率表／top-k fixture 分别扰动版本、logprob、support 和 packing，验证检测器。GPU 升级先测同权重／tokens／采样配置的概率差，再独立改变精度和 replay；报告路由重合、mask、loss。Miles Replay 使用 CUDA／pinned memory，CPU 替代不算执行该实现。

**验收与误解：** 为什么入队合格的样本出队仍会过期？取消算不算任务失败？同版本是否足以保证概率一致？routing replay 能否修复错的 tokenizer 或陈旧权重？权重同步异常时如何避免 actor 永久暂停？“GPU 利用率高”不等于学习效率高。

**交付物：** `m13/version-contract.md`、事件模拟器、概率差异矩阵、一次故障恢复设计。**工时：** CPU 主线 20–35 小时；多 GPU／MoE 升级另计 20–40 小时以上。**下一步：** 将 admission、版本和有效样本统计纳入 M14 报告，而非另放一张吞吐图。

<a id="m14"></a>

## M14｜独立评估、安全、发布与反馈：先定义怎样相信结果

**先修与跳过诊断：** M03 后开始，每阶段回访，最后综合。面对“提升 3 点”，能否询问样本、预算、方差、模板、版本和调参使用？不能则先做 A；有经验仍须审查 reward 与验收的共享漏洞。

**细目：** 分离训练、开发、冻结 holdout、发布回归集；按来源／模板／仓库划分，检查近重复。固定模型、tokenizer、harness、权限、预算、重试、judge、镜像；明确 pass@1、成本和延迟的统计单位。记录区间、失败、缺失、多次挑 checkpoint 偏差；分别设事实性、成功率、鲁棒性、安全、成本门槛。

**核心阅读，依次完成：**

1. 读 [HELM 原论文](https://arxiv.org/abs/2211.09110) 的多指标方法与 [后训练手册](../handbook/03-posttraining-agent-rl-evaluation.md) 的评估／交付。写覆盖表，注明未测能力；课程门槛不是闭源实验室标准。
2. 外部补课入口：[lm-evaluation-harness 官方 task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)，用于学习模型评估任务配置；[Inspect Tasks](https://inspect.aisi.org.uk/tasks.html) 与 [Scorers](https://inspect.aisi.org.uk/scorers.html)，用于追 dataset、solver、scorer 的边界。两者尚未加入本地源码范围，先读接口，不把当前网页视为作者项目锁定版本。
3. 读 [Harbor 笔记](../notes/repositories/harbor.md)，追 [trial.py](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/trial/trial.py) 的 shared／separate verifier 与 [verifier.py](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/verifier/verifier.py#L166) 的 reward 解析；对照 Verifiers 的 runtime-only scoring。返回合法数值与正确衡量任务是两层保证。

**选读：** [Marin 535B 案例](../handbook/04-marin-535b-live-case-study.md) 学习预测、版本与运行告警如何分开。对 agent 安全，先研究本地临时环境的权限和评分隔离，不需要执行真实外部危险操作。

**画图与推导：** 画 reward→独立评估→候选→灰度→监控→审查→新数据版本。计算成功率区间；同题多采样按任务聚类重采样，解释相关性为何影响区间。记录模型／judge 的随机种子和重复次数。

**练习 A：评估审判。** CPU 构造两组预测，加入长度优势、重复、缺失、只提升开发集的候选。计算总体、分层和预算匹配结果，写发布／补证／拒绝理由。GPU 升级换真实 checkpoint 输出，保持冻结协议，不为未运行模型填分数。

**练习 B：评分与安全回归。** CPU 脚本尝试改可见 reward 文件、写错产物路径、返回非法内容，检查独立评分。用临时无害 fixture 测越权请求、提示注入和秘密占位符泄漏。GPU 升级换模型，同时测正常任务误拒绝率。

**验收与误解：** 能否从原始 artifacts 重算分数？缺失评分如何进入分母？judge 与训练 reward 共享模型和提示有什么风险？多次观察 holdout 后它是否仍适合最终验收？平均分提升不能自动抵消严重安全回归，也不能证明所有未测领域安全。

**交付物：** `m14/eval-protocol.md`、冻结任务清单、结果与区间、失败簇、模型卡草案、回滚条件和数据反馈审批点。生产反馈须去除不必要的敏感信息、审查来源并版本化，不自动回灌训练。**工时：** 主线 16–28 小时；GPU 升级另计 10–20 小时。**下一步：** 先冻结 capstone 的成功标准，再开始 M15。

<a id="m15"></a>

## M15｜综合 capstone：交付可复查成果，选择与资源相符的范围

**先修与跳过诊断：** M01–M14 相关模块。能解释数据、目标、训练／推理、恢复、评估；一页写清问题、假设、最小系统、预算、反证条件。若只有榜单或演示目标，回 M14；capstone 必须交实物，不能靠阅读跳过。

**细目：** 连接任务、数据、token／mask、更新、导出、评估、恢复；区分依赖与借鉴，记录变更。有限预算优先验证一个机制，未跑层级写明假设。发现方法无效、复现缺口或错误评估也可成为有证据的成果。

**核心阅读，依次完成：**

1. 回读 [端到端手册](../handbook/00-end-to-end.md) 与 [Marin 实战案例](../handbook/04-marin-535b-live-case-study.md)，列出“完整大跑需要、此次缩小项目省略”的环节，说明为什么省略不会破坏当前研究问题。
2. 读 [RL 连接笔记](../notes/connections/rl.md) 与 [端到端连接](../notes/connections/end-to-end.md)，从所选项目笔记提取实际 commit、版本锁与数据限制；APEX 的未公开训练数据、Prime RL 的 Verifiers gitlink、Open Instruct 的 OLMo-core pin 都不能靠相邻目录自动补齐。
3. 回到 M10／M11／M14 的原论文及所选一条源码链，建立“主张→公式／函数→配置→实验→artifact”索引。最多选择一个主训练框架、一个任务 runtime、一个评估入口，先完成可解释的系统再增加组合。

**选读：** 本库其余框架只用于一个明确比较问题，例如 token mask、队列准入或重放协议；不把“同时接了很多框架”当成果。

**三选一 capstone：**

| 路线 | 必须完成的核心工作 | CPU 替代与 GPU 升级 | 明确不能声称 |
|---|---|---|---|
| A：无 GPU 证据审计 | 一条 recipe 的依赖／版本／数据流；三条主张追到实现，两项反证检查，一个边界 fixture | CPU 审读和状态实验；GPU 验证选定边界 | 静态阅读不是训练复现，未知数据不能补造 |
| B：小模型端到端 | 随机初始化小 decoder→极小预训练→保留 base→SFT→偏好或同步 RL→导出／评估／恢复 | CPU 字符模型；GPU 扩模型与 tokenizer。从现成权重开始须标“后训练端到端” | 玩具任务不是 frontier 能力，加载权重不是从零预训练 |
| C：系统／数据契约改进 | 选 staleness、mask、评分隔离、checkpoint 或 replay；构造问题、局部修复、回归检查 | CPU fixture；GPU 才测真实开销，匹配有效数据和质量预算 | 模拟时间不是真实吞吐，更快不一定训练更好 |

**画图：** 每条路线都交一张完整系统图和一张“本项目实际验证范围”图；未运行的组件使用虚线，并标明是外部服务、未取得的数据还是概念假设。

**练习 A：设计审查。** 实现前冻结基线、变量、指标、失败分类和成本。CPU 合成任务须解释与真实系统的对应；GPU 升级估计显存与停止条件。请读者复述如何推翻结论，无同伴可隔一天自审。

**练习 B：复查与故障演练。** 在干净输出目录凭记录重算结果；注入中断、过期样本或无效 reward。GPU 升级核对恢复后的 step、随机状态、优化器、权重版本；跨精度／并行配置设误差容限，不先假定逐位一致。

**验收问题：** 哪个证据能推翻结论？换一种数据／硬件还成立吗？哪些组件只读过，哪些实际运行？

**Rubric，满分 100：** 证据／版本 25；数学／token／状态正确 25；基线与对照设计 20；失败与外推边界 15；可复查 artifacts 15。规划及格 75 且每项过半；伪造执行、关键分母或泄漏未解释则不验收。评项目而非“frontier 专家”资格，不要求正向收益。

**交付物：** `m15/project.md`、实现、配置／环境锁、原始结果、复算脚本、失败案例、决策日志及演示。**工时：** 核心阅读＋CPU 路线 A 24–45、B 50–100、C 35–70 小时；GPU 升级人力、墙钟和费用另算。**下一步：** 沿一个已暴露问题继续复现或贡献，从解释边界进步到负责子系统，持续接受外部复查。
