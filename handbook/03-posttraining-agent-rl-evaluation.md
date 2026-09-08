# 03｜从 base model 到可靠 agent：后训练、RL 与评估闭环

[返回全流程](00-end-to-end.md) · [分布式执行](02-distributed-pretraining-operations.md) · [Marin 535B 案例](04-marin-535b-live-case-study.md)

更新：2026-09-08。本文接续预训练章节，讨论超大模型获得可用行为后的训练与交付。**公开事实**来自官方报告和本地固定源码快照；**工程综合**是据此整理的设计建议，不代表所有实验室的真实内部流程。闭源 frontier 模型的完整数据配比、reward、训练预算、算法组合与发布阈值通常不可知；这里不填造这些空白。文中没有本地训练或 benchmark 实测结果。

## 1. 接收 base checkpoint，然后按能力缺口选择分支

后训练团队接收的不应只有权重，还应包括 tokenizer、已有的交互协议、模型与并行配置、训练数据版本、基础评估、长上下文表现及数值状态。分别诊断知识覆盖、指令与工具协议、搜索 / 推理策略和环境交互，再用受控实验选择继续预训练、mid-training、示范学习或 RL。SFT / RL 也可以改善能力和策略，不能只把它们理解为表达已有知识；一次任务失败也不足以判定相关能力不存在。长上下文扩展可能安排在后训练之前或与后续阶段交织。

OLMo 3 公开了 base、mid-training、长上下文及不同后训练分支的模型流；它展示了一套可研究的完整路径，不是超大模型唯一顺序。[OLMo 3 官方说明](https://allenai.org/blog/olmo3)

| 分支 | 学习信号与主要目的 | 关键决策 |
|---|---|---|
| SFT / cold start | 示范回答或成功轨迹的监督 token loss；建立指令、工具协议、输出结构和初始探索能力 | 示例质量、覆盖度与生成来源；哪些角色/token 接收梯度 |
| 偏好优化 | 成对/排序偏好，或由偏好训练出的 reward model；塑造有用性、风格和行为约束 | DPO 等直接方法或 reward-model RL；标注者与 judge 偏差 |
| RLVR / agent RL | 执行后可验证的答案、测试、文件产物、任务状态或组合评分 | 环境是否可信、任务难度是否产生信号、成本是否可承受 |
| 拒绝采样 / 蒸馏 | 生成多个候选，验证筛选后回流 SFT，或使用教师分布 | 保留来源与筛选规则，防止只模仿格式或错误推理 |

这些分支可以交替、混合或省略。不能写成所有模型必然“先 DPO 再 RL”。DeepSeek-R1-Zero 展示了不先做 SFT 的 RL 路径；R1 则采用 cold-start 数据和多阶段训练来处理行为问题。它证明存在不同选择，不证明任意 base 都能稳定直接 RL。[DeepSeek-R1 技术报告](https://arxiv.org/abs/2501.12948)

工程上先冻结一小套开发任务，比较起点：是否遵守工具 schema、能否偶尔成功、是否频繁语言混杂或格式失效。全部失败时先改善数据、harness 或 cold start；全部成功时增加难度。不要用“多加几轮 RL”代替诊断。

## 2. 把 reward 和环境当成训练系统的组成部分

一个 agent 训练样本是“任务初态 → 动作与观察 → 终态 → 评分证据”。应版本化任务数据、容器镜像、工具 schema、依赖与网络条件、时间/资源预算、verifier 和 reward 聚合规则；复现时这些条件与模型权重同样重要。

先把能力目标拆成可检查事件：代码是否通过独立测试，表格是否有正确公式与数值，文档是否含要求内容且可解析，操作是否达到目标状态。客观检查优先；需要模型 judge 时，固定 rubric、judge 版本、抽样复核和分歧处理。格式正确可作辅助信号，却不能代替任务完成；否则模型可能优化评分代理。

要分开记录任务失败、环境故障、工具超时、模型截断和预算耗尽。环境故障直接给负奖励可能教会模型回避故障任务；全部删掉又会造成选择偏差。Verifiers 的 `open → step → close` 区分正常停止和 failed，管理 runtime 所有权，并在关闭过程中保留 artifact 与评分机会，是可直接学习的边界设计。[Verifiers 生命周期实现](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/rollout.py#L178-L540)

**reward hacking 检查**应主动尝试：修改测试/评分文件、读取答案、伪造日志或产物、利用 judge 格式偏好、用无意义长输出获得分数。将隐藏检查与 agent 可写环境隔离；以不同 verifier 或人工复核审计高 reward 样本。高 reward 只是训练系统测得的值，仍需验证是否对应真实能力。

## 3. Harness 改变训练分布，轨迹必须保留 token 来源

harness 决定模型每轮看见什么：系统提示、工具选择、错误反馈、搜索结果、上下文裁剪、记忆、子 agent 调度、重试和停止。换 harness 等于改变环境与观察分布；“同一模型”在更长预算或更强工具下得到更高分，不自动意味着参数能力更强。

训练与部署应对齐这些条件，并保留差异实验。一个严格训练轨迹至少要关联：

```text
task_id / group_id / trajectory_id / branch_id
实际输入 token、采样输出 token、逐 token rollout logprob
loss mask、采样参数及必要的 sampling mask
policy/model version、工具与环境版本
停止原因、reward 分项、评分 artifact、耗时与资源消耗
```

工具观察可作为上下文，但通常不接收 action loss；初始 prompt、模板 token 和 padding 也需正确处理。这里的 response region 可能包含多轮工具观察，不能将其全标为模型输出。重新 tokenize 最终聊天文本可能改变原来采样 token，导致 logprob 与 loss 对不上。

compaction 或子 agent 分叉尤其容易破坏这个对应关系。slime 的实现会检查前缀、重对齐或拆分 Sample，将失去可靠来源的区域 mask 掉，并避免共享生成前缀被重复训练；本次快照给每个 Sample 完整 outcome reward，不是简单按分支数平分。实际归一化还要继续追 advantage 与 loss reducer。[slime 轨迹实现](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/agent/trajectory.py#L141-L501)

## 4. 一次 GRPO 更新具体做什么

以下是**同步、outcome reward 的教学示意**，不是任一仓库的完整默认配方：

1. 从训练任务分布取 prompt，以固定行为策略 `μ` 为每题采样 G 条轨迹；执行工具和 verifier，保存原始 token/logprob 与 reward。
2. 按 task/group identity 计算相对优势，如 `A_i = (R_i − mean(R_group)) / (std(R_group)+ε)`；有些实现只减均值。归一化、长度 shaping、全同分组如何处理都必须固定。
3. learner 对相同 token 上下文重算 `log πθ`；同步示意中 `r_it = exp(log πθ(a_it|h_it) − log μ(a_it|h_it))`。保留旧概率基线，不能随着 minibatch optimizer step 一起漂移。
4. 只在有效 action token 上求 clipped surrogate，例如最大化 `min(r*A, clip(r, 1−ε_low, 1+ε_high)*A)`；裁剪边界按配方固定，可选 reference KL、熵项或其他正则。明确按 token、轨迹还是组归一化，这会改变长短回答权重。
5. 做 backward、梯度同步/裁剪、optimizer step；记录 entropy、KL、clip/drop fraction、reward 分布和有效 token。保存 checkpoint，再把完整的新 policy version 提供给 rollout。

这些步骤里，组标识比 batch 行位置可靠；负优势、padding、空样本和分叉都应有小型数值测试。把训练 step reward 上升当作成功远远不够：还需独立 holdout 的能力进步，以及相同成本下的收益。

## 5. 从同步扩展到异步：减少等待，同时控制数据变旧

长程 agent 的耗时差别很大，环境执行还大量消耗 CPU、内存、网络和 sandbox 配额。同步方案等整组/整批最慢轨迹结束；异步方案把采样、环境执行、评分、数据缓存、learner 与权重广播重叠。扩展时应分别配置这些资源，不只扩大 GPU 数。

但队列中的已完成轨迹会继续变旧。在途轨迹也可能跨多次权重更新；必须定义它是固定策略完成，还是允许分段更新，并保留可恢复的版本/概率来源。记录最老版本与陈旧度分布，限制 in-flight 数和队列长度，设置 admission、取消、丢弃与背压规则。

Prime RL 特别区分提前取消和入队后清理：前者节约计算，发送前的 stale sweep 才约束实际训练数据。它也不把 stale cancellation 当作任务答错去更新 curriculum。这是算法统计与调度语义的交点。[Prime RL 入队/陈旧度实现](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/train_sink.py#L131-L325)

off-policy 修正可使用 importance weighting、截断、拒绝采样等，但不能宣称有限轨迹、混合版本和裁剪后无代价恢复严格 on-policy。比较异步方案必须同时固定训练目标，报告有效样本/秒、wall time、丢弃比例和独立评估，而不是只比 rollout tokens/s。

## 6. 策略变旧与数值不一致是两个问题

**版本差异**来自 learner 已更新权重，而数据由较旧 policy 产生；**数值/执行差异**则可能在同版本权重下出现：训练和推理 engine、低精度量化、kernel、MoE top-k 路由、温度或截断采样分布不同。必须分别诊断，不能把所有 logprob 差异统称 staleness。

Open Instruct 的路径将更新比率 `π_current / π_old_train` 与校正比率 `π_old_train / π_rollout` 分开，并可对后者 clamp 或 mask。后一项在实际系统中可能包含版本与引擎差异，需按运行模式解释；记录两种比率及采样配置才能定位来源。截断和过滤也会改变训练分布。[Open Instruct correction/loss](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_utils.py#L410-L538)

MoE 还多了一层离散执行路径：微小 score 变化可能选出不同 expert。Miles 的 replay 保存路由 index，并按 microbatch、CP/SP 和层布局对齐后重放；它控制离散选择，而不是冻结全部概率或梯度。路由 replay、精度选择、权重更新原子性与 sampling mask 是不同检查项；任何一项都不能单独证明全链一致。[Miles replay 实现](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/utils/replay_base.py#L14-L241)

## 7. 评估与发布门槛贯穿全程

训练开始前就建立三层集合：用于调参的开发集、较少访问的 holdout、最终发布评估。按来源/时间/任务家族拆分通常比随机切题更能检验泛化；除文本匹配，还要检查同仓库代码、题目变体、答案泄漏、生成教师和工具检索是否接触测试材料。去重、语义近似搜索和人工抽检各有盲区，不能声称一次 decontamination 证明完全无污染。

Tülu 3 明确区分开发与未见评估，并公开去污染和评估流程。这种区分值得延续；反复根据 holdout 改 reward、harness 或模型，实际上会逐渐把 holdout 变成开发集。[Tülu 3 评估设计](https://allenai.org/blog/tulu-3-technical)

建议每次阶段交接和候选发布保留如下门槛，阈值由应用风险与统计不确定性决定，而不是在本文虚构统一数字：

- **能力回归：** 推理、代码、知识、语言、指令遵循、长上下文、工具使用分别评估，不能只看一个汇总分。
- **行为与安全回归：** 合理请求的完成率、过度拒绝、危险行为、工具越权、隐私泄露、提示注入与错误恢复；奖励模型本身也需审计。
- **系统回归：** timeout、崩溃、显存、首 token/整体延迟、每任务 token 与成本；使用相同任务、工具、预算和采样配置比较。
- **证据完整性：** 保存模型/tokenizer/harness/config/镜像/数据标识、原始轨迹、最终文件、verifier logs、失败样本和随机性设置。用配对任务与重复采样估计不确定性；pass@k 和更高预算结果应单列。

只有有效样本的高分会掩盖环境故障，只有平均成功率会掩盖某种语言或任务退化。发布门槛应能把候选退回数据、reward、harness、系统或数值问题的具体责任层。

## 8. 用四个源码接口把知识接起来

| 接口 | 上游产物 → 下游输入 | 首读位置 |
|---|---|---|
| Verifiers → Prime RL | Task/Harness/Runtime 执行得到 Episode/Trace → group credit、过滤、TrainingSample | `verifiers/v1/rollout.py` → `orchestrator/train_sink.py` |
| APEX/Harbor → SkyRL | Trial verifier reward + TITO tokens/masks/logprobs → GeneratorOutput → trainer | `tito_harbor_generator.py`；SkyRL 是本库范围外依赖 |
| slime agent → train backend | sampled TurnRecord → 分叉 Sample → rollout batch → actor update | `agent/trajectory.py` → `train.py` / Megatron actor |
| verl/Miles → 计算与采样后端 | 带 mask/group/logprob 的 batch 与参数版本 → engine 计算/权重更新 | verl `ray_trainer.py`；Miles `actor.py` / replay data |

这些是软件接口或对照入口，不是四套框架已经互相兼容。Prime RL 锁定的 Verifiers submodule 与本库独立 HEAD 不同；APEX 锁 Harbor 0.21.0，SkyRL 使用作者机器上的 release checkout 路径。**APEX 官方明确因许可问题未开源训练数据**，公开 eval traces 不能补成原训练集；当前只能研究配方并替换为有授权且可复现的任务数据，不能声称复现原运行。[APEX 数据说明](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/README.md#L47-L68)

## 9. 部署是下一轮数据工程的起点

上线前用真实 serving 配置重测：量化、chat template、工具 schema、context budget、路由和限流都可能改变行为。采用可回退的版本、灰度流量和明确停止条件，保留部署模型与离线候选的映射。

将授权采集的失败案例、成本异常和用户反馈送回离线分析，先归因再决定更新数据、harness、reward 或模型。用户反馈不是天然真值，线上日志也不应未经权限、隐私和污染检查就自动进入训练。完整交付物是可追溯的权重、运行协议、评估证据和回退机制；下一轮改进从这些证据开始。
