# Frontier LLM 学习清单

更新：2026-09-08。19 个项目已经克隆，初读笔记以 [sources.lock.json](sources.lock.json) 中的版本为准。

如果当前问题是“一个超大 LLM 究竟如何从零训练出来”，先读 [全流程总览](handbook/00-end-to-end.md)，再按数据设计 → 分布式执行 → 后训练进入对应详章。[Marin 535B 案例](handbook/04-marin-535b-live-case-study.md) 把这些概念放回一个正在进行的真实运行；下表仍保留原始逐库索引，并不要求先读完所有 harness 再学习预训练。

## 学习等级

- **L0 已定位**：找到项目与来源。
- **L1 源码初读**：看过一个真实实现/配置链路，能够指出输入、输出和至少一个取舍。
- **L2 机制追踪**：跨多个模块追踪正常路径和至少一个边界情况，形成可检验的解释。
- **L3 实验验证**：针对某个明确机制运行可重现的实验，并记录范围与结果。
- **L4 可复用贡献**：形成经过验证的代码、环境、评测或研究结果。对外提交另行决定。

等级针对已记录的学习范围，不代表整个仓库已掌握。当前所有项目至少完成 L1；Pi / Harbor Cookbook 有进一步局部学习。

## 总清单

下表顺序是默认学习路线；每项笔记都含固定版本链接与具体文件入口。

| 顺序 | 项目 / 笔记 | 核心问题 | 下一项动手产物 |
|---:|---|---|---|
| 1 | [Pi](notes/repositories/pi.md) | 模型、工具、steering、上下文如何组成循环？ | 一个受控双工具事件 trace，区分完成次序和模型消息次序 |
| 2 | [DeepSeek Harness](notes/repositories/deepseek-harness.md) | 插件修改行为后，模型请求怎样由日志重建？ | 日志重建、失败尝试与恢复的对照说明 |
| 3 | [Harbor Cookbook](notes/repositories/harbor-cookbook.md) | 现实任务怎样获得可验证的奖励？ | 已有实验 001；下一步运行真实 Trial 的奖励读取 |
| 4 | [Harbor](notes/repositories/harbor.md) | 环境启动、agent、verifier、清理如何形成稳定生命周期？ | 一个有正常/超时/失败分支的 Trial 时序图 |
| 5 | [Verifiers](notes/repositories/verifiers.md) | 任务、harness、runtime、评分怎样共享接口？ | 一个小 taskset，含明确 evaluator 与 held-out cases |
| 6 | [Prime RL](notes/repositories/prime-rl.md) | rollout 如何通过编排进入 learner？ | 小模型训练 trace：tokens、reward、policy version、loss |
| 7 | [APEX Agents Recipe](notes/repositories/apex-agents-skyrl-recipe.md) | 大型知识工作任务怎样接入异步训练？ | 画出 adapter 边界，使用自有/公开任务替代未公开训练数据 |
| 8 | [SmolLM](notes/repositories/smollm.md) | 数据混合、模型结构和阶段训练怎样共同决定配方？ | 固定 token budget 的一个局部 ablation 设计 |
| 9 | [OLMo-core](notes/repositories/olmo-core.md) | 一次真实预训练怎样从数据配置进入训练生命周期？ | 追踪数据→batch→loss→checkpoint→eval |
| 10 | [Open Instruct](notes/repositories/open-instruct.md) | SFT 与 RL 的数据、概率、masking 有何不同？ | 构造 logprob/mask 张量，解释 ratio 与 correction |
| 11 | [Marin](notes/repositories/marin.md) | 怎样把模型研发变成可追踪的实验与执行图？ | 定义一个数据变体实验，追踪缓存/版本/依赖 |
| 12 | [TorchTitan](notes/repositories/torchtitan.md) | 并行、重计算、精度与 checkpoint 怎样组合？ | 一次只改一个配置，比较有效 tokens、显存、吞吐 |
| 13 | [Megatron-LM / Core](notes/repositories/megatron-lm.md) | TP/PP/CP/EP 的进程组和通信如何配合？ | 画出一个小拓扑与 MoE dispatch/combine 数据流 |
| 14 | [verl](notes/repositories/verl.md) | 算法控制与分布式计算怎样分开表达？ | 沿一个 PPO/GRPO batch 追踪 workers、数据分发与 loss |
| 15 | [slime](notes/repositories/slime.md) | agent 分叉轨迹怎样转换为有效训练样本？ | 检查共享前缀、tool mask、outcome reward 的实际分配 |
| 16 | [Miles](notes/repositories/miles.md) | token/路由/精度/版本不一致怎样影响 RL？ | 一个明确的 TITO 或 routing replay 不变量检查 |
| 17 | [SGLang](notes/repositories/sglang.md) | 请求调度、KV 复用和计算通信如何影响 rollout 成本？ | 共享前缀与无共享前缀 workload 的指标比较 |
| 18 | [DeepGEMM](notes/repositories/deepgemm.md) | 低精度、grouped GEMM、JIT 如何决定内核效率？ | 对一个支持的 GPU 跑 shape/数值误差/吞吐扫描 |
| 19 | [DeepEP](notes/repositories/deepep.md) | MoE token 怎样跨 GPU 分发与合并？ | dispatch/combine shape 与正确性、通信耗时分析 |

## 阶段 A：Agent 与任务环境（1–5）

先形成一条可解释的应用轨迹：用户输入 → 模型请求 → 工具 → 环境状态 → verifier。把模型实际看到的内容、UI 事件、持久化日志、奖励区分开。阶段完成标准是能在没有训练 GPU 的情况下，用受控例子解释一次成功、一种失败和一次恢复。

当前已完成的实践：[实验 001](experiments/001-harbor-reward-contract/README.md)。接下来验证真实环境中的奖励加载与 scalarization，再设计训练任务。

## 阶段 B：最小 Agent RL 闭环（6–7）

从 Prime RL 的小模型示例学习，而不是先运行 APEX 的大型生产配置。重点跟踪 task / trajectory / token / reward / loss / weight update 六种对象。APEX 原始训练数据未公开，配方还含特定路径和依赖 pin，应把它当架构参考，另选可获取任务来验证机制。

进入此阶段需要 Linux/CUDA 等兼容环境、模型数据与明确的资源预算。默认 Reverse Text 示例面向两张 GPU；本次没有安装或启动。完成标准是能指出一条 rollout 的 reward 怎样进入一次参数更新，并确认评测任务不参与训练。

## 阶段 C：模型研发与数据（8–11）

SmolLM 读配方与消融，OLMo-core 读正式模型训练路径，Open Instruct 读后训练，Marin 读实验组织与过程知识。外部配套资料：

- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)：训练研究决策。
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)：分布式计算与通信。关联的 nanotron 暂作外部节点，没有计入 19 个 clone。
- SmolLM 的 nanotron/datatrove 引用表示运行依赖或数据示例来源；需要时先固定相关版本，再进入对应代码。

完成标准是能给出一次 ablation 的假设、控制变量、token budget、评测集与判断标准。小模型结论不自动外推到更大规模。

## 阶段 D：分布式训练（12–13）

TorchTitan 与 Megatron 对照学习。先弄清张量 shape、每个 rank 持有什么，再讨论并行名词。追踪 forward/backward、gradient accumulation、collective、optimizer 与 checkpoint。单机笔记可先做拓扑推演；吞吐/通信结论须在实际多卡硬件上验证。

Megatron-LM 包含训练入口和脚本；Megatron Core 是可组合的训练构件库。下游依赖 Core 不等于运行顶层 pretrain 脚本。学习它与 slime/Miles/verl 的连接时，要追踪 backend 调用与版本 pin。

## 阶段 E：RL 系统正确性（14–16）

在理解一个训练器后比较其他框架，避免同时混合它们的配置语义。共用比较题：old logprob 来自哪里、模型版本何时变化、tool tokens 怎样 mask、分叉共享前缀怎样处理、长尾 rollout 怎样调度、更新失败如何恢复。完成标准是选择一个问题，提供三套实现的证据对照，而非罗列支持的算法名称。

## 阶段 F：Serving、MoE 与 GPU（17–19）

从 SGLang 的 scheduler/cache 下探到 DeepGEMM 的计算与 DeepEP 的通信。记录硬件、precision、shape、batch、并行拓扑后再比较性能。DeepEP 版本间接口存在差异；Megatron 的 `Buffer` 接入与本次 DeepEP V2 `ElasticBuffer` 不能未经检查直接拼接。

## 每阶段结束交付什么

- 一份带源码证据的笔记，明确哪些问题仍未解决。
- 一张执行链路或对象关系图；引用的是同一个固定源码版本。
- 一个可运行实验或完整实验设计；未运行明确标注。
- 至少一条跨仓库连接，标明关系类型与实际版本约束。
- 在 [PROGRESS.md](PROGRESS.md) 中留下下一次的具体入口。
