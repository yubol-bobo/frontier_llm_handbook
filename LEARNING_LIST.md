# Frontier LLM 仓库学习索引

更新：2026-09-08。20 个项目已经克隆，初读笔记以 [sources.lock.json](sources.lock.json) 中的版本为准。

默认学习顺序已经统一到 [ROADMAP.md](ROADMAP.md)：从先修、模型与梯度，经数据、预训练与系统，再进入后训练和 agent RL。详细模块在 [curriculum](curriculum/01-foundations-to-pretraining.md)，资源用途见 [RESOURCE_ATLAS](RESOURCE_ATLAS.md)。下表保留稳定登记编号，方便查找旧笔记；编号不是学习先后。

## 学习等级

- **L0 已定位**：找到项目与来源。
- **L1 源码初读**：看过一个真实实现/配置链路，能够指出输入、输出和至少一个取舍。
- **L2 机制追踪**：跨多个模块追踪正常路径和至少一个边界情况，形成可检验的解释。
- **L3 实验验证**：针对某个明确机制运行可重现的实验，并记录范围与结果。
- **L4 可复用贡献**：形成经过验证的代码、环境、评测或研究结果。对外提交另行决定。

等级针对已记录的学习范围，不代表整个仓库已掌握。维护者目前对所有项目至少完成一条 L1 初读；Pi / Harbor Cookbook / Marin 有进一步局部学习。这不是读者的课程成绩。

## 总清单

下表是原始登记索引；每项笔记含固定版本链接与具体文件入口。新读者按 ROADMAP 的 M00–M15 进入这些材料，不必依次通读 20 个仓库。

| 登记编号 | 项目 / 笔记 | 核心问题 | 下一项动手产物 |
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
| 20 | [Claude Code / Agent SDK](notes/repositories/claude-agent-sdk.md) | 复杂 harness 怎样管理上下文、权限与恢复？ | 固定 SDK 调用链、快照证据边界、CPU 故障 trace |

## 怎样把索引变成学习产物

每次只从当前模块选择一个主问题。读概念、追指定代码，再做一个能验证或推翻解释的小练习；把不确定性和版本边界写进笔记。核心路线不要求同时精读六套 RL 框架，也不要求没学梯度前先接异步 agent RL。

每模块的先修、按序核心阅读、两项练习与验收已经在 [课程路线](ROADMAP.md) 定义，避免这里维护第二套阶段顺序。现有 [训练全流程](handbook/00-end-to-end.md) 解释模型研发发生的过程；[Marin 535B](handbook/04-marin-535b-live-case-study.md) 提供真实决策例子。

自学记录使用 [学习者模板](templates/learner-progress.md)。本仓库实际完成的研究和实验仍记录在 [PROGRESS.md](PROGRESS.md)；拟做的练习不要填成已执行。
