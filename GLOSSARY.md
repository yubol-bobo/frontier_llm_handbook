# 共用术语与容易混淆的接口

这些是用于串联本学习树的工作定义；各项目具体含义以对应源码笔记为准。

| 术语 | 本学习树中的含义 | 阅读锚点 |
|---|---|---|
| Harness | 包围模型的执行系统：上下文、工具、运行状态、错误处理与会话 | [Pi](notes/repositories/pi.md)、[DeepSeek](notes/repositories/deepseek-harness.md) |
| Environment / Sandbox | 任务状态、工具作用的位置及其执行隔离；二者不是同义词 | [Harbor](notes/repositories/harbor.md) |
| Verifier / Rubric | 将任务产物或轨迹映射为评分的逻辑 | [Cookbook](notes/repositories/harbor-cookbook.md)、[Verifiers](notes/repositories/verifiers.md) |
| Rollout | 使用某个策略与环境交互而采集的样本过程 | [Prime RL](notes/repositories/prime-rl.md) |
| Trajectory | 记录采样过程的消息、token、工具反馈、奖励与相关元数据 | [slime](notes/repositories/slime.md) |
| Scalarization | 把多个奖励维度变成优化器消费的标量；策略必须明确 | [实验 001](experiments/001-harbor-reward-contract/README.md) |
| Loss mask | 决定哪些 token 对某个训练目标贡献梯度，区别于 attention mask | [slime](notes/repositories/slime.md)、[Open Instruct](notes/repositories/open-instruct.md) |
| Attention mask | 决定位置之间能否互相注意；packing/文档边界会改变它 | [SmolLM](notes/repositories/smollm.md) |
| Old / rollout logprob | 计算策略比率所用的历史或采样概率；来源未必相同 | [Open Instruct](notes/repositories/open-instruct.md) |
| Policy staleness | 样本的行为策略与当前训练参数之间的版本滞后 | [Prime RL](notes/repositories/prime-rl.md)、[verl](notes/repositories/verl.md) |
| TITO | 保留实际采样 token IDs 的输入/输出链路，减少重新分词造成的漂移 | [Miles](notes/repositories/miles.md) |
| Routing replay | 在训练中复用采样时的 MoE expert 路由，处理路由不一致 | [Miles](notes/repositories/miles.md) |
| DP / FSDP / HSDP | 数据并行及参数/梯度/优化器状态分片；具体 group 组合依实现而定 | [OLMo-core](notes/repositories/olmo-core.md)、[TorchTitan](notes/repositories/torchtitan.md) |
| TP / PP / CP | 按算子张量、模型层、上下文维度切分计算 | [Megatron](notes/repositories/megatron-lm.md) |
| EP | 把不同 MoE experts 放到不同设备，伴随 token 路由通信 | [Megatron](notes/repositories/megatron-lm.md)、[DeepEP](notes/repositories/deepep.md) |
| Prefill / Decode | 分别处理输入上下文与逐 token 生成的计算阶段 | [SGLang](notes/repositories/sglang.md) |
| KV cache | 复用已计算注意力键值状态；需要分配、锁定与回收策略 | [SGLang](notes/repositories/sglang.md) |
| Grouped GEMM | 把多个形状/分组矩阵乘法交由专门内核处理，常见于 MoE | [DeepGEMM](notes/repositories/deepgemm.md) |
| Dispatch / Combine | 按 expert 分发 token，再将专家输出还原/聚合 | [DeepEP](notes/repositories/deepep.md) |
| Checkpoint | 模型/优化器/数据位置等状态的持久化产物；不必然等同可推理权重 | [OLMo-core](notes/repositories/olmo-core.md)、[Marin](notes/repositories/marin.md) |
| MFU / Throughput | 硬件计算利用与单位时间工作量；比较时需固定 shape、精度和有效 token 口径 | [TorchTitan](notes/repositories/torchtitan.md) |

## 三种“正确性”要分别检查

1. **应用语义正确：** 请求、工具反馈、任务状态及日志一致。
2. **训练语义正确：** token、mask、reward、logprob、policy version 与采样过程一致。
3. **数值与分布式正确：** 切分、归一化、精度、collective 与梯度更新一致。

一种正确性成立，不能自动证明另外两种。知识树中的跨层连接，主要围绕这些接口展开。
