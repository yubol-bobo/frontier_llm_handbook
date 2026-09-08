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
| MFU / Throughput | 模型必需 FLOPs 对应的利用率（MFU）与单位时间工作量（throughput）；比较时需固定 shape、精度和有效 token 口径 | [TorchTitan](notes/repositories/torchtitan.md) |

## 三种“正确性”要分别检查

1. **应用语义正确：** 请求、工具反馈、任务状态及日志一致。
2. **训练语义正确：** token、mask、reward、logprob、policy version 与采样过程一致。
3. **数值与分布式正确：** 切分、归一化、精度、collective 与梯度更新一致。

一种正确性成立，不能自动证明另外两种。知识树中的跨层连接，主要围绕这些接口展开。

## 中英术语对照与英文写法

两种语言使用同一组技术概念。下表统一站内写法；原论文标题、项目名称和源码标识保持原样。复合定语可按英文语法加连字符，例如 mixture-of-experts model；这不改变概念含义。

| 中文 | 标准英文 | 常用缩写 |
|---|---|---|
| 预训练 | pretraining | — |
| 后训练 | post-training | — |
| 监督微调 | supervised fine-tuning | SFT |
| 偏好优化 | preference optimization | — |
| 强化学习 | reinforcement learning | RL |
| 直接偏好优化 | direct preference optimization | DPO |
| 组相对策略优化 | group relative policy optimization | GRPO |
| 交叉熵损失 | cross-entropy loss | — |
| 损失掩码 | loss mask | — |
| 注意力掩码 | attention mask | — |
| 梯度累积 | gradient accumulation | — |
| 梯度裁剪 | gradient clipping | — |
| 激活检查点／激活重计算 | activation checkpointing / activation recomputation | — |
| 训练检查点 | training checkpoint | — |
| 数据并行 | data parallelism | DP |
| 全分片数据并行 | fully sharded data parallelism | FSDP |
| 混合分片数据并行 | hybrid sharded data parallelism | HSDP |
| 张量并行 | tensor parallelism | TP |
| 流水线并行 | pipeline parallelism | PP |
| 上下文并行 | context parallelism | CP |
| 序列并行 | sequence parallelism | SP |
| 专家并行 | expert parallelism | EP |
| 混合专家 | mixture of experts | MoE |
| 键值缓存 | key-value cache | KV cache |
| 预填充 | prefill | — |
| 逐 token 解码 | decoding | — |
| 策略采样／交互采样过程 | rollout | — |
| 交互轨迹 | trajectory | — |
| 奖励标量化 | reward scalarization | — |
| 策略滞后 | policy staleness | — |
| 模型浮点运算利用率 | model FLOPs utilization | MFU |
| 吞吐量 | throughput | — |
| 智能体执行框架 | agent harness | — |

以下区别需要特别保留：

- **CP 与 SP：** 在 Megatron 的术语中，CP 沿序列维度划分输入与各层激活，SP 是与 TP 配合的部分激活划分方式，二者不能作为同义词。见 [NVIDIA 的 CP 说明](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) 和 [并行策略总览](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。
- **FSDP 与 HSDP：** HSDP 结合分片与复制维度，具体进程组和状态所有权以实现为准。见 [PyTorch FSDP 与 HYBRID_SHARD](https://docs.pytorch.org/docs/2.14/fsdp.html)。
- **MFU 与 HFU：** MFU 是 model FLOPs utilization，关注模型必需计算对应的有效吞吐；HFU 是 hardware FLOPs utilization，可以计入重计算等额外执行。它们都不等同于设备监控中的 GPU busy 百分比。见 [PaLM §4.1 与附录 B](https://arxiv.org/html/2204.02311v5)。
- **DPO 与 GRPO：** 分别采用 [DPO 原论文](https://arxiv.org/abs/2305.18290) 和 [DeepSeekMath](https://arxiv.org/abs/2402.03300) 中的全称。GRPO 中的 group 指用于相对比较的一组样本，不应译为 GPU 进程组优化。
- **Loss mask 与 attention mask：** 不计算某个目标的 loss，不代表它作为上下文对后续预测没有影响。Rollout 是采样过程，trajectory 是相应记录；training checkpoint 与用于推理的权重导出也要分开说明。
