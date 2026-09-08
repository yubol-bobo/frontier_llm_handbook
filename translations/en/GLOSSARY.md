<a id="共用术语与容易混淆的接口"></a>

# Shared terms and easily confused interfaces

These working definitions connect the learning tree; refer to the corresponding source notes for each project's specific meaning.

| Term | Meaning in this learning tree | Reading anchor |
|---|---|---|
| Harness | The execution system surrounding a model: context, tools, runtime state, error handling, and sessions | [Pi](notes/repositories/pi.md), [DeepSeek](notes/repositories/deepseek-harness.md) |
| Environment / Sandbox | Task state, where tools act, and execution isolation; the two terms are not synonyms | [Harbor](notes/repositories/harbor.md) |
| Verifier / Rubric | Logic mapping task artifacts or trajectories to scores | [Cookbook](notes/repositories/harbor-cookbook.md), [Verifiers](notes/repositories/verifiers.md) |
| Rollout | The process of collecting samples by interacting with an environment under a particular policy | [Prime RL](notes/repositories/prime-rl.md) |
| Trajectory | Messages, tokens, tool feedback, rewards, and associated metadata recording the sampling process | [slime](notes/repositories/slime.md) |
| Scalarization | Converting multiple reward dimensions into the scalar consumed by the optimizer; the rule must be explicit | [Experiment 001](experiments/001-harbor-reward-contract/README.md) |
| Loss mask | Determines which tokens contribute gradients to a particular training objective; distinct from an attention mask | [slime](notes/repositories/slime.md), [Open Instruct](notes/repositories/open-instruct.md) |
| Attention mask | Determines whether positions can attend to one another; packing/document boundaries change it | [SmolLM](notes/repositories/smollm.md) |
| Old / rollout logprob | Historical or sampling probabilities used to compute a policy ratio; they need not have the same origin | [Open Instruct](notes/repositories/open-instruct.md) |
| Policy staleness | Version lag between a sample's behavior policy and the current training parameters | [Prime RL](notes/repositories/prime-rl.md), [verl](notes/repositories/verl.md) |
| TITO | An input/output path that preserves the actual sampled token IDs, reducing drift from retokenization | [Miles](notes/repositories/miles.md) |
| Routing replay | Reusing the sampling-time MoE expert routes during training to address routing inconsistency | [Miles](notes/repositories/miles.md) |
| DP / FSDP / HSDP | Data parallelism / fully sharded data parallelism / hybrid sharded data parallelism, including parameter/gradient/optimizer-state sharding; the specific group composition depends on the implementation | [OLMo-core](notes/repositories/olmo-core.md), [TorchTitan](notes/repositories/torchtitan.md) |
| TP / PP / CP | Tensor / pipeline / context parallelism: partitioning computation across operator tensors, model layers, or the context dimension | [Megatron](notes/repositories/megatron-lm.md) |
| EP | Expert parallelism: placing different MoE experts on different devices, with token-routing communication | [Megatron](notes/repositories/megatron-lm.md), [DeepEP](notes/repositories/deepep.md) |
| Prefill / Decode | The compute stages for processing the input context and generating tokens one at a time, respectively | [SGLang](notes/repositories/sglang.md) |
| KV cache | Reuses already computed attention key/value states; requires allocation, locking, and reclamation policies | [SGLang](notes/repositories/sglang.md) |
| Grouped GEMM | Processing multiple matrix multiplications with different shapes/groups through specialized kernels, commonly in MoE | [DeepGEMM](notes/repositories/deepgemm.md) |
| Dispatch / Combine | Sending tokens to their experts, then restoring/aggregating expert outputs | [DeepEP](notes/repositories/deepep.md) |
| Checkpoint | A persisted artifact containing state such as model/optimizer/data position; not necessarily equivalent to weights ready for inference | [OLMo-core](notes/repositories/olmo-core.md), [Marin](notes/repositories/marin.md) |
| MFU / Throughput | Model FLOPs utilization / work per unit time; comparisons require fixed shapes, precision, and effective-token definitions | [TorchTitan](notes/repositories/torchtitan.md) |

<a id="三种正确性要分别检查"></a>

## Check three kinds of correctness separately

1. **Application semantics:** Requests, tool feedback, task state, and logs are consistent.
2. **Training semantics:** Tokens, masks, rewards, log probabilities, and policy versions are consistent with the sampling process.
3. **Numerical and distributed correctness:** Partitioning, normalization, precision, collectives, and gradient updates are consistent.

Establishing one kind of correctness does not automatically prove the other two. Cross-layer connections in the knowledge tree primarily develop around these interfaces.

<a id="中英术语对照与英文写法"></a>

## Chinese–English terminology and spelling conventions

Both languages use the same technical concepts. This table standardizes the handbook's terminology; paper titles, project names, and source-code identifiers retain their original spelling. Compound modifiers may be hyphenated where English grammar requires it, as in mixture-of-experts model, without changing the concept.

| Chinese | Standard English | Common abbreviation |
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

Keep the following distinctions explicit:

- **CP and SP:** In Megatron terminology, CP partitions inputs and layer activations along the sequence dimension. SP partitions selected activations in conjunction with TP. They are not synonyms. See [NVIDIA's CP explanation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) and [parallelism overview](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).
- **FSDP and HSDP:** HSDP combines sharding and replication dimensions. The implementation determines the process groups and state ownership. See [PyTorch FSDP and HYBRID_SHARD](https://docs.pytorch.org/docs/2.14/fsdp.html).
- **MFU and HFU:** MFU means model FLOPs utilization and reflects useful throughput for the model's required computation. HFU means hardware FLOPs utilization and may count additional execution such as recomputation. Neither is the GPU-busy percentage shown by a device monitor. See [PaLM §4.1 and Appendix B](https://arxiv.org/html/2204.02311v5).
- **DPO and GRPO:** Their expansions follow the [original DPO paper](https://arxiv.org/abs/2305.18290) and [DeepSeekMath](https://arxiv.org/abs/2402.03300). The group in GRPO refers to samples used for relative comparison, not a GPU process group.
- **Loss mask and attention mask:** Excluding a target's loss does not imply that its input context cannot affect later predictions. A rollout is a sampling process, and a trajectory records it. A training checkpoint also needs to be distinguished from a weight export for inference.
