# Training 连接：从数据身份到梯度，再到 MoE 通信

日期：2026-09-08。证据等级：L1，读取本地固定 commit 的源码；没有完成跨库集成运行。各仓库完整 SHA 与验证范围见对应笔记。

## 有源码证据的真实连接

| 起点 → 终点 | 关系类型 | 固定源码证据 | 可以推出什么 / 不能推出什么 |
|---|---|---|---|
| Open Instruct → OLMo-core | 直接依赖，版本固定 | [pyproject.toml L7–74](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/pyproject.toml#L7-L74) 声明 `ai2-olmo-core`，`tool.uv.sources` pin `fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d`；[SFT imports L37–48](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/olmo_core_finetune.py#L37-L48) 实际使用 trainer / data / optimizer。 | 能沿 SFT caller 学习 pretraining infra 的复用；本地 OLMo-core HEAD 是 `92870a33…`，不能直接替代 pin 后声称兼容。 |
| Megatron-LM → DeepEP | 可选通信后端 | [Flex dispatcher L1835–1843](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/token_dispatcher.py#L1835-L1843) 选择 `_DeepepManager`；[fused_a2a import L11–17](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/fused_a2a.py#L11-L17) 导入 `deep_ep.Buffer`。 | 它是实质的 Python/CUDA 通信接口连接，不是因为两项目都谈 MoE 而联想；并非所有 Megatron MoE 都必须安装 DeepEP。独立 clones 的 HEAD 组合尚未编译验证。 |
| Marin → monorepo 内 Levanter | 直接训练依赖 | [experiment/train.py L30–50](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/experiment/train.py#L30-L50) 导入 Levanter config；[L192–241](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/experiment/train.py#L192-L241) 构造训练配置和 ArtifactStep。 | Marin 可追踪实验 DAG，Levanter 承担内层训练；不是两个名字指同一层。未深入 JAX 内层 optimizer loop。 |
| SmolLM FineMath 示例 → datatrove | 示例的直接依赖；外部 companion | [finemath-tokenize.py L20–62](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/continual-pretraining/finemath/finemath-tokenize.py#L20-L62) 用 ParquetReader / DocumentTokenizer / SlurmPipelineExecutor。 | 能学习原始文本变为训练 tokens 的模块界面；本轮 19 库没有独立 clone datatrove，不能说其实现也已读完。 |
| SmolLM SFT 示例 → TRL / PEFT | 示例的直接依赖；外部 companion | [train.py L9–21](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/finetuning/train.py#L9-L21) 导入，L54–122 构造 LoRA / SFTTrainer。 | SFT 示例可作为缩小练习；不是 SmolLM3 base 的原始预训练循环。 |

这里未列的 RL→Megatron 边，由 [verl](../repositories/verl.md)、[slime](../repositories/slime.md)、[miles](../repositories/miles.md) 各自源码笔记证明。不要因为把仓库放在同一目录就视为已完成适配。

## 值得对照的机制，均不是软件依赖边

| 学习问题 | 两端证据 | 需要保留的差异 |
|---|---|---|
| 一个 optimizer step 有多少有效 tokens？ | [SmolLM](../repositories/smollm.md) 配方固定 tokens/update；[OLMo-core](../repositories/olmo-core.md) rank batch loss 分母；[TorchTitan](../repositories/torchtitan.md) global_valid_tokens；[Open Instruct](../repositories/open-instruct.md) response mask / accumulation token counts。 | 预训练的 token packing、SFT 的 label mask、RL 的 response mask 不是同一种过滤。DP/SP/CP 也不能重复计算为额外独立样本。 |
| 算法如何影响系统排程？ | TorchTitan 的 AC→compile→FSDP 顺序；OLMo-core 的最后 microbatch 同步；Megatron 的 route→dispatch→expert→combine；Open Instruct 的 weight sync。 | 开启 CUDA graphs、异步 generation 或 MoE capacity 会增加约束；不能将优化开关任意重排。 |
| 什么才算同一个可复现实验？ | Marin 的 name/version + config fingerprint + provenance；SmolLM / OLMo-core 的正式配置；Open Instruct 的 dependency pin。 | 配置公开、代码已 clone、数据可访问、环境可构建、数值复现是不同证据等级。配置 fingerprint 不是完整源代码/权重 hash。 |
| 更高吞吐会不会改变训练目标？ | Open Instruct 的 new/old ratio 与 train/infer rho；Megatron 的 capacity dropping；OLMo-core 的 instance_mask 特殊分母。 | 吞吐比较必须同步检查有效样本/有效 token、drop rate、概率与数值差异，不能只比较 tokens/s。 |

## 一个可持续扩充的学习链

1. 先读 [SmolLM](../repositories/smollm.md) 配方，手算 DP×microbatch×accumulation×sequence 的 tokens/update，并标出外部路径与引擎版本。
2. 在 [OLMo-core](../repositories/olmo-core.md) 或 [TorchTitan](../repositories/torchtitan.md) 跟到一次真实 optimizer step；用自己的微型固定 batch 检查 masking、分母与梯度累积。此处实验均待执行。
3. 用 [Marin](../repositories/marin.md) 的身份/依赖思路整理自己的输入、代码 SHA、环境、实验配置和输出，而不是只存最终 loss 截图。
4. 进入 [Open Instruct](../repositories/open-instruct.md)，把 rollout engine、old/new policy、response mask、weight version 加入同一数据流。
5. 需要模型/集群规模时再深入 [Megatron-LM](../repositories/megatron-lm.md) → [DeepEP](../repositories/deepep.md)，将数据流具体落到 token dispatch、buffers 与 collective。

每新增一条真实依赖边，保存 caller 的固定源码位置和版本约束；每新增一条概念对应，写清“共同问题”和“不能直接迁移的差异”。下一轮 L2 应收窄到单个不变量或跨模块协议；只有实际运行并保存配置、日志、指标后才能升为 L3。
