# SmolLM：数据、训练配方与运行环境如何一起决定模型

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/huggingface/smollm @ `a041759883ec7152d18fb985ea49be641a0bceef`  
验证范围：读取本地 YAML 与 Python 实现，核对配置和调用关系；没有安装 nanotron / datatrove / TRL，没有下载数据、权重或执行训练。

## 核心问题

从 4K 预训练进入 32K 长上下文训练，究竟改变了什么？这个仓库公开的研究配方与可独立运行的训练引擎之间有什么边界？

## 源码路径与执行链路

以下路径相对本笔记；链接固定到本地 clone 的 commit。

1. [stage1_8T.yaml](../../sources/smollm/text/pretraining/smollm3/stage1_8T.yaml)：数据路径和权重、模型结构、优化器、并行拓扑、token budget 同处一个配置。重点是 [模型至 batch 配置，L163–255](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/stage1_8T.yaml#L163-L255)。配置引用 nanotron，但本仓库不包含它的预训练主循环。
2. [long_context_4k_to_32k.yaml](../../sources/smollm/text/pretraining/smollm3/long_context_4k_to_32k.yaml)：[L206–298](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/long_context_4k_to_32k.yaml#L206-L298) 同时调整 attention backend、RoPE、CP、学习率和 batch accumulation。它是另一阶段配方，不是只改 `max_position_embeddings`。
3. [finemath-tokenize.py](../../sources/smollm/text/pretraining/continual-pretraining/finemath/finemath-tokenize.py)：[L18–62](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/continual-pretraining/finemath/finemath-tokenize.py#L18-L62) 的执行链是 `args → 数据子集表 → ParquetReader → DocumentTokenizer → SlurmPipelineExecutor.run()`。这是独立的 FineMath 持续预训练数据示例；没有证据说明它直接生成本次 SmolLM3 全部输入。
4. [text/finetuning/train.py](../../sources/smollm/text/finetuning/train.py)：[L54–122](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/finetuning/train.py#L54-L122) 用 HF 模型与数据构建 `SFTTrainer`，传入 LoRA / 可选 NF4 配置后 `trainer.train()`。这是另一个较小的动手入口，不是 SmolLM3 base 的原始训练实现。

## 已确认的机制与取舍

**配置事实。** 两阶段均为 36 层、hidden size 2048、16 个 query heads / 4 个 KV heads、共享词嵌入，`no_rope_layer: 4`、doc masking 开启。4K 阶段用 `flash_attention_2`、RoPE theta 50,000；32K 阶段改为 `llama3_ring_attention`、theta 2,000,000、CP 从 1 到 4。这里确认的是配置要求；NoPE / ring attention 内部如何执行，需要继续读对应 nanotron 分支。

**可计算的不变量。** 4K 配置：DP 192 × microbatch 3 × accumulation 1 × sequence 4096 = 2,359,296 tokens/update。32K 配置：DP 12 × microbatch 1 × accumulation 6 × sequence 32768 = 同一 token 数。TP=2；CP 不应再次计入独立数据样本数。按各并行维度乘积解释，前者配置 384 ranks，后者 96 ranks；这是配置推导，不能冒充本机验证的实际 launch 资源。

**优化器配方。** BF16 模型、FP32 gradient accumulation、AdamW、gradient clipping=1、embedding 排除 weight decay 保持一致；长上下文阶段学习率从 2e-4 降至 2e-5，并改变学习率调度。比较阶段结果时不能把收益全归于 CP 或长序列，因为数据混合与其他训练参数也改变了。

**复现边界。** YAML 保留 `/scratch`、`/fsx`、S3 resume/checkpoint 路径，以及特定集群的 evaluation / Slurm 配置。FineMath 脚本 `args.email` 在本文件 parser 中没有定义；按直接运行路径可推断会在构造 executor 参数时报属性错误，尚未实际触发。SFT 脚本默认 `push_to_hub=True`，多个布尔参数使用 `type=bool`；未来实验应先在自己的实验副本中显式关闭发布并修正参数解析，再执行。没有修改 upstream。

## 与其他项目的连接

| 关系 | 证据与含义 |
|---|---|
| datatrove：示例的直接依赖 | `finemath-tokenize.py:20–24` 导入 executor、reader、tokenizer；需要另备依赖环境。本次清单未将 datatrove 作为独立 clone。 |
| TRL / PEFT：SFT 示例直接依赖 | `train.py:9–21` 导入相关包，L91–118 实际构建 trainer / adapter。 |
| nanotron：配方运行器 | YAML 中的并行、data stage、checkpoint 配置供外部引擎使用；本次没有追踪引擎消费这些配置的代码，不能标为完整复现。 |
| OLMo-core / Marin：概念对应 | 都把数据混合、token budget 与阶段训练绑定；不是这些仓库彼此导入的证据。见 [训练连接笔记](../connections/training.md)。 |

## 动手实验

**状态：待执行。** 首个实验宜验证 batch 语义，不必训练 3B。

- 问题：保持 tokens/update，改变 sequence length / accumulation 是否保持数据采样与优化步数可比？
- 输入：固定随机种子的少量人工生成 token 序列、一个缩小模型；使用隔离的 nanotron 环境及准确版本。
- 控制变量：tokenizer、数据顺序、模型、总训练 token 数、优化器；仅调整序列长度和 accumulation，使乘积恒定。
- 指标：每步实际有效 tokens、padding 比例、梯度范数、loss、峰值显存、吞吐。
- 预期：token 计数应相等；不同序列边界改变条件上下文，loss/gradient 不要求完全相等。CP 加速与显存收益要在多卡环境另验。
- 算力：先用单卡缩小模型检查数据语义；CP 比较需多卡。未测显存需求和耗时，不能套用原始 384-rank 配置。
- 实际结果：无；本次只做配置计算与静态阅读。

## 下一步与疑问

- [ ] 固定 SmolLM3 所用 nanotron / datatrove 分支及 commit，再追踪 doc masking 的 positions 如何进入 attention。
- [ ] 对照 stage2 / stage3 的数据权重，计算各语料实际重复次数，而非仅看混合比例。
- [ ] 将上述示例的环境路径映射、布尔参数和缺失 email 修复放在自己的实验目录，保留原始 snapshot。
