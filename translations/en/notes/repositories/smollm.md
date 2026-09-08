<a id="smollm数据训练配方与运行环境如何一起决定模型"></a>

# SmolLM: How data, training recipes, and runtime environments jointly determine a model

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/huggingface/smollm @ `a041759883ec7152d18fb985ea49be641a0bceef`  
Verification scope: Read local YAML and Python implementations and checked configurations and call relationships; did not install nanotron / datatrove / TRL, download data or weights, or execute training.

<a id="核心问题"></a>

## Core question

What actually changes when moving from 4K pretraining to 32K long-context training? Where is the boundary between this repository's public research recipes and an independently runnable training engine?

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

The following paths are relative to this note; links are pinned to the local clone's commit.

1. [stage1_8T.yaml](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/stage1_8T.yaml): data paths and weights, model architecture, optimizer, parallel topology, and token budget share one configuration. The focus is [model through batch configuration, L163–255](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/stage1_8T.yaml#L163-L255). The configuration references nanotron, but this repository does not contain its pretraining main loop.
2. [long_context_4k_to_32k.yaml](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/long_context_4k_to_32k.yaml): [L206–298](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/long_context_4k_to_32k.yaml#L206-L298) changes the attention backend, RoPE, CP, learning rate, and batch accumulation together. It is a recipe for another stage, beyond changing only `max_position_embeddings`.
3. [finemath-tokenize.py](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/continual-pretraining/finemath/finemath-tokenize.py): the execution chain at [L18–62](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/continual-pretraining/finemath/finemath-tokenize.py#L18-L62) is `args → dataset subset table → ParquetReader → DocumentTokenizer → SlurmPipelineExecutor.run()`. This is an independent data example for FineMath continued pretraining; there is no evidence that it directly produces all SmolLM3 inputs in this study.
4. [text/finetuning/train.py](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/finetuning/train.py): [L54–122](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/finetuning/train.py#L54-L122) constructs `SFTTrainer` from HF models and data, passes LoRA / optional NF4 configuration, and calls `trainer.train()`. This is another smaller hands-on entry point, not the original training implementation for SmolLM3 base.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

**Configuration facts.** Both stages have 36 layers, hidden size 2048, 16 query heads / 4 KV heads, tied token embeddings, `no_rope_layer: 4`, and doc masking enabled. The 4K stage uses `flash_attention_2` and RoPE theta 50,000; the 32K stage switches to `llama3_ring_attention`, theta 2,000,000, and CP from 1 to 4. These are confirmed configuration requirements; understanding the internals of NoPE / ring attention requires further reading of the corresponding nanotron branch.

**A calculable invariant.** 4K configuration: DP 192 × microbatch 3 × accumulation 1 × sequence 4096 = 2,359,296 tokens/update. 32K configuration: DP 12 × microbatch 1 × accumulation 6 × sequence 32768 = the same token count. TP=2; CP must not be counted again as independent data samples. Interpreting the product of parallel dimensions gives 384 ranks for the former configuration and 96 ranks for the latter; this is a configuration-derived calculation, not locally verified launch resources.

**Optimizer recipe.** The BF16 model, FP32 gradient accumulation, AdamW, gradient clipping=1, and exclusion of embeddings from weight decay stay the same. The long-context stage reduces learning rate from 2e-4 to 2e-5 and changes the learning-rate schedule. Differences in stage results cannot all be attributed to CP or long sequences, because data mixtures and other training parameters also change.

**Reproduction boundary.** The YAML retains `/scratch`, `/fsx`, S3 resume/checkpoint paths, and cluster-specific evaluation / Slurm configuration. The FineMath script's `args.email` is not defined in this file's parser; direct execution is therefore inferred to raise an attribute error when constructing executor arguments, but this has not actually been triggered. The SFT script defaults to `push_to_hub=True`, and several Boolean parameters use `type=bool`. Before future experiments, explicitly disable publishing and fix argument parsing in a personal experimental copy. Upstream was not modified.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Relationship | Evidence and meaning |
|---|---|
| datatrove: direct dependency of the example | `finemath-tokenize.py:20–24` imports executor, reader, and tokenizer; a separate dependency environment is needed. This study's inventory does not include datatrove as an independent clone. |
| TRL / PEFT: direct dependencies of the SFT example | `train.py:9–21` imports the packages, and L91–118 actually constructs the trainer / adapter. |
| nanotron: recipe runner | Parallelism, data-stage, and checkpoint settings in YAML are consumed by an external engine; this reading did not trace the engine code that consumes them and cannot be labeled complete reproduction. |
| OLMo-core / Marin: conceptual correspondence | All tie data mixtures and token budgets to staged training; this is not evidence that the repositories import one another. See the [training-connections note](../connections/training.md). |

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** The first experiment should verify batch semantics; training 3B is unnecessary.

- Question: with tokens/update held constant, do changes to sequence length / accumulation preserve comparability of data sampling and optimization-step counts?
- Inputs: a small number of synthetic token sequences with a fixed random seed and a scaled-down model, using an isolated nanotron environment and the exact version.
- Controls: tokenizer, data order, model, total training tokens, and optimizer; adjust only sequence length and accumulation to keep their product constant.
- Metrics: actual valid tokens per step, padding fraction, gradient norm, loss, peak GPU memory, and throughput.
- Expected: token counts should be equal; different sequence boundaries change conditioning context, so identical losses/gradients are not required. CP speed and memory benefits need separate multi-GPU validation.
- Compute: first check data semantics with a scaled-down model on one GPU; CP comparisons require multiple GPUs. Memory requirements and duration were not measured, and the original 384-rank configuration cannot simply be applied.
- Actual result: none; this study performed only configuration calculations and static reading.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Pin the nanotron / datatrove branches and commits used by SmolLM3, then trace how doc-masking positions enter attention.
- [ ] Compare data weights in stage2 / stage3 and calculate actual repetition counts for each corpus, beyond mixture proportions alone.
- [ ] Put environment-path mappings, Boolean-argument fixes, and the missing-email fix for these examples in a personal experiment directory, preserving the original snapshot.
