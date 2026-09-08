<a id="torchtitan并行变换的次序与全局-token-归一化"></a>

# TorchTitan: Parallel transformation order and global token normalization

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/pytorch/torchtitan @ `4e20e76b235ec415bd81912ce09eed46fa39f717`  
Verification scope: Local Llama3 sharding / parallelize, the training entry point, and Trainer token accumulation, forward/backward, and finite checks; did not install nightly PyTorch or run GPU/convergence tests.

<a id="核心问题"></a>

## Core question

DP, TP, PP, CP, activation checkpointing, and compile are not switches that can be reordered arbitrarily. How can their composition order be understood from implementation, and how can we confirm that different batch/parallelism configurations still optimize the same objective?

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

The entry point [torchtitan/train.py](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/train.py):17–68 parses configuration, creates the trainer with `config.build()`, and calls `trainer.train()`; the seed checkpoint has a dedicated single-device branch. The current core loop is in `trainer.py`, rather than the long `train.py` that older tutorials may show.

1. [models/llama3/sharding.py](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/models/llama3/sharding.py), [L25–74](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/models/llama3/sharding.py#L25-L74): fills in decoder/layer sharding configuration. At `model.py:69–83`, `update_from_config` calls this function; declaration is separate from activation on the actual mesh.
2. [models/llama3/parallelize.py](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/models/llama3/parallelize.py), [`parallelize_llama`, L23–87](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/models/llama3/parallelize.py#L23-L87): `model.parallelize → AC → per-block compile → FSDP/mixed precision`. PP is handled separately through the trainer/model_spec pipelining path.
3. [trainer.py](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/trainer.py), [`forward_backward_step` / `_forward_backward_body`, L749–830](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/trainer.py#L749-L830): preprocess_inputs handles model/parallel layouts, followed by model → loss_fn(global_valid_tokens) → backward. PP has a separate list-of-microbatches branch.
4. Same file, [`train_step`, L860–1003](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/trainer.py#L860-L1003): first collects microbatch groups and valid tokens for one optimizer step, aggregates the denominator across distributed workers, then performs forward/backward, gradient clipping, non-finite checks, and optimizer update.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

**Layout declarations and activation transformations.** Llama3 sharding configuration is always populated; the actual `Module.parallelize()` determines which declarations apply based on enabled mesh axes. Sequence parallelism can be configured independently: when enabled, activations around attention/FFN are sharded and output projection uses reduce-scatter; when disabled, activations remain replicated and use all-reduce. The code calls shared decoder_sharding helpers; this reading did not audit every collective in their DTensor/SPMD lowering.

**Transformation order is an implementation constraint.** `parallelize_llama` applies layouts first, then wraps activation checkpointing, compiles by TransformerBlock, and applies FSDP last. The source recommends constructing the model on the meta device to avoid first placing complete weights on one GPU. FSDP shard degree=1 still installs MixedPrecisionPolicy; no sharding does not mean the wrapper has no semantics.

**Reuse between training and generation has boundaries.** `skip_dp` skips FSDP for inference; the comment explains that its forward hooks are incompatible with vLLM's inference_mode. AC / compile must be disabled through configuration. This is only a reuse entry point present in the code; actual vLLM-engine integration was not verified here.

**Batch budget and the actual denominator are two layers.** `trainer.py:455–474` derives accumulation steps from tokens/microbatch/DP-rank, PP microbatch count, and DP degree, and checks divisibility. `train_step` first reads `num_valid_tokens` from every group and sums them on device across the batch mesh. The former determines scheduling and resources; the latter excludes padding/masking and determines gradient weighting. The non-PP returned loss is explicitly local sum / global_valid_tokens, rather than a separate mean on each rank.

**Communication optimization is constrained by CUDA graph capture.** HSDP all-reduces only in the final accumulation group, but toggles this behavior only when accumulation=1 or CUDA graphs are disabled. The source notes that a graph is captured on the first group and reused for later groups; ordinary Python timing optimizations therefore cannot be applied unconditionally to the captured path.

**Asynchrony does not remove consistency requirements.** Before the optimizer step, loss-finiteness state is reduced over the relevant loss / PP meshes, combined with gradient-norm finiteness, and checked with device-side `_assert_async` to stop invalid updates. It also calls `maybe_wait_for_staging()` before modifying parameters. This demonstrates ordering constraints among numerical checks, checkpoint staging, and the optimizer. Failure injection and checkpoint recovery were not measured.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Relationship | Evidence and boundaries |
|---|---|
| OLMo-core: conceptual correspondence | Compare their valid-token denominators, final-microbatch HSDP synchronization, and Trainer/module layering. Current implementations differ; neither depends on the other. |
| Open Instruct: conceptual correspondence | RL response_mask and num_valid_tokens here both affect distributed loss weighting; RL additionally has policy ratios, rho, and version lag, so the pretraining objective cannot be copied directly. |
| Megatron-LM: conceptual correspondence | Both address model sharding/communication/recomputation. TorchTitan offers a clear composition entry point, while Megatron's MoE dispatcher provides a deeper case study; there is no evidence of direct dependency. |
| vLLM: code-supported reuse boundary | `parallelize.py:58–62` contains an explicit inference skip_dp branch and constraints; this alone does not establish that a complete service ran. |

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** First reproduce changing AC without changing the computational objective.

- Inputs: fixed seed, tiny Llama3 config, and fixed short-token batch; initially use the same devices and parallelism.
- Controls: model initialization, data order, dtype, kernel, and optimizer; toggle only activation checkpointing, initially disabling CUDA graphs and studying the graph path separately once stable.
- Metrics: per-step loss/grad_norm at full reporting precision, updated parameters, valid tokens, peak GPU memory, and step time after warmup.
- Expected: AC trades recomputation for memory. Under identical determinism conditions, compare numerical consistency according to project requirements, beyond rounded logged loss. Performance measurement must at least pass warmup; this note does not claim it will necessarily be faster.
- Compute: start with a tiny model on one GPU; TP/HSDP, CP, and communication traces require multiple GPUs. Specific memory/time costs have not been measured.
- Actual result: none; upstream tests and training were not executed.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Read `models/common/decoder_sharding.py` and `protocols/module.py` to trace how layout declarations become individual collectives.
- [ ] Read reduction / gradient scaling in the loss implementation to check the exact relationship between global_valid_tokens and DP averaging.
- [ ] Read distributed checkpoint staging: why must the optimizer wait for certain staging operations, and which I/O can remain asynchronous?
