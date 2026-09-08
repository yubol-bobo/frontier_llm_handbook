<a id="training-连接从数据身份到梯度再到-moe-通信"></a>

# Training connections: from data identity to gradients to MoE communication

Date: 2026-09-08. Evidence level: L1, reading local source at pinned commits; no cross-repository integration run was completed. See the corresponding notes for full repository SHAs and verification scopes.

<a id="有源码证据的真实连接"></a>

## Actual connections with source evidence

| Source → target | Relationship type | Pinned-source evidence | What it establishes / does not establish |
|---|---|---|---|
| Open Instruct → OLMo-core | Direct dependency, pinned version | [pyproject.toml L7–74](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/pyproject.toml#L7-L74) declares `ai2-olmo-core`, with `tool.uv.sources` pinning `fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d`; [SFT imports L37–48](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/olmo_core_finetune.py#L37-L48) actually uses trainer / data / optimizer. | The SFT caller provides a path for studying reuse of pretraining infrastructure; the local OLMo-core HEAD is `92870a33…` and cannot replace the pin with a claim of compatibility. |
| Megatron-LM → DeepEP | Optional communication backend | [Flex dispatcher L1835–1843](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/token_dispatcher.py#L1835-L1843) selects `_DeepepManager`; [fused_a2a import L11–17](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/fused_a2a.py#L11-L17) imports `deep_ep.Buffer`. | This is a substantive Python/CUDA communication-interface connection, beyond an association because both projects discuss MoE. Not every Megatron MoE requires DeepEP. The independent-clone HEAD combination has not been compiled and verified. |
| Marin → Levanter within its monorepo | Direct training dependency | [experiment/train.py L30–50](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/experiment/train.py#L30-L50) imports Levanter config; [L192–241](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/experiment/train.py#L192-L241) constructs training configuration and ArtifactStep. | Marin traces the experiment DAG while Levanter handles inner training; the two names do not refer to the same layer. The inner JAX optimizer loop was not examined in depth. |
| SmolLM FineMath example → datatrove | Direct example dependency; external companion | [finemath-tokenize.py L20–62](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/continual-pretraining/finemath/finemath-tokenize.py#L20-L62) uses ParquetReader / DocumentTokenizer / SlurmPipelineExecutor. | Provides module interfaces for learning how raw text becomes training tokens; datatrove was not independently cloned among this round's 19 repositories, so its implementation cannot also be called fully read. |
| SmolLM SFT example → TRL / PEFT | Direct example dependency; external companion | [train.py L9–21](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/finetuning/train.py#L9-L21) imports them, and L54–122 constructs LoRA / SFTTrainer. | The SFT example can serve as a scaled-down exercise; it is not the original pretraining loop for SmolLM3 base. |

RL→Megatron edges omitted here are established by the respective source notes for [verl](../repositories/verl.md), [slime](../repositories/slime.md), and [miles](../repositories/miles.md). Placing repositories in one directory does not mean adaptation is complete.

<a id="值得对照的机制均不是软件依赖边"></a>

## Mechanisms worth comparing, none of them software-dependency edges

| Learning question | Evidence on each side | Differences to preserve |
|---|---|---|
| How many valid tokens are in one optimizer step? | [SmolLM](../repositories/smollm.md) recipes fix tokens/update; [OLMo-core](../repositories/olmo-core.md) rank-batch loss denominator; [TorchTitan](../repositories/torchtitan.md) global_valid_tokens; [Open Instruct](../repositories/open-instruct.md) response mask / accumulation token counts. | Pretraining token packing, SFT label masks, and RL response masks are not the same filtering operation. DP/SP/CP also must not be counted repeatedly as additional independent samples. |
| How do algorithms affect system scheduling? | TorchTitan's AC→compile→FSDP order; OLMo-core's final-microbatch synchronization; Megatron's route→dispatch→expert→combine; Open Instruct's weight sync. | Enabling CUDA graphs, asynchronous generation, or MoE capacity adds constraints; optimization switches cannot be arbitrarily reordered. |
| What counts as the same reproducible experiment? | Marin's name/version + config fingerprint + provenance; official SmolLM / OLMo-core configurations; Open Instruct's dependency pin. | Public configuration, cloned code, accessible data, a buildable environment, and numerical reproduction are different evidence levels. A configuration fingerprint is not a complete source-code/weight hash. |
| Can higher throughput change the training objective? | Open Instruct's new/old ratio and train/infer rho; Megatron's capacity dropping; OLMo-core's special instance_mask denominator. | Throughput comparisons must also check valid samples/tokens, drop rates, probabilities, and numerical differences, beyond tokens/s alone. |

<a id="一个可持续扩充的学习链"></a>

## A learning chain that can keep expanding

1. First read the [SmolLM](../repositories/smollm.md) recipes, calculate tokens/update as DP×microbatch×accumulation×sequence by hand, and mark external paths and engine versions.
2. Trace an actual optimizer step in [OLMo-core](../repositories/olmo-core.md) or [TorchTitan](../repositories/torchtitan.md); use a personal tiny fixed batch to check masking, denominators, and gradient accumulation. All experiments here remain pending.
3. Use [Marin](../repositories/marin.md)'s identity/dependency approach to organize personal inputs, code SHA, environment, experimental configuration, and outputs, beyond saving only a screenshot of final loss.
4. Move into [Open Instruct](../repositories/open-instruct.md), adding the rollout engine, old/new policy, response mask, and weight version to the same data flow.
5. When model/cluster scale requires it, go deeper into [Megatron-LM](../repositories/megatron-lm.md) → [DeepEP](../repositories/deepep.md), making the data flow concrete through token dispatch, buffers, and collectives.

For each new actual dependency edge, save the caller's pinned source location and version constraints. For each new conceptual correspondence, state the shared question and the differences that cannot be transferred directly. The next L2 pass should narrow to one invariant or cross-module protocol; promotion to L3 requires actual execution with saved configuration, logs, and metrics.
