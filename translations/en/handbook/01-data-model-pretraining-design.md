<a id="01从能力目标到可交付基础模型数据架构与预训练设计"></a>

# 01 | From Capability Goals to a Deliverable Base Model: Data, Architecture, and Pretraining Design

[Back to the end-to-end process](00-end-to-end.md) · [Distributed execution](02-distributed-pretraining-operations.md) · [Marin 535B case study](04-marin-535b-live-case-study.md)

Checked: 2026-09-08. This chapter discusses starting from randomly initialized weights and ultimately delivering a base model suitable for post-training. Reusing a public tokenizer, data processor, or training framework still counts as training from scratch; loading existing model weights and continuing training is continued training. Local work consisted only of reading source code, configurations, and public reports. No training corpus or model weights were downloaded, and no experiments were executed.

Evidence falls into three categories: **[Public fact]** means an author report or located implementation; **[Engineering synthesis]** means an engineering process connecting experience across projects, not any lab's complete recipe; **[Unknown]** means closed-frontier details that public material cannot confirm. Each stage's Gate is a suggested validation condition. Register concrete thresholds at project launch; this chapter is not a universal hyperparameter table.

**A reference for current scale.** [Public fact] On 2026-09-03, Open Athena disclosed that Marin had begun training an MoE with 535B total parameters and 23B active parameters per token, planning approximately three months and 18T tokens. The announcement describes an ongoing run; it does not justify claiming that the model is complete or has achieved an evaluation score, or treating estimated training duration as measured cost. [Marin 535B-A23B launch note](https://openathena.ai/blog/marin-535b-launch-note/)

<a id="gate-0先确定要买到的能力再分配训练预算"></a>

## Gate 0: Decide Which Capabilities You Are Buying Before Allocating the Training Budget

**Inputs:** target use cases, deployment budget, available data, and compute resources. **Outputs:** capability matrix, frozen evaluation protocol, resource budget, and range of candidate models.

[Engineering synthesis] Break “a strong model” into goals such as language coverage, code, mathematics, knowledge, long-document understanding, and prerequisite capabilities for tool use. For each, specify the measurement method, minimum acceptable level, and capabilities that must not be sacrificed. Base-model evaluation can use completion, likelihood, and controlled few-shot tasks. Diagnose chat behavior that requires SFT for stable performance separately, to avoid mistaking failure to follow a template for lack of knowledge. Isolate evaluation sets from training and data filtering, and record versions, prompt formats, sampling, and aggregation methods.

Break the budget into at least data processing, exploratory experiments, main training, recovery and rework, evaluation, and post-training/inference validation. A fixed GPU count does not imply fixed effective compute: network, storage, downtime, context length, and model shape all change the token count that can be completed. Measure stable throughput with candidate implementations before estimating wall-clock time, and allocate an explicit downtime allowance. On the deployment side, also constrain resident weights, KV cache, time to first token, and total consumption per task. The model requiring the least training compute is not necessarily the model with the lowest lifecycle cost.

[Public fact] Chinchilla investigates compute-optimal allocation between model parameters and training data under its training setup. The common dense-Transformer approximation `C ≈ 6 × N × D` can help construct an initial budget; the definition of `N` must be explicit, and the approximation omits or simplifies costs such as attention. It does not imply a fixed tokens/parameter ratio applicable to all eras, data quality, training lengths, and serving objectives. [Original Chinchilla paper](https://arxiv.org/abs/2203.15556)

**Tradeoffs and failure signals:** the budget covers main training without room for validation and recovery; target context reduces throughput far below estimates; training cost is acceptable but deployment cannot meet target latency. **Passing conditions:** at least one candidate meets the full budget under a conservative throughput estimate, and the evaluation protocol distinguishes insufficient base knowledge from an interaction-format mismatch.

<a id="gate-1把-scaling-law-当成需要验证的预测模型"></a>

## Gate 1: Treat a Scaling Law as a Predictive Model That Requires Validation

**Inputs:** candidate architectures, initial data mixture, tokenizer, and budget. **Outputs:** a reproducible scaling recipe, prediction errors at held-out scales, and a decision record before main training.

[Engineering synthesis] First eliminate clearly unstable combinations at small scale, then run equal-compute sweeps across several budgets: at a given budget, compare larger models trained on fewer tokens with smaller models trained on more tokens. How architecture, optimizer, initialization, batch, and learning rate change with width, depth, and duration is itself a hypothesis under test. Data-quality or mixture ablations should hold these variables fixed where possible. When tokenizers differ, absolute per-token cross-entropy values are not directly comparable.

Separate small runs used for fitting from larger runs used to validate predictions. Retain short runs close to the target model shape to inspect routing, memory, communication, and numerical stability, and sufficiently long runs to test extrapolation in training duration. Normal throughput in a short run does not prove stability tens of thousands of steps later; faster early loss reduction does not prove a better final outcome. Measure downstream capabilities separately rather than replacing all capability goals with average validation loss.

[Public fact] Marin's Delphi publicly documents recipe changes after a failure: its first small-scale fit looked good but failed to predict larger runs. After revision, a preregistered prediction for a `1e23 FLOPs` run was within 0.2% of its final loss. This result applies to its specific data, architecture, and hyperparameter-transfer scheme; it is not a guarantee that arbitrary models can extrapolate by a factor of 300. [Delphi scaling suite authors' account](https://openathena.ai/blog/delphi/)

**Tradeoffs and failure signals:** a mixture preferred by small models loses at larger scale; gradient spikes appear only after extended training; fitting error is small but held-out-scale predictions diverge; repeated selection on one evaluation set causes overfitting. **Passing conditions:** key choices have ablation evidence, held-out-run errors are acceptable, and main-training predictions and redesign triggers are registered. Where evidence is lacking, retain uncertainty intervals rather than inventing precise gains.

<a id="gate-2把数据做成可追踪可重建的训练产品"></a>

## Gate 2: Make Data a Traceable, Rebuildable Training Product

**Inputs:** raw web pages, code, papers/OCR, mathematics, long documents, and any necessary synthetic data. **Outputs:** a document collection with provenance, an evaluation-isolation list, tokenized shards, a data-mixture manifest, and quality reports.

[Engineering synthesis] A suggested pipeline is source registration → extraction and normalization → language/domain identification → quality filtering → deduplication and evaluation decontamination → splitting and sharding → tokenization → mixture sampling. Adapt the order to the source: removing page templates first may improve approximate deduplication, for example, but preserve original-text identifiers for investigation. Each record should link at least to its source, crawl or publication version, license/terms of use, original identifier, content checksum, processing-code version, filtering reason, and parent record. Synthetic data adds generation-model, prompt, sampling, and verifier versions, distinguishing new-sample counts from demonstrable information gains.

Quality is not a single classifier score. For web pages, inspect residual navigation and templates; for code, preserve file, repository, and dependency context; for OCR, inspect damage to formulas and layout; for multilingual filtering, check systematic false removal of minority languages. High-quality filtering may improve short-term learning efficiency while losing coverage and rare capabilities, so perform manual sampling and retention audits by domain, language, and length.

Deduplication must consider exact duplicates, near duplicates, and source clusters together. Code forks, web mirrors, and revised documents can cross training/validation boundaries; deduplicating only within individual shards does not handle global duplication. Decontamination should cover evaluation questions, answers, explanations, and approximate variants, preserving traceable match records. String and semantic similarity each have false positives and false negatives; a zero-match metric does not establish the absolute absence of contamination.

Specify whether mixture proportions use document counts, bytes, or valid target tokens, and record both planned and actual sampled proportions. Repeated sampling adds training tokens without an equal increase in unique information; small, heavily weighted corpora should report effective epochs, repetition, and signs of degradation. Reserve long-document, mathematics, and code pools for later stages where appropriate, instead of reaching midtraining only to discover that the same material has already been learned repeatedly.

[Public fact] Marin's `ArtifactStep` includes configuration and dependency versions in its fingerprint, while artifact addresses use explicit names and versions. This makes data and experimental dependencies objects in a graph. The source does not thereby guarantee automatic, complete content addressing for raw data and all execution code, so users must still record data checksums and code SHAs. [Marin artifact identity and version implementation](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/lazy.py#L213-L237)

**Tradeoffs and failure signals:** quality scores rise while language coverage falls; usable data collapses after deduplication; training and validation losses become unusually close; actual mixtures diverge from configuration; a named data version produces different contents. **Passing conditions:** provenance is traceable, isolation and deduplication policies are explainable, actual token distributions match the design, and any sampled training segment can be traced to its source and transformation chain.

<a id="gate-3冻结-tokenizer-与模型之间的数据契约"></a>

## Gate 3: Freeze the Data Contract Between Tokenizer and Model

**Inputs:** samples representative of the target distribution and candidate tokenizers. **Outputs:** a pinned tokenizer, special-token specification, vocabulary and embedding shapes, and an encoding test set.

[Engineering synthesis] Compare token inflation for languages and code, number splitting, byte fallback, encode/decode round trips, special-symbol handling, and vocabulary size. Shorter tokenization can reduce sequence costs, but a larger vocabulary adds embedding, output-projection, and optimizer states; compression cannot replace downstream quality. Across tokenizers, supplement comparisons with metrics normalized over the same raw text or by bytes, plus identical task evaluations, explaining the limitations of each measure.

Freeze vocabulary files, normalization rules, and BOS/EOS/PAD IDs and meanings, and verify that the tokenizer's document-length configuration is not confused with the model's position range. Adding special tokens later affects model shapes and new-row initialization; it must be an explicit migration, not an unrecorded file replacement.

[Public fact] OLMo-core represents vocabulary size, EOS, PAD, and optional BOS separately and provides `padded_vocab_size`, which by default rounds the vocabulary dimension of the embedding table up to a multiple of 128 for throughput. Padded model rows are not newly added real tokens; the alignment value is an implementation choice. [OLMo tokenizer configuration](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/data/tokenizer.py#L48-L107)

**Tradeoffs and failure signals:** abnormal length inflation in a language; special tokens misinterpreted within ordinary content; conflicting PAD/EOS handling; out-of-range data IDs. **Passing conditions:** boundary samples and round-trip behavior are defined, every data shard and the model reference the same tokenizer version, and metric-comparison definitions agree.

<a id="gate-4联合选择模型架构与可实现的计算形状"></a>

## Gate 4: Jointly Choose the Model Architecture and Executable Compute Shapes

**Inputs:** capability goals, scaling results, training cluster, and deployment constraints. **Outputs:** architecture specification, parameter/FLOPs/memory breakdown, and validated candidate system configurations.

[Engineering synthesis] Dense models offer relatively straightforward execution paths and cost structures. MoE increases total capacity while computing only some experts, but must specify total parameters, active parameters per token, shared layers, expert count, top-k, routing method, and capacity policy together. Describing a 535B MoE by 23B active parameters does not mean its weights or optimizer states contain only 23B parameters. State sharding, expert loads, and communication distribution each need separate calculations. Active parameters also cannot fully represent attention, routing, data movement, and recomputation costs.

[Public fact] Megatron's routing implementation supports different routing branches and applies expert-capacity limits, token dropping, auxiliary balancing losses, and expert-bias updates according to configuration. These branches show that MoE is not a fixed algorithm; the existence of a feature does not mean every model enables it. [Megatron MoE routing implementation](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/router.py#L750-L841)

[Engineering synthesis] On target hardware, inspect tokens/expert distributions, drop rates, expert communication time, throughput tails, and peak GPU memory. Dropless execution removes a class of dropping but does not automatically remove imbalance or communication overhead. Attention designs such as GQA change training and inference resource requirements; full attention's long-sequence compute term is not captured by a simple `6ND`. Validate positional encoding, normalization, sparse attention, and hybrid structures as complete candidate recipes; a single-kernel benchmark cannot establish final quality.

**Tradeoffs and failure signals:** theoretical FLOPs fall while wall-clock time rises; a few experts remain overloaded; training runs but checkpoints cannot be reliably exported and deployed. **Passing conditions:** quality, stability, measured throughput, and serving constraints all pass; key execution paths have been checked close to target shapes before the very large main run.

<a id="gate-5把-packingmask-和位置当作训练目标的一部分"></a>

## Gate 5: Treat Packing, Masks, and Positions as Part of the Training Objective

**Inputs:** tokenized documents, length distributions, and context policy. **Outputs:** an explicit batch structure, valid-target-token counting rules, and a boundary-correctness report.

[Engineering synthesis] Packing improves utilization and changes relationships between documents. Define separately which positions may attend to each other, which targets enter loss, whether position IDs reset, and whether the next token after a document ends is a prediction target. Causal attention masks, document-isolation masks, padding masks, and loss masks are not equivalent. Segmenting long samples also requires specifying overlap, truncation, and cross-segment context.

[Public fact] OLMo-core's `get_labels` first marks positions specified by label, attention, and instance masks as ignored, then shifts labels left and appends an ignored entry at the end. This helper alone does not fully define inter-document attention or automatically prove correct cross-document target handling; continue checking through the dataloader and attention paths. [OLMo label construction](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/data/utils.py#L590-L605)

[Engineering synthesis] Minimal checks to design, but not executed here, include comparing logits/loss at valid positions for a single document and equivalent packing; checking EOS, empty documents, all-PAD inputs, length-one inputs, and final positions; changing a masked document and checking whether other documents' outputs remain unchanged as specified; and comparing loss normalization on one GPU versus distributed splits. Training-token budgets should distinguish input positions from actual supervised targets, especially with padding, filtered instances, and selective loss.

**Tradeoffs and failure signals:** packing improves throughput but introduces unintended cross-document leakage; loss is low because too many positions are ignored; changing GPU count changes update magnitude. **Passing conditions:** target alignment and attention boundaries match the specification, batch and loss-denominator calculations are reviewable, and distributed partitioning preserves the intended objective.

<a id="gate-6主预训练交付的是一条可解释的模型轨迹"></a>

## Gate 6: Main Pretraining Delivers an Explainable Model Trajectory

**Inputs:** the frozen recipe, data, and execution configuration above. **Outputs:** a sequence of checkpoints with complete state, consumed-data records, domain-evaluation curves, and candidate models for the next stage.

[Engineering synthesis] Main training begins from random initialization, recording initialization method, precision, optimizer parameter groups, scheduler, batch, and data-consumption order. Resource changes, data replacements, and learning-rate adjustments during a run should become auditable branches or events. Resuming training restores more than weights: optimizer, schedule, random state, and data position are also required; see the training-infrastructure chapter for system implementations.

Judge global and per-domain metrics together: an improved overall loss can conceal deterioration in language or coding capabilities. Gradient spikes may come from bad samples, numerical problems, recovery errors, or schedule changes; reducing the learning rate is not a default solution for every case. Keep regular evaluation protocols stable and separate diagnostic sets from the final holdout. Specify in advance when to continue, pause for diagnosis, roll back, or stop early, so that substantial compute already spent does not become the justification for continuing.

**Tradeoffs and failure signals:** repeated data reads, gradually declining throughput, spikes triggered by specific sources, mixture drift, and persistent divergence between training loss and key capabilities. **Passing conditions:** candidate checkpoints have complete provenance, key capabilities and stability meet stage requirements, and recovery correctness and inference-export consistency have been validated during execution rather than postponed until training ends.

<a id="gate-7按能力缺口设计-midtraining-与长上下文阶段"></a>

## Gate 7: Design Midtraining and Long-Context Stages Around Capability Gaps

**Inputs:** a base checkpoint, capability gaps, targeted data pools, and long-document evaluation. **Outputs:** stage comparisons, the next-stage recipe, and a base model ready for SFT/RL.

[Public fact] OLMo 3's public process explicitly separates broad pretraining, targeted high-quality midtraining, and a long-context stage containing long documents. Its 7B recipe reports the following. These are actual publicly reported scales for that model, not recommended resource allocations for very large models. [OLMo 3 official stages and run table](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/scripts/official/OLMo3/README.md#L23-L77)

| OLMo 3 7B stage | Publicly reported training tokens | Publicly reported GPU configuration |
|---|---:|---:|
| Pretraining | 5.93T | 512 H100 |
| Midtraining | 100B | 128 H100 |
| Long-context | 50B | 256 H100 |

[Engineering synthesis] Midtraining can address gaps with more targeted code, mathematics, knowledge, and task formats, but should be compared with continuing the original recipe at the same budget while monitoring forgetting of general capabilities. Decide data proportions, learning rate, duration, and retention of broad data jointly. Stage names have no universal boundaries; checkpoint averaging and multiple stages are not required for every model.

Long context requires considering long-document supply, positional encoding, attention kernels, context parallelism, batch, and schedules together. Randomly concatenating short documents supplies length but cannot replace supervision for cross-chapter reasoning. Evaluation should cover short-context regressions, different length ranges, information positions, and real multi-segment retrieval and reasoning, rather than a single needle-in-a-haystack score.

[Public fact] SmolLM3's public 4K-to-32K configurations change RoPE theta, the attention backend, CP, DP, accumulation steps, learning rate, and data lists together. For example, theta changes from 50,000 to 2,000,000 and CP from 1 to 4. Both configurations have a nominal 2,359,296 input tokens/update under `DP × microbatch × accumulation × sequence length`. CP partitions the same context and must not be multiplied in again as an independent-sample factor. This is a static configuration calculation without execution validation; the nominal input count also does not guarantee that every position enters loss. [SmolLM3 4K configuration](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/stage1_8T.yaml#L163-L255), [32K continuation configuration](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/long_context_4k_to_32k.yaml#L213-L298)

**Tradeoffs and failure signals:** target-domain gains accompany general-capability losses; longer positions are supported but distant information is not used; long sequences are unstable; extending context unexpectedly changes actual tokens/update or the loss denominator. **Passing conditions:** gains over the base checkpoint and budget-matched controls are acceptable, short- and long-context capabilities meet goals, and inference supports the same position and attention semantics.

<a id="交给下一章的产物与证据边界"></a>

## Artifacts for the Next Chapter and Evidence Boundaries

[Engineering synthesis] Before large-scale execution and post-training, deliver a set of artifacts with mutually referenced versions: capability and evaluation protocols, budgets and scaling predictions, data provenance and mixture manifests, tokenizer, architecture, batch semantics, optimizer and schedule configurations, recovery protocol, checkpoint lineage, and stage evaluations. Together they form an inspectable, accountable training specification; a YAML file that starts a run covers only part of it.

[Unknown] This chapter cannot reconstruct the exact mixtures, private synthetic corpora, full ablations, implicit cleaning rules, or complete architecture techniques of closed frontier models. The 7B/3B projects provide transparent mechanisms but do not prove that their recipes transfer unchanged to hundreds of billions of parameters. The ongoing Marin 535B project also has no post-completion quality conclusion yet. The 19 repositories are complementary evidence sources; neighboring directories are not an already-compatible dependency set.

The local source anchors in this chapter correspond to the pinned snapshots below; web reports were read as of the check date above. The existence of the linked code functionality does not mean this work has validated its runtime behavior or reproduced the authors' results.

| Local snapshot | Full commit SHA |
|---|---|
| `sources/marin` | `5e2436d0f61462983003bd8b6eaef8235ecab78c` |
| `sources/olmo-core` | `92870a33c3fee060d57c3faec52b0b369ece85a6` |
| `sources/smollm` | `a041759883ec7152d18fb985ea49be641a0bceef` |
| `sources/megatron-lm` | `c9b53d0a87cb926f47115259593ecfeb351ca29f` |
