<a id="m00m04从-python-到能审查预训练配方"></a>

# M00–M04 | From Python to Reviewing a Pretraining Recipe

Design and link check: 2026-09-08. This is a **curriculum design** for self-directed learners, not a personal record of completed learning or an internal training program from any lab. The ten exercises and GPU extensions in this article have not yet been executed; where other independent lessons have execution evidence, consult their individual records. Reading source code, running a small example, validating performance, and reproducing a public model are four different accomplishments.

The assumed background is Python functions, classes, files, and common containers; no research background is required. Follow **M00 → M01 → M02 → M03 → M04 → M05**. A CPU is sufficient for self-assessed completion of this article; some exercises require a separately prepared CPU version of PyTorch, and this article does not install an environment. Time estimates include the core reading and two CPU exercises; the full original assignments and GPU reproductions require additional time. All are plans, and toy results cannot establish frontier-level quality.

The original Stanford CS336 2025 course requires deep learning, PyTorch, linear algebra, calculus, and probability. This curriculum adds bridging material; **it does not claim that proficiency in Python alone prepares someone to complete the full original assignments directly**. The [official 2025 archive](https://cs336.stanford.edu/spring2025/) is the course entry point; its GPU prices are historical information. Complete the core reading and exercises first, then select further reading to address gaps. You do not need to read every paper in advance.

Keep your own explanations, calculations, source locations, and unresolved questions. Exercises can live in a personal directory named by module ID; the listed deliverables have not yet been generated. You may skip learning material you already know, but you may not skip completion evidence.

<a id="m00"></a>
<a id="m00入门诊断与复现素养"></a>

## M00 | Entry Diagnostics and Reproducibility Literacy

<a id="先修与可跳过诊断"></a>

### Prerequisites and diagnostics for skipping material

Only basic Python is required. First try explaining the three axes of a `B×T×D` array, calculating the dot product of two length-three vectors by hand, explaining why probabilities sum to one, and describing what it means for a function's output to change when its input changes slightly. Fill any individual gap first; you do not need to complete an entire mathematics course before starting. Learners who can record environment versions, assert shapes, locate a function's callers, and distinguish planned results from measured results can proceed directly to this module's self-assessed completion checks. They must still pass the mathematics exit check below before entering M01.

The mathematics bridge covers vectors and weighted sums, conditional probability, mean and variance, and local rates of change, in that order. M01 then applies the chain rule to a model's computation graph. If a proof is not yet clear, identify its assumptions and inputs and outputs first instead of memorizing symbols alone.

**Study only the parts where the diagnostic reveals a gap.** The following bridges are available as needed. They do not require reading all of D2L and do not replace the three core readings below. Try the exit questions first; skip an item if you can calculate and explain its answer independently. If you get stuck, read only the specified concepts, retry the question, and stop there.

| Where the diagnostic exposed a gap | Textbook entry point and reading scope | When to stop |
|---|---|---|
| Dot products, matrix multiplication, or shape reasoning | [D2L Linear Algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html): scalars, vectors, matrices, dot products, matrix multiplication | You can express one output element as a weighted sum of inputs and explain why the inner dimensions must match; skip eigendecomposition for now |
| Derivatives, partial derivatives, and parameter changes | [D2L Calculus](https://d2l.ai/chapter_preliminaries/calculus.html): small examples of derivatives, gradients, and the chain rule | You can propagate a rate of change through a two-step function and distinguish a function value from its derivative; this module does not require integration or matrix-calculus proofs |
| Joint versus conditional probability | [D2L Probability and Statistics](https://d2l.ai/chapter_preliminaries/probability.html): sample spaces, joint and conditional probability, expectation, and variance | You can explain how conditioning changes the denominator and why a mean cannot capture all variation; skip inference with complex distributions for now |

You can begin with pencil and paper and Python lists. Use the textbook's tensor examples for comparison; there is no need to install its entire supporting environment for this bridge. Recording your first mistake and the denominator or dimension you used to correct it in a prerequisite-gap list is more useful than repeatedly watching the same explanation.

<a id="学习目标细分"></a>

### Detailed learning objectives

- Distinguish model weights, training configuration, code snapshot, data version, logs, and evaluation protocol, and understand how each missing item limits reproducibility.
- Understand that the training set drives updates, the development set guides selection, and the held-out set supports less frequently accessed generalization checks; a random seed cannot eliminate data leakage.
- Report “runs successfully,” “numerically correct,” “matches the original setup,” and “reproduces the authors' score” separately, and identify the evidence supporting a conclusion.
- Explain a tensor's shape, dtype, and device; distinguish parameters, activations, gradients, and optimizer states, without yet needing to calculate large-model GPU memory requirements.

<a id="核心阅读按顺序最多三项"></a>

### Core reading: in order, no more than three items

1. [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html): read Tensors first, then examine the organization of the model, optimization, and saving sections. In image examples, follow only the tensors and training loop; do not treat the image task as an LLM recipe.
2. [From One Token to One Update](../lessons/01-one-token-to-update.md): follow one concrete example first, then map its terminology to [Section 8 of the end-to-end handbook](../handbook/00-end-to-end.md). Understanding distributed details is not yet required.
3. [OLMo-core source notes](../notes/repositories/olmo-core.md): read only the snapshot, entry points, and evidence boundaries; verify in the source that `get_labels` is a function and that the token count in a training report is run metadata. They answer different questions.

**Optional reading:** prerequisites and course structure on the CS336 archive page. The goal is to identify gaps in your foundations, not to make “watching every lecture” a condition for starting the exercises.

<a id="应画或推导的对象"></a>

### What to draw or derive

Draw a `raw text → token IDs → logits → loss → gradients → parameter update` diagram, labeling each edge with its data type and owner. Then draw an experiment-record relationship diagram connecting the code SHA, configuration, data checksum, and result files. When you encounter “loss decreased,” use the diagram to identify the missing evaluation distribution and comparison baseline.

<a id="两个-cpu-练习"></a>

### Two CPU exercises

**E00-A: Create a reviewable record.** Use the Python standard library to read ten short sentences you wrote yourself, and save the normalization rules, file SHA-256, record count, and generation time. Create a JSON file containing the seed, data identifier, objective, and stopping condition. Change one sentence and recalculate to verify that the record reflects the content change. Deliver the original record and the resulting diff; do not use the current date or filename as a content identity.

**E00-B: Reproduce a statistical judgment.** Generate two groups of simulated scores, calculate their means and sample standard deviations, plot every point, and explain whether the mean is sufficient to establish an improvement. Construct an example in which the average score rises while a subgroup declines, explicitly labeling it as synthetic data. For one repository fact, write a four-part card: “claim / source / what it verifies / what it cannot verify.”

**GPU extension:** this module has no necessary GPU exercise. Carry the recording conventions forward to future GPU experiments.

<a id="验收问题与错误理解"></a>

### Self-assessed completion questions and misconceptions

Can you answer without hints: Why does the same seed not guarantee bitwise-identical results across versions? Why might inference-ready weights be insufficient for fully resuming training? Why is a data directory with the same name insufficient to define the same experiment? If your answer is only “fix the random seed,” return to the evidence relationship diagram. Reporting README descriptions as personally verified results or treating a single score as a certain gain both count as failing this check.

**Mathematics exit check for M01: calculate by hand first, then verify.** All three questions are teaching constructions, not model experiments; code may be used only to check your work afterward.

1. Let `A=[[1,0,2],[0,1,1]]` and `B=[[1,2],[3,4],[5,6]]`. Write the shape of each matrix and of `AB`, and expand the sum for one element. The reference result is `[[11,14],[8,10]]`; stating the result without explaining the inner dimensions and output axes does not pass.
2. Among one hundred simulated tasks, forty are coding tasks, and thirty of those succeed. Find the probability of success given that a task is a coding task, and the proportion of tasks that are both coding tasks and successful. The reference values are `0.75` and `0.30`; you must explain why the denominators are forty and one hundred, respectively, rather than merely apply a formula.
3. Let `f(w)=(2w−1)²`. At `w=1`, find the function value and derivative, then predict how the function changes if the parameter increases slightly. The reference values are `1` and `4`; you should be able to explain the chain rule through the inner and outer functions and state why the derivative is not the exact change for an arbitrary step size.

Enter M01 only after you can independently explain all three answers and pass the reproducibility checks. For an incorrect answer, revisit only the corresponding bridge and try a different set of numbers to avoid mistaking memorization for understanding. The first lesson's handwritten gradients and finite differences can help with the third question, but do not establish that you have mastered autograd or completed a decoder.

**Deliverables:** `M00-evidence-card.md`, manifests for two data versions, a statistics script and plot, and a prerequisite-gap list. **Suggested workload:** 8–16 hours; allow another 12–24 hours for mathematics or PyTorch bridging. These are all estimates. **Next module:** [M01](#m01).

<a id="m01"></a>
<a id="m01从零写出一个-decoder理解-tensor-与-autograd"></a>

## M01 | Write a Decoder from Scratch and Understand Tensors and Autograd

<a id="先修与可跳过诊断-1"></a>

### Prerequisites and diagnostics for skipping material

Complete M00; matrix multiplication, conditional probability, and local derivatives are enough to begin. First trace the shapes of embeddings, Q/K/V, attention outputs, and vocabulary logits on paper. Learners who can independently implement a small causal decoder, demonstrate that future tokens do not affect past outputs, and explain backpropagation and gradient accumulation may submit just the exercises and source comparison without repeating every reading section.

<a id="学习目标细分-1"></a>

### Detailed learning objectives

- Understand next-token prediction through `p(x₁…x_T)=∏p(x_t|x_<t)`; distinguish simultaneous computation at multiple positions during training from step-by-step sampling during generation.
- Understand what embedding lookup, linear projection, splitting into multiple heads, scaled dot products, causal masks, residual connections, normalization, the MLP, and the output head each change.
- Trace `[B,T] → [B,T,D] → [B,H,T,d] → [B,H,T,T] → [B,T,V]` and explain why transposition and broadcasting do not create meaning on their own.
- Understand that autograd records the actual computation graph and that shared parameters accumulate gradients; `detach`, no-gradient contexts, and the timing of gradient clearing change the update.
- Distinguish writing the key computations from scratch from training from random weights. The former is for learning; the latter may use an existing framework.

<a id="核心阅读按顺序最多三项-1"></a>

### Core reading: in order, no more than three items

1. [CS336 2025 Assignment 1 official handout](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf): select only the model architecture and training-loop sections for this module; leave BPE for M03. You may consult `tests/adapters.py` in the official repository to understand the interfaces. Completing the whole assignment at once is not required.
2. [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html): trace one forward pass, backward pass, and gradient accumulation. Write the expected gradients yourself before running the verification.
3. [OLMo Transformer.forward](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/nn/transformer/model.py#L523-L610): read embedding → blocks → lm_head and compare with the [existing notes](../notes/repositories/olmo-core.md). Initially skip the CP, TP, and compile branches, noting only their interface boundaries.

**Optional reading:** the attention and architecture definitions in [Attention Is All You Need](https://arxiv.org/abs/1706.03762); its original encoder-decoder design does not describe all of today's decoder-only LLMs. For diagrams, see the [pinned CS336 2025 architecture lecture](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf).

<a id="应画或推导的对象-1"></a>

### What to draw or derive

Calculate single-head attention for three tokens by hand: write QKᵀ first, add the causal constraint, then obtain the weighted V. Identify the axis along which softmax operates. Draw a backward path for a parameter shared by two positions, explaining why its gradients must add. List the parameter counts of the embedding and linear layers, distinguishing parameter count from the size of the intermediate attention matrix.

<a id="两个-cpu-练习-1"></a>

### Two CPU exercises

**E01-A: Build an explainable toy decoder.** Use a fixed byte or character vocabulary and write attention, the MLP, residual connections, a position table, and the output head yourself. You may use basic PyTorch operators and automatic differentiation, but do not call a Transformer module directly. Two layers, width 64, four heads, and length 32 are possible debugging starting points, and are only toy dimensions. Try fitting a small subset of short sentences you wrote yourself, retaining curves and failure records. On failure, reduce the sample size and inspect the targets first; do not hide errors by enlarging the model.

**E01-B: Find counterexamples for your model.** Fix the weights, disable dropout, change the latter half of the sequence, and compare logits in the first half. Change another sequence in the batch and check that the first remains unchanged as expected. Intentionally remove the causal mask or transpose the wrong axes, and record the assertions that catch these errors. Map three functions to OLMo and identify the omitted mechanisms without claiming implementation equivalence.

**GPU extension:** expand the corpus and model only after correctness checks pass. Compare CPU and GPU errors on the same inputs, then record throughput and GPU memory use. GPU results are not a prerequisite for completing this module.

<a id="验收问题与错误理解-1"></a>

### Self-assessed completion questions and misconceptions

Why can training compute an entire sequence in parallel while sampling still has sequential dependencies? Why is the last logits axis the vocabulary, and why can logits not be added directly as probabilities? Why does a very low training loss fail to rule out access to future tokens? Why might changing batch order introduce slight numerical differences without changing the mathematical objective? Treating attention as a fixed dictionary for retrieving source text or treating automatic differentiation as numerical differentiation requires a revised explanation.

**Deliverables:** a toy decoder, shape table, causality checks, fault-injection report, and interface comparison with OLMo. **Suggested workload:** 20–35 hours. **Next module:** [M02](#m02); make updates correct before pursuing greater scale.

<a id="m02"></a>
<a id="m02优化数值与-loss-正确性"></a>

## M02 | Optimization, Numerics, and Loss Correctness

<a id="先修与可跳过诊断-2"></a>

### Prerequisites and diagnostics for skipping material

Complete M01 and be able to identify how the loss depends on trainable parameters. Diagnostic: how should you combine average losses from two microbatches with different numbers of valid target tokens? Where should labels be shifted left? What are the gradients after two consecutive `backward` calls? If you can answer with hand calculations, assertions, and counterexamples, and explain the relationship between AdamW states and restarts, you can shorten the introductory reading.

<a id="学习目标细分-2"></a>

### Detailed learning objectives

- Express cross-entropy as the sum of negative log probabilities over valid targets, and identify target shifting, the ignore index, padding, and document boundaries.
- Distinguish averaging local averages from summing and then dividing by the total number of valid target tokens; understand that reduction changes sample weighting and gradient scale.
- Move from SGD to AdamW: understand first- and second-moment states, bias correction, epsilon, decoupled weight decay, learning-rate schedules, and the placement of clipping.
- Distinguish floating-point dynamic range, precision, and accumulation order; explain overflow through stable log-sum-exp computation instead of attributing every NaN to the learning rate.
- Understand that weights, gradients, and optimizer states can use different dtypes; mixed precision does not mean indiscriminately converting the entire model to a single type.

<a id="核心阅读按顺序最多三项-2"></a>

### Core reading: in order, no more than three items

1. [The first lesson's derivation of loss and updates](../lessons/01-one-token-to-update.md), then the optimizer and training sections of the A1 handout. Understand the arithmetic with small numbers before examining scheduler APIs.
2. [OLMo train_batch](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/train_module/transformer/train_module.py#L345-L441), tracing upward to `get_labels`. Pay particular attention to the instance-filtering comment: this path adds tokens from some ignored instances back to the denominator, artificially lowering reported loss. Do not replace the actual recipe with your ideal formula.
3. [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html): focus on non-associativity, extreme values, and backend differences. The documentation changes over time, so record the reading date and do not copy hardware switches as universal recommendations.

**Optional reading:** [OLMo RMSNorm](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/nn/layer_norm.py#L210-L236), to explain why the internal computation type and output type are separated; and the example in [PyTorch autograd mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html) showing how invalid operations can contaminate backpropagation even if masked afterward.

<a id="应画或推导的对象-2"></a>

### What to draw or derive

Write `L=(Σ_j S_j)/(Σ_j n_j)`, where `S_j` is the sum of valid negative log probabilities in segment j and `n_j` is its number of valid targets; derive the correct weighting of segment averages. Then draw the state flow `clear gradients → accumulate microbatches → clip → optimizer → scheduler`, making clear that AdamW's momentum states are not model weights.

<a id="两个-cpu-练习-2"></a>

### Two CPU exercises

**E02-A: Expose normalization errors with unequal-length batches.** Construct two small batches with different numbers of valid target tokens and different average difficulty. Calculate the full-batch gradient, the correctly weighted per-microbatch gradient, and the incorrect gradient from local averages. Disable dropout, compare errors using small FP64 or FP32 tensors, and explain the differences. Then add padding, EOS, and fully masked instances. Define whether a batch with no valid targets is rejected or skipped; unexplained zero denominators are prohibited. For a few parameters, also check autograd gradients with central differences, explaining the problems caused by steps that are too large or too small.

**E02-B: Build a collection of numerical and recovery failures.** Compare direct exponentiation with stable cross-entropy using large-magnitude logits, recording finite-value checks. Then compare uninterrupted training with saving and resuming a toy model, restoring weights, optimizer, counters, random state, and data position. Deliberately restore only the weights and observe whether the next update changes. Also inject failures from missing gradient clearing and dividing by batch size twice. Require at least one check that catches each error, instead of inspecting only the final loss.

**GPU extension:** compare precision schemes using identical inputs and parameters; record operator error, gradient deviation, and short-training trends separately. CPU simulation cannot establish the correctness of real FP8/FP4 kernels, communication reductions, or mixed-precision performance.

<a id="验收问题与错误理解-2"></a>

### Self-assessed completion questions and misconceptions

Why can FP64 help without guaranteeing that an incorrect objective is fixed? Why should changing how the same batch is split preserve the mathematical update without necessarily producing bitwise-identical floating-point values? Why are weight decay and an L2 term not freely interchangeable in adaptive optimizers? Why must clipping be checked against the full semantics after accumulation and scaling? Treating low loss as a sign of health, treating continued execution as correct recovery, or treating statistical changes caused by more padding as model progress all fail this check.

**Deliverables:** a loss/gradient comparison table, numerical failure examples, a recovery-state checklist, and minimal counterexamples for correct and incorrect implementations. **Suggested workload:** 18–30 hours. **Next module:** [M03](#m03); only now expand the data pipeline.

<a id="m03"></a>
<a id="m03数据tokenizer-与评估隔离"></a>

## M03 | Data, Tokenizers, and Evaluation Isolation

<a id="先修与可跳过诊断-3"></a>

### Prerequisites and diagnostics for skipping material

Complete the labels/mask checks in M02 and be able to read JSONL and calculate content hashes. Diagnostic: do different URLs for the same web page count as duplicates? Can old token shards be reused after the BPE vocabulary changes? Why might a random document split place copies from the same code repository on both sides? Learners who can provide a data flow and version records with counterexamples may proceed directly to the exercises.

<a id="学习目标细分-3"></a>

### Detailed learning objectives

- Distinguish the responsibilities of collection, extraction, language/domain identification, quality filtering, exact/approximate deduplication, decontamination, and mixture sampling.
- Understand byte encoding, pre-tokenization, BPE merges, and special tokens; compare vocabulary size, compression, language coverage, and model cost, recognizing that fewer tokens do not necessarily imply greater capability.
- Track provenance, licenses/terms of use, document identifiers, processing versions, and filtering reasons; add teacher, prompt, and verifier identities for synthetic content.
- Distinguish document counts, unique content, tokenized tokens, sampled tokens, and valid supervised targets; review mixtures using actual consumption statistics.
- Enforce evaluation isolation across source clusters, question variants, and answers; understand that a zero sampling weight isolates a component but does not prove that near-duplicate content is absent.

<a id="核心阅读按顺序最多三项-3"></a>

### Core reading: in order, no more than three items

1. [CS336 Assignment 4 official handout](https://github.com/stanford-cs336/assignment4-data/blob/main/cs336_assignment4_data.pdf): read the extraction, filtering, and deduplication task definitions. This curriculum uses only a self-created miniature corpus and does not require downloading all of Common Crawl. Revisit the relevant A1 section for BPE mechanics.
2. [OLMo tokenizer and vocabulary conventions](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/data/tokenizer.py#L48-L107), followed by Gates 2, 3, and 5 in the [data and mask handbook](../handbook/01-data-model-pretraining-design.md). Verify that the true token count differs from the number of embedding rows padded for throughput.
3. [Marin ArtifactStep identity implementation](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/lazy.py#L213-L237), together with the [Marin notes](../notes/repositories/marin.md), to draw a dependency graph. Note that a configuration fingerprint does not automatically replace checksums for all source files and data contents.

**Optional reading:** the Lecture 12–14 evaluation and data slides on the CS336 archive page; the DataTrove tokenization and multi-source mixture entry points in the [SmolLM notes](../notes/repositories/smollm.md). Internal paths in source code describe the authors' environment; they are not a promise of publicly downloadable data.

<a id="应画或推导的对象-3"></a>

### What to draw or derive

Draw where a document goes at each filter and why it might be rejected, then show the relationship between training, development, held-out partitions, and duplicate clusters. Work through several BPE merges by hand and specify deterministic tie-breaking rules. Calculate expected token consumption and repetition counts for a small source under a target mixture, and explain why raw byte length cannot directly represent training tokens.

<a id="两个-cpu-练习-3"></a>

### Two CPU exercises

**E03-A: Build an auditable miniature corpus.** Write 60–100 Chinese, English, code, and mathematics samples, including duplicates, paraphrases, template noise, versions from the same source, and evaluation answers. Normalize, perform exact deduplication and approximate matching with character/word-segment sets, and split training and evaluation by source cluster. Label known duplicate pairs and measure false positives, false negatives, and retention. Save provenance, transformation chains, and rejection reasons. Thresholds from this small sample do not establish suitability for internet-scale corpora.

**E03-B: Trace tokenization to supervised targets.** Implement small-vocabulary BPE on the miniature corpus, retaining a byte baseline for comparison. Check round-trip behavior for Chinese and English, indentation, numbers, special markers, and unseen characters. Record bytes/token and length distributions for each category, sample small batches with different token proportions, and explicitly define document attention boundaries, position IDs, and label masks. Finally, trace every valid target to its position in the original text or to a declared special token. Use the held-out set only for the agreed final checks; do not repeatedly tune merges and filtering thresholds against it.

**GPU extension:** fix the tokenizer, model, budget, and evaluation protocol, then compare training results from two data-processing approaches. If you change both the vocabulary and the mixture, acknowledge that attribution is impossible. All data-tracing checks must still pass before GPU training.

<a id="验收问题与错误理解-3"></a>

### Self-assessed completion questions and misconceptions

Why might high-quality filtering harm minority languages? Why do ten repetitions not equal ten times as much new information? Why can per-token loss not directly rank models using different tokenizers? Why are deduplication and decontamination different metrics? Why are packing boundaries more than storage details? You should be able to refute “stricter rules are always better,” “zero matches means no contamination,” and “the tokenizer is just preprocessing and can be changed at any time” with your own counterexamples.

**Deliverables:** a miniature corpus manifest, deduplication and isolation report, tokenizer files and version, sampling statistics, and a trace diagram from original text to targets. **Suggested workload:** 18–32 hours. **Next module:** [M04](#m04).

<a id="m04"></a>
<a id="m04scaling实验设计与完整-pretraining-recipe"></a>

## M04 | Scaling, Experiment Design, and a Complete Pretraining Recipe

<a id="先修与可跳过诊断-4"></a>

### Prerequisites and diagnostics for skipping material

Complete M00–M03, with an explainable toy model, valid loss, and data records. Diagnostic: can you distinguish comparisons at equal token counts, equal theoretical FLOPs, and equal wall-clock time? Why might the best mixture for a small model fail to extrapolate directly? Why might two YAML files with the same name produce different training runs? Learners with a complete experimental report may use it to answer these questions, then proceed directly to reviewing a large-scale recipe.

<a id="学习目标细分-4"></a>

### Detailed learning objectives

- Turn capability, deployment latency, training budget, and data availability into candidate designs; do not choose the largest parameter count first and seek justification afterward.
- Understand scaling laws as empirical models dependent on recipes and distributions; distinguish fitting, interpolation, extrapolation, and validation on points excluded from fitting.
- Design single-factor ablations and necessary interaction experiments, recording repeated seeds, errors, the number of selection attempts, and stopping rules; accept “insufficient evidence” as a conclusion.
- Distinguish total dense parameters from total/active MoE parameters, and computation from storage/communication costs; treat `6ND` as a limited approximation rather than the true budget.
- Trace a formal configuration to its model, tokenizer, data, optimizer, schedule, batch, checkpoint, and evaluation, and understand the interfaces between pretraining, midtraining, and long-context stages.

<a id="核心阅读按顺序最多三项-4"></a>

### Core reading: in order, no more than three items

1. [Pinned CS336 2025 scaling basics lecture](https://github.com/stanford-cs336/spring2025-lectures/blob/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf): understand the curves and experimental design before reading the formulas; use it here only as a conceptual introduction.
2. [The Delphi authors' scaling experiment record](https://openathena.ai/blog/delphi/): focus on the first failure, revisions, and validation at held-out scales. Then compare the [Marin 535B case study](../handbook/04-marin-535b-live-case-study.md), separating historical predictions, current code, and plans still in progress.
3. [OLMo 3 7B pretraining configuration](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py#L40-L108), together with the [OLMo notes](../notes/repositories/olmo-core.md), tracing into later stages. Read the actual parameter groups, dtypes, and duration instead of copying only the learning rate. Comments about a planned extension in the current file must not replace the final run report.

**Optional reading:** the fixed-compute-budget question in the [original Chinchilla paper](https://arxiv.org/abs/2203.15556); the [SmolLM 4K recipe](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/stage1_8T.yaml#L163-L255) and long-context configurations in the existing SmolLM notes, comparing how stages jointly change data and systems. The current main branch of the CS336 [Assignment 3 repository](https://github.com/stanford-cs336/assignment3-scaling) already says Spring 2026 and must not be presented as a pinned 2025 version. Its course API has access conditions; this module does not depend on that service.

<a id="应画或推导的对象-4"></a>

### What to draw or derive

Draw an equal-compute comparison of model size, training tokens, and validation loss under three budgets, using distinct markers for fitting points and held-out points. Create a resource table for candidate models listing parameters, activations, optimizer states, attention intermediates, and estimated communication. Finally, draw the formal recipe's dependency graph, identifying which invalidated artifacts would require retokenization, refitting, or recompilation.

<a id="两个-cpu-练习-4"></a>

### Two CPU exercises

**E04-A: Design and judge a small experiment.** First write the question, single main variable, fixed conditions, evaluation metrics, budget, and result that would overturn the hypothesis. You may use a tiny toy-model budget to compare two widths and three token horizons, with at least two repetitions. If the CPU is too slow, reduce the task; do not report a plan as completed. Separately, practice fitting and extrapolation with an explicitly labeled synthetic table: hide the largest-budget point, compare predictions from different fitting choices, and show a case where a good fit predicts poorly. Real small runs and synthetic fits must have separate plots and conclusions. Neither supports a claim of a transferable frontier scaling law.

**E04-B: Review a public recipe.** For OLMo 3 7B, fill out a table covering model, vocabulary, mix, tokens/update, parameter groups, precision, stopping, recovery, and evaluation. Include source line numbers and distinguish defaults from overrides. Compare with SmolLM long-context or Marin hero, listing three directly comparable quantities and three that cannot be compared directly. Write one page on what the next allocation of budget should validate, without presenting copied large-run parameters as a validated design.

**GPU extension:** execute a small number of preregistered pilots and calibrate the throughput budget with short runs close to the target shapes. Numerical, recovery, and performance checks on an actual cluster belong in M05 and later systems modules. More GPUs do not remove the need for held-out scales and independent evaluation.

<a id="验收问题与错误理解-4"></a>

### Self-assessed completion questions and misconceptions

Why is training-compute optimality not necessarily serving-cost optimality? Why does small-scale stability not prove stability over a long horizon? Why might improved capability scores result from longer inference or a larger agent budget? Why does changing the tokenizer prevent unconditional continuation of an old scaling curve? Can you explain why OLMo's midtraining data and long-context stage are research choices rather than universal steps every model must follow? Treating correlation as causation, curve fitting as a law, or a single-kernel speedup as a speedup for the entire training run all require correction through counterexamples.

**Deliverables:** a preregistered experiment card, separate real and synthetic curves, a public-recipe review table, a resource budget, and a decision memo on next steps. **Suggested workload:** 20–35 hours. **Next module:** [M05: Distributed Parallelism](02-training-systems.md).

<a id="本篇完成后应能独立交付什么"></a>

## What You Should Be Able to Deliver Independently After This Article

You should be able to trace how a document becomes a supervised target, explain causality, loss aggregation, gradient updates, and checkpoint state, then map these to a public recipe while distinguishing hands-on verification, source reading, and missing evidence.

Check in the order deliverable → counterexample → explanation → decision. If you cannot explain the update, return to M02; if the data or metrics are untrustworthy, return to M03. Enter M05 only after small tests pass. Reading the curriculum does not automatically advance personal progress; actual deliverables and independent verification establish completion.
