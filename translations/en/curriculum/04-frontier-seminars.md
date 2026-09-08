<a id="前沿专题核心课程之后怎样继续深入"></a>

# Frontier Topics: Going Deeper After the Core Curriculum

[Back to the roadmap](../ROADMAP.md) · [Resource map](../RESOURCE_ATLAS.md) · [Coverage status](../COVERAGE.md)

Version: 2026-09-08. F01–F06 are seminar designs with prerequisites, questions, and deliverables. External source entry points have been checked, but not all source code has been pinned or experiments completed. These support further study; this page is not six completed advanced textbooks. Choose one topic at a time and initially spend 6–12 hours on evidence reading and a proposal for each group (a planning estimate; experiments require additional time).

<a id="f01"></a>

<a id="f01-长上下文新-attention-与混合架构"></a>

## F01 Long Context, New Attention Mechanisms, and Hybrid Architectures

**Prerequisites:** M01/M04/M05; add M06 when discussing sparse models. Performance conclusions require M07/M09.

Read [long-context design](../handbook/01-data-model-pretraining-design.md), the [DeepSeek-V4 architecture and systems report](https://arxiv.org/html/2606.19348v1), and optionally [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060), in that order. The first provides recipe and engineering questions; the latter two offer different sequence-computation designs. Do not combine them into an unvalidated universal architecture.

Study questions: what state do prefill and token-by-token decoding each retain? What information tradeoffs do sparse selection, KV compression, or recurrent state make? After changing positional encoding and training length, is distant information actually used? What support is required in training, inference, and checkpoint formats?

**Exercise:** for identical sequence length, batch, and dtype, draw the state/data flows of two approaches, calculate theoretical storage term by term, and list omitted costs. Then design an evaluation covering short-task regressions, cross-segment reasoning, and position changes. The ledger and evaluation design require no GPU, but cannot support a claim of measured acceleration.

**Deliverables and self-assessed completion:** an architecture comparison, at least one question that complexity formulas cannot answer, and a list of details the sources do not disclose. Distinguish algorithmic approximation, implementation optimization, and capability evidence; only then select one pinned implementation for a small experiment.

<a id="f02"></a>

<a id="f02-领域-specialist蒸馏与-test-time-compute"></a>

## F02 Domain Specialists, Distillation, and Test-Time Compute

**Prerequisites:** M09–M11/M14. First be able to calculate KL and probability ratios by hand and explain sampling budgets.

Read the [post-training branches](../handbook/03-posttraining-agent-rl-evaluation.md), then specialists and on-policy distillation in [DeepSeek-V4 §5](https://arxiv.org/html/2606.19348v1#S5). Focus on student-sampled trajectories, teacher distributions, capability integration, and execution cost, rather than simply remembering the number of teachers.

**Exercise A:** calculate both KL directions for distributions over three tokens, swap teacher/student, and explain the difference. Diagram the data sources for demonstration SFT, offline distillation, and on-policy distillation. **Exercise B:** compare budget-matched inference using one long generation, multiple candidates, and verifier selection, defining success rate, cost, failure counts, and confidence intervals in advance. Actual model sampling requires separate resources; a paper design is not evidence of improved capability.

**Self-assessed completion:** does a stronger teacher guarantee a stronger student? How should increases in inference tokens, improvements to the verifier, and weight changes be controlled separately? How do numerical paths in quantized training and rollouts affect probabilities? Deliver a falsifiable experiment design and a teacher/actor/learner state diagram, then proceed to a small integration experiment.

<a id="f03"></a>

<a id="f03-图像视频音频扩展文本主干"></a>

## F03 Images, Video, and Audio: Extending the Text Backbone

**Prerequisites:** M01–M04/M09/M10; add M05/M08 for long video. First bridge representation concepts such as patches, temporal sampling, and visual/audio encoders.

Reading order: begin with [training stages in the official Molmo2 README](https://github.com/allenai/molmo2#training-and-evaluations), and develop source questions along `launch_scripts/pretrain.py`, `launch_scripts/sft.py`, and the data-preprocessing entry points. Then use the [Qwen3-Omni authors' report](https://arxiv.org/abs/2509.17765) and [official repository](https://github.com/QwenLM/Qwen3-Omni) to compare streaming audiovisual inputs and outputs. These are external entry points and have not currently been cloned or run.

Make the starting point explicit: Molmo2's public multimodal training pipeline begins with a trained LLM and visual encoder. Reproducing that stage is not equivalent to randomly initializing every weight. Qwen's inference code, model descriptions, and report likewise cannot automatically supply the complete original training corpus.

**Exercise:** use a few images/videos you create yourself to explain frame sampling, resolution, temporal information, and token budgets. Draw a shape/state diagram for preprocessor → encoder → connector → LM → output. When splitting video across datasets, check for leakage from the same video and adjacent segments. Begin with data and interface audits; downloading a large model is unnecessary. If extending to training, first specify which components are frozen and which are updated.

**Self-assessed completion and deliverables:** a data card, a multimodal sample diagram, a stage initialization/freezing table, and a protocol for separately scoring spatial/temporal localization and ordinary question answering. Distinguish evidence that inputs connect, training converges, and the model understands temporal order, then trace a complete source chain.

<a id="f04"></a>

<a id="f04-数据质量合成数据与数据混合研究"></a>

## F04 Research on Data Quality, Synthetic Data, and Data Mixtures

**Prerequisites:** M03/M04/M14; online generation also requires M09, and trajectory data requires M12.

Read [datatrove data-processing examples](https://github.com/huggingface/datatrove), then the [DCLM workflow](https://github.com/mlfoundations/dclm#workflow-overview-and-exp-data), and finally revisit the [Marin data-budget case study](../handbook/04-marin-535b-live-case-study.md). The sequence is data artifacts first, controlled comparisons second, and stage consumption budgets last.

**Exercise:** construct a small corpus containing exact duplicates, near-duplicate templates, rare valid samples, and evaluation variants. Compare the retention rates, false removals, and missed removals of two filtering rules. Add teacher/prompt/verifier identifiers to synthetic samples and design a held-out set that does not rely on the teacher's familiarity with the questions. If teacher access is unavailable, verify the pipeline with handwritten samples first; do not call this a real synthetic-data gain.

**Self-assessed completion:** what should you do if quality scores rise while coverage falls? How do you separate more training tokens from more information? Which model, compute, and evaluation conditions should data comparisons fix? Deliver a processing pipeline, sample-level provenance, domain distributions, and a preregistered ablation, then validate on a small model.

<a id="f05"></a>

<a id="f05-训练基础设施存储编译调度和运维"></a>

## F05 Training Infrastructure: Storage, Compilation, Scheduling, and Operations

**Prerequisites:** M05/M07/M08; add M09/M13 when studying rollout scheduling.

Read [Marin's actual operations](../handbook/04-marin-535b-live-case-study.md), [3FS Design Notes](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md), then choose the [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) or [NCCL Tests](https://github.com/NVIDIA/nccl-tests) according to the bottleneck. Place training steps, data reads, saves, and evaluation on the same timeline.

**Exercise:** use public records to create a causal diagram for a hypothetical incident, distinguishing observed facts, candidate explanations, and the next evidence to collect. Estimate total checkpoint state and lower bounds on read/write time, then design a check for each of cache invalidation, node interruption, and compilation restart. CPU simulation shows only scheduling/state semantics; it cannot establish actual storage and network performance.

**Self-assessed completion and deliverables:** topology and data flow, a fault runbook, recovery-success conditions, and uncovered risks. Explain why normal operation within one rack does not prove stability across the full topology and why completed asynchronous staging does not mean completed persistence, then proceed to controlled multi-machine experiments.

<a id="f06"></a>

<a id="f06-长期-agent记忆与持续学习先明确改变了什么"></a>

## F06 Long-Running Agents, Memory, and Continual Learning: Specify What Changed First

**Prerequisites:** M12–M14; M11/M13 must be completed if weight updates are involved.

Start with [Pi state and context](../notes/repositories/pi.md), [Verifiers task lifecycles](../notes/repositories/verifiers.md), and [Prime RL sample and version management](../notes/repositories/prime-rl.md). These provide local mechanisms, not a complete lifelong-learning recipe. Long-term generalization and stable online weight learning still contain many open questions in the current field.

**Exercise:** design four conditions for the same task spanning multiple rounds: fixed context, external persistent memory, retrieval of existing material, and actual parameter updates. Specify stored content, visible information, reset conditions, temporal splits, and evaluation budgets. Verify information flow with controlled records first, then decide whether a model experiment is needed. Better performance on the next task alone does not establish that the model learned through weight updates.

**Self-assessed completion and deliverables:** a state-ownership table, a long-term evaluation protocol, and counterexamples involving forgetting, contamination, and feedback bias. Distinguish task-state accumulation, in-context adaptation, data retrieval, and optimizer updates before reviewing which part of the problem a public work actually solves.
