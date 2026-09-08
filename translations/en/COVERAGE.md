<a id="知识覆盖与缺口"></a>

# Knowledge coverage and gaps

Updated: 2026-09-08. The goal is to present a complete map of end-to-end frontier LLM training knowledge and the material needed to build and advance engineering skills, with a learning path and inspectable artifacts for each item. **A complete coverage map, a curriculum design, instructional explanations, source tracing, and actual experiments are different levels of maturity.** This page does not conflate them into a single percentage.

<a id="状态口径"></a>

## Status definitions

- **Curriculum:** Prerequisites, reading sequence, exercises, and completion criteria have been designed; this does not mean the exercises are complete.
- **Explanation:** A handbook chapter / lesson provides a continuous explanation or actual case study.
- **Source:** A targeted review exists at a pinned SHA; this does not mean the entire project has been audited.
- **Experiment:** Commands, results, and scope are recorded for work actually executed in this repository; GPU and cluster validation are counted separately.
- **Entry point:** A primary source has been checked, but pinned-source review or full instructional development is not yet complete.
- **Gap:** Not yet explored in depth, missing public material, or outside the current course scope; content is not fabricated to fill it.

<a id="能力地图"></a>

## Capability map

| Capability area | Path | Current depth of evidence | Next valuable addition |
|---|---|---|---|
| Python, mathematics, probability, and automatic differentiation prerequisites | M00–M02 | Curriculum and external tutorials; CPU evidence is limited to manually implemented gradients, finite differences, and normalization in the first lesson | Step-by-step derivations and experiments for automatic differentiation and a complete decoder |
| Transformer, causal attention, positional encodings, initialization | M01–M02 | Curriculum, targeted source review | Complete training record for our own small decoder |
| Loss masks, gradients, global normalization | M02, M05, M11 | Explanation, source, CPU Experiment 002 | Numerical comparison with actual DP |
| Data sources, extraction, quality, deduplication, contamination | M03 | Curriculum, explanation, targeted source review; datatrove/DCLM entry points | A reproducible small data pipeline and analysis of false-positive removal |
| Tokenizers, packing, order, and effective-token counting | M01/M03 | Curriculum, explanation, source | End-to-end tokenization/packing checks |
| Scaling, ablations, budgets, statistical uncertainty | M04 | Curriculum, explanation, public run records | Our own prediction experiment at a held-out scale |
| Dense/MoE architectures and routing | M04/M06 | Curriculum, explanation, source | Small MoE output/gradient and load experiments |
| TP/PP/CP/DP/FSDP and optimizer sharding | M05 | Curriculum, explanation, source | Multi-GPU updates and communication traces |
| Low precision, quantization state, kernels, compilation | M07 | Curriculum, explanation, source; Triton/FA entry points | Error and steady-state performance measurements on supported hardware |
| Network topology, communication overlap, memory/bandwidth accounting | M05–M08 | Curriculum, explanation, source | Small-/large-message and tail-latency assessment on a real cluster |
| Training scheduling, storage, failures, recovery, shard migration | M08, F05 | Curriculum, Marin case study, source; 3FS entry point | Controlled recovery comparison after fault injection |
| Long context and targeted training stages | M04/M06, F01 | Explanation, configuration arithmetic, source | Budget-matched comparisons of short-/long-context capabilities and load |
| Serving, KV, prefill/decode, batching | M09 | Curriculum, explanation, source | End-to-end latency/throughput under a controlled workload |
| SFT, preferences, reward models | M10 | Curriculum, explanation, source | Full-parameter baseline on small data; LoRA/QLoRA as separate optional reading |
| RL objectives, advantages, KL, sampling, synchronous updates | M11 | Curriculum, explanation, source | One traceable loop with an actual small model |
| Harnesses, tools, state, compaction, recovery | M12 | Curriculum, explanation, source | Event-replay experiment with a controlled provider |
| Sandboxes, tasks, verifiers, reward aggregation | M12/M14 | Curriculum, source, CPU reward Experiment 001 | Actual Trial lifecycle and scoring isolation |
| Asynchronous RL, policy versions, TITO, routing replay | M13 | Curriculum, explanation, source | Queue simulation → comparison of actual training/inference versions |
| Independent evaluation, judges, costs, errors, safety regressions | M03/M14 | Curriculum, explanation; lm-eval/Inspect entry points | Local evaluation suite, scorer audit, and repeated sampling |
| Teachers, synthetic data, distillation, test-time compute | F02/F04 | Reading guidance and primary reports; some Open Instruct evidence | Complete controlled comparison of data generation/filtering/training/evaluation |
| New attention mechanisms, SSMs, hybrid architectures | F01 | Existing long-context explanation, primary entry points for new architectures | Pinned implementation, representative shapes, and quality comparisons |
| Images/video/audio, multimodal alignment | F03 | Topic study design, Molmo2/Qwen3-Omni entry points | Training pipeline at a pinned version and review of small visual samples |
| Experiment management, code review, reproducibility, technical writing | M00/M04/M15 | Templates, versioned notes, real project decision cases | A project independently verified by another learner |
| Continual learning, online adaptation, long-term agent memory | F06 | Problem definitions and public resource entry points | Experiments that rigorously distinguish contextual memory, retrieval, and weight learning |
| Complete reproduction at very large scale | Advanced direction for M15 | Not executed; gaps in data, compatible environments, and compute | Validate progressively only once material and budget are available |

See the [course roadmap](ROADMAP.md) for module details and the [frontier seminars](curriculum/04-frontier-seminars.md) for F01–F06. Interfaces among model training, data, and task systems are covered in [cross-layer contracts](notes/connections/end-to-end.md).

<a id="开放材料仍无法自动回答的问题"></a>

## Questions that open material still cannot automatically answer

A closed lab's complete data mixture, human-feedback process, full training environment, internal filtering rules, unpublished negative experiments, operating costs, and final recipe generally cannot be reconstructed by assembling external code. Support for a feature in an open-source library also does not prove that a particular model used it. Some public recipes omit training data or retain the authors' internal paths; APEX is a concrete example in the current material.

Multimodal understanding and generation require additional data, encoders, and evaluations. The current core course still focuses on text/code, autoregressive LLMs, post-training, and agent systems. Robotics control, diffusion-based image/video generation, and chip circuit design are outside the core training curriculum, so that “as complete as possible” does not become an unbounded technology encyclopedia.

<a id="下一轮补全顺序"></a>

## Priorities for the next round

1. First expand M01–M04 from “curriculum + external readings” into more complete lessons and small experiments; these are the entry requirements for outside learners to understand later code.
2. Add actual distributed numerical comparisons for M05 and recovery experiments for M08, then advance the small loops in M09/M11.
3. Connect M12 task scoring and trajectories to M13, retaining a synchronous baseline before studying asynchronous execution.
4. After completing one core path, pin sources and add instructional material and completion criteria for F01–F06 one by one.

Curriculum maintenance progress and learners' personal progress are stored separately; adding a document does not mark the corresponding model training or course module as completed.
