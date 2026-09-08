<a id="frontier-llm-仓库学习索引"></a>

# Frontier LLM repository study index

Updated: 2026-09-08. All 20 projects have been cloned; initial reading notes refer to the versions in [sources.lock.json](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/sources.lock.json).

The default learning sequence is now unified in [ROADMAP.md](ROADMAP.md): from prerequisites, models, and gradients through data, pretraining, and systems, then into post-training and agent RL. Detailed modules are in the [curriculum](curriculum/01-foundations-to-pretraining.md); resource purposes are in [RESOURCE_ATLAS](RESOURCE_ATLAS.md). The table below retains stable registration numbers for locating older notes; the numbers do not indicate learning order.

<a id="学习等级"></a>

## Learning levels

- **L0 Located:** Found the project and its origin.
- **L1 Initial source reading:** Read one actual implementation/configuration path and can identify its inputs, outputs, and at least one tradeoff.
- **L2 Mechanism tracing:** Traced the normal path and at least one edge case across multiple modules, producing a testable explanation.
- **L3 Experimental validation:** Ran a reproducible experiment for a specific mechanism and recorded its scope and results.
- **L4 Reusable contribution:** Produced validated code, an environment, an evaluation, or research results. External submission is a separate decision.

Levels apply to the recorded study scope, not mastery of the entire repository. The maintainer has completed at least one L1 initial reading for every project; Pi / Harbor Cookbook / Marin have additional targeted study. This is not the reader's course grade.

<a id="总清单"></a>

## Complete index

The table below is the original registration index; each note contains pinned-version links and specific file entry points. New readers should access this material through M00–M15 in ROADMAP and need not read all 20 repositories in order.

| Registration number | Project / notes | Core question | Next hands-on artifact |
|---:|---|---|---|
| 1 | [Pi](notes/repositories/pi.md) | How do the model, tools, steering, and context form a loop? | A controlled two-tool event trace distinguishing completion order from model-message order |
| 2 | [DeepSeek Harness](notes/repositories/deepseek-harness.md) | How are model requests reconstructed from logs after plugins change behavior? | A comparison explaining log reconstruction, failed attempts, and recovery |
| 3 | [Harbor Cookbook](notes/repositories/harbor-cookbook.md) | How do real-world tasks receive verifiable rewards? | Experiment 001 exists; next run reward reading in an actual Trial |
| 4 | [Harbor](notes/repositories/harbor.md) | How do environment startup, agent, verifier, and cleanup form a stable lifecycle? | A Trial timeline with normal/timeout/failure branches |
| 5 | [Verifiers](notes/repositories/verifiers.md) | How do tasks, harnesses, runtimes, and scoring share interfaces? | A small taskset with an explicit evaluator and held-out cases |
| 6 | [Prime RL](notes/repositories/prime-rl.md) | How does orchestration take a rollout into the learner? | A small-model training trace: tokens, reward, policy version, loss |
| 7 | [APEX Agents Recipe](notes/repositories/apex-agents-skyrl-recipe.md) | How do large knowledge-work tasks enter asynchronous training? | Diagram adapter boundaries and replace unpublished training data with your own/public tasks |
| 8 | [SmolLM](notes/repositories/smollm.md) | How do data mixtures, model architecture, and staged training jointly determine a recipe? | Design one targeted ablation with a fixed token budget |
| 9 | [OLMo-core](notes/repositories/olmo-core.md) | How does actual pretraining move from data configuration into the training lifecycle? | Trace data → batch → loss → checkpoint → eval |
| 10 | [Open Instruct](notes/repositories/open-instruct.md) | How do SFT and RL differ in data, probabilities, and masking? | Construct logprob/mask tensors and explain ratios and corrections |
| 11 | [Marin](notes/repositories/marin.md) | How can model development become a traceable experiment and execution graph? | Define a data-variant experiment and trace caches/versions/dependencies |
| 12 | [TorchTitan](notes/repositories/torchtitan.md) | How do parallelism, recomputation, precision, and checkpoints fit together? | Change one configuration at a time and compare effective tokens, memory, and throughput |
| 13 | [Megatron-LM / Core](notes/repositories/megatron-lm.md) | How do TP/PP/CP/EP process groups and communication work together? | Draw a small topology and MoE dispatch/combine dataflow |
| 14 | [verl](notes/repositories/verl.md) | How are algorithm control and distributed computation expressed separately? | Trace workers, data distribution, and loss for one PPO/GRPO batch |
| 15 | [slime](notes/repositories/slime.md) | How are branching agent trajectories converted into valid training samples? | Check actual allocation of shared prefixes, tool masks, and outcome rewards |
| 16 | [Miles](notes/repositories/miles.md) | How do token/routing/precision/version inconsistencies affect RL? | Check one explicit TITO or routing-replay invariant |
| 17 | [SGLang](notes/repositories/sglang.md) | How do request scheduling, KV reuse, compute, and communication affect rollout cost? | Compare metrics for workloads with and without shared prefixes |
| 18 | [DeepGEMM](notes/repositories/deepgemm.md) | How do low precision, grouped GEMM, and JIT determine kernel efficiency? | Run a shape/numerical-error/throughput sweep on a supported GPU |
| 19 | [DeepEP](notes/repositories/deepep.md) | How are MoE tokens dispatched and combined across GPUs? | Analyze dispatch/combine shapes, correctness, and communication time |
| 20 | [Claude Code / Agent SDK](notes/repositories/claude-agent-sdk.md) | How does a complex harness manage context, permissions, and recovery? | A pinned SDK call chain, snapshot evidence boundaries, and CPU failure traces |

<a id="怎样把索引变成学习产物"></a>

## Turning the index into learning artifacts

Choose only one main question from the current module at a time. Read the concepts, trace the specified code, then perform a small exercise that can verify or refute the explanation; record uncertainty and version boundaries in your notes. The core path does not require studying six RL frameworks in depth simultaneously or connecting asynchronous agent RL before learning gradients.

Each module's prerequisites, ordered core reading, two exercises, and completion criteria are defined in the [course roadmap](ROADMAP.md), avoiding a second stage sequence here. The existing [full training process](handbook/00-end-to-end.md) explains how model development happens; [Marin 535B](handbook/04-marin-535b-live-case-study.md) provides examples of real decisions.

Use the [learner template](templates/learner-progress.md) for self-study records. Research and experiments actually completed in this repository remain recorded in [PROGRESS.md](PROGRESS.md); do not record proposed exercises as executed.
