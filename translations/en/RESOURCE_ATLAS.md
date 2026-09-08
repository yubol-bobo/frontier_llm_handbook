<a id="资源地图在哪个阶段读为什么读读到哪里停"></a>

# Resource atlas: when to read, why to read, and where to stop

Verified: 2026-09-08. See [ROADMAP](ROADMAP.md) for the curriculum sequence. Resources here are organized by learning purpose, not popularity or stars. On the first pass, choose only the core material for each module and consult the rest as specific questions arise.

**Status definitions:** “Pinned source” means cloned, with a recorded SHA and targeted notes; it does not mean the full project has been run. “External entry point” means the official page and learning purpose were checked in this round, but the resource is not among the 19 source snapshots, has not been installed, and its assignments have not been completed. Paper conclusions apply within their experimental conditions; model reports, inference demos, and complete training recipes are identified separately.

<a id="1-主干19-个固定源码项目"></a>

## 1. Backbone: 19 pinned source projects

| Project / pinned-source notes | First encounter | Most useful question to read with | First-pass stopping point |
|---|---|---|---|
| [SmolLM](notes/repositories/smollm.md) | M03–M04 | How do data, stages, length, and budget change together? | Explain the differences between two configurations and account for the token budget |
| [OLMo-core](notes/repositories/olmo-core.md) | M01–M04 | How does one sample become labels, loss, an update, and a checkpoint? | Follow an official recipe into one actual training path |
| [Marin](notes/repositories/marin.md) | M03–M04, M08 | How are research decisions, data lineage, predictions, and run identity preserved? | Trace one artifact DAG, then read the 535B case study |
| [TorchTitan](notes/repositories/torchtitan.md) | M05 | How do meshes, normalization, and state fit together in PyTorch training? | Draw the rank layout and locate gradient and state handoffs |
| [Megatron-LM / Core](notes/repositories/megatron-lm.md) | M05–M06 | What are the actual boundaries of TP/PP/CP/EP and distributed optimizers? | Trace one backend, without reading every branch |
| [DeepEP](notes/repositories/deepep.md) | M06–M07 | How are expert assignments dispatched across devices, computed, and returned? | Explain shapes, streams, metadata, and compatible interfaces |
| [DeepGEMM](notes/repositories/deepgemm.md) | M07 | How do layouts, quantization scales, JIT, and grouped GEMM work together? | Choose one kernel contract and list numerical and performance criteria |
| [SGLang](notes/repositories/sglang.md) | M09 | How do prefill/decode, KV cache, and scheduling change rollout costs? | Trace one request and one cache lifecycle |
| [Open Instruct](notes/repositories/open-instruct.md) | M10–M11 | How are SFT, preference, reward, and RL objectives actually implemented? | Hand-calculate a mask / probability-ratio example and find where it is consumed |
| [verl](notes/repositories/verl.md) | M11, M13 | How are responsibilities divided between the algorithm driver and compute backend? | Fix one algorithm/backend and trace one update |
| [Harbor Cookbook](notes/repositories/harbor-cookbook.md) | M12 | What kinds of real-world tasks provide verifiable feedback? | A contract for one task and its reward vector |
| [Harbor](notes/repositories/harbor.md) | M12, M14 | How do environment, agent, and verifier lifecycles end reliably? | Explain normal, timeout, and failure branches |
| [Verifiers](notes/repositories/verifiers.md) | M12 | Who owns state: task, harness, runtime, or rollout? | Trace one rollout through scoring and artifacts |
| [Pi](notes/repositories/pi.md) | M12 | How do events, tools, context, and durable sessions connect? | Trace one controlled tool call and cancellation/recovery questions |
| [DeepSeek Harness](notes/repositories/deepseek-harness.md) | M12 | How is a log projected into a model request? | Identify the invariants that context and logs must satisfy |
| [Prime RL](notes/repositories/prime-rl.md) | M11 overview, M13 in depth | How do task generation, grouping, filtering, queues, and training connect? | Follow one sample's reward/tokens/version to the learner |
| [slime](notes/repositories/slime.md) | M13 | How do multiple turns, branching, and compaction become training samples? | A counterexample involving shared prefixes and loss masks |
| [Miles](notes/repositories/miles.md) | M13 | How are tokens, expert routes, precision, and policy versions kept aligned? | A method to locate and assess one kind of inconsistency |
| [APEX / SkyRL recipe](notes/repositories/apex-agents-skyrl-recipe.md) | M13 integration | How do long-horizon knowledge-work tasks connect to training? | Identify TITO/Harbor/SkyRL handoffs and gaps from unpublished data |

See [REPO_RELATIONSHIPS](REPO_RELATIONSHIPS.md) for actual software relationships and differing pins. For example, downstream use of Megatron Core does not mean running Megatron's top-level scripts; an independent DeepEP HEAD is not guaranteed to match the adapter being studied.

<a id="2-补足基础数据与系统的外部入口"></a>

## 2. External entry points for foundations, data, and systems

This is not another list to finish all at once; each resource fills a specific prerequisite or evidence gap.

| Primary resource | Modules | What to read first / why it is needed | Status and boundaries |
|---|---|---|---|
| [Stanford CS336 2025](https://cs336.stanford.edu/spring2025/) | M00–M05, M10 | Selected lectures and assignments on basics, systems, scaling, data, and alignment | External entry point; the course itself has mathematics/deep-learning prerequisites. This path supplies a bridge and does not require taking the full course immediately |
| [D2L linear algebra](https://d2l.ai/chapter_preliminaries/linear-algebra.html), [calculus](https://d2l.ai/chapter_preliminaries/calculus.html), [probability](https://d2l.ai/chapter_preliminaries/probability.html) | M00 bridge as needed | Fill only gaps in matrix multiplication, derivatives, and conditional probability identified by the assessment | Authors' textbook entry points; use M00 exit questions to decide when to stop prerequisite review |
| [PyTorch Autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | M01–M02 | Computation graphs, gradient accumulation, and disabling gradient recording | External entry point; the online version changes, so record the experimental environment separately |
| [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) | M04 | Training design, ablations, and data/model decisions | Authors' engineering account; compare with pinned SmolLM configurations |
| [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | M05–M08 | Illustrated explanations of parallelism, recomputation, communication, and memory | External entry point; its performance results are not measurements on your own machine |
| [nanotron](https://github.com/huggingface/nanotron) | M04–M05 optional | The training engine behind SmolLM recipes | External source entry point, not yet cloned; execution requires alignment with the recipe version |
| [datatrove](https://github.com/huggingface/datatrove) | M03, F04 | Data-processing stages and artifacts in `examples/fineweb.py` and the MinHash example | External source entry point; reading a pipeline does not provide the complete corpus |
| [DCLM](https://github.com/mlfoundations/dclm) | M03–M04, F04 | Source selection → processing → tokenize/shuffle → train → eval, and reference JSON records | External source entry point; real-scale work requires the corresponding data and compute |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | M07 | Read with the authors' papers to understand IO, tiling, recomputation, and attention | External source entry point; verify interfaces and supported hardware for the specific version |
| [Official Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | M07 | Vector addition first, then fused softmax / matmul | External tutorials; perform performance exercises only on supported GPUs |
| [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) / [NCCL Tests](https://github.com/NVIDIA/nccl-tests) | M05, M08 | Collective output semantics, message sizes, and bandwidth definitions | External entry points; do not cover every custom EP communication operation |
| [DeepSeek 3FS](https://github.com/deepseek-ai/3FS) | M08, F05 | Design Notes: storage dataflow for training data and state | External source/design entry point; no need to build a storage cluster first |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | M03, M14 | Task definitions, prompts, scoring, and reproducible evaluation | External source entry point; its role differs from an agent execution harness |
| [Inspect](https://inspect.aisi.org.uk/) | M12–M14 | Dataset / Solver / Scorer, logs, sandboxes, budgets, and error handling | External official documentation; actual evaluation also requires a model or controlled mock |

**Course-version trap:** A CS336 annual page and an assignment repository's `main` branch may refer to different years. The current scaling assignment entry point has changed across years and has API-access requirements, so modules prioritize fixed lecture material and locally completable alternatives; a public webpage does not imply access to all instructional services.

<a id="3-论文与作者记录怎样搭配代码读"></a>

## 3. Reading papers and author records alongside code

| Resource | Placement | Questions to answer after reading |
|---|---|---|
| [Chinchilla](https://arxiv.org/abs/2203.15556) | M04 | What are the compute-optimal assumptions, and which variables/costs are not covered together? |
| [Delphi](https://openathena.ai/blog/delphi/) | M04 | How are fitting and validation scales separated, and what changed after the first failure? |
| [Marin 535B public tracker](https://github.com/marin-community/marin/issues/8435) | M04, M08, M15 | What is a prior prediction, conditional plan, current code state, or actual result? |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | M10 | What signals do demonstrations, preferences, and reward models each provide? |
| [DPO](https://arxiv.org/abs/2305.18290) | M10 | What roles do the reference, log probabilities, and preference labels play in the objective? |
| [PPO](https://arxiv.org/abs/1707.06347) / [DeepSeekMath](https://arxiv.org/abs/2402.03300) | M11 | How are the policy ratio, advantage, clipping, and GRPO grouping derived? |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | M11 | Which behavior/exploration problems do different starting points and stages address, beyond memorizing the training sequence? |
| [DeepSeek-V4](https://arxiv.org/html/2606.19348v1) | F01–F02 | How are long-context architecture, numerics, domain training, and capability integration designed together? |
| [LoRA](https://arxiv.org/abs/2106.09685) / [QLoRA](https://arxiv.org/abs/2305.14314) | Optional after M10 | Which parameters are updated, and how do frozen/quantized states affect resources and expressive capacity? |
| [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060) | F01 optional | What different costs and inductive biases arise from sequence mixing and state updates? |

Limit paper reading to the current module's questions; verify theoretical derivations on small tensors first, then trace actual reducers, masks, and communication points in the code. Earlier original papers establish concepts; they do not imply that their recipes remain the default optimum in every lab today.

<a id="4-多模态扩展的两个不同证据入口"></a>

## 4. Two distinct evidence entry points for multimodal extensions

- [Official Molmo2 training repository](https://github.com/allenai/molmo2): Provides entry points for staged training, data processing, and evaluation, suitable for tracing in F03. Its public pipeline starts from a pretrained LLM and vision encoder; “multimodal pretraining” must not be misrepresented as random initialization of all weights. Continue auditing specific training/data content at the selected version.
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) and the [authors' report](https://arxiv.org/abs/2509.17765): Used to understand text/image/audio/video interfaces and architectural decisions such as Thinker–Talker. Released weights, inference examples, and a technical report do not automatically mean complete raw training data and all run recipes are public.

These are external entry points, not pinned clones in this repository. Their purpose is to extend the capability map beyond the core text curriculum, rather than ask readers to immediately launch a large multimodal model.

<a id="5-为什么不继续无限加-repo"></a>

## 5. Why not keep adding repositories indefinitely?

A resource enters the core path only when it can answer: “Which question does it fill, what are its prerequisites, which part should be read, and what evidence should it produce?” Register a new resource's purpose and openness first; it becomes a core source resource after review at a pinned version. This preserves access to frontier updates while preventing the resource count from growing faster than depth of understanding.

See the [contributing guide](CONTRIBUTING.md) for addition rules and the [coverage map](COVERAGE.md) for uncovered areas.
