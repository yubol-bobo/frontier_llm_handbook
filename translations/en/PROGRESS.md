<a id="内容建设与维护者学习进度"></a>

# Content development and maintainer study progress

Updated: 2026-09-11. The project is in its first round of study and has not completed the entire field or reproduced any complete large-model training run.

This page records the maintainer's actual work. Readers enter the course through [ROADMAP](ROADMAP.md) and use the [personal learning template](templates/learner-progress.md); completing curriculum design does not mean an individual has completed the course. See [COVERAGE](COVERAGE.md) for the maturity of knowledge coverage.

Following the user's explicit positioning, “end-to-end frontier LLM training knowledge” and “engineering foundations and advancement” have been aligned across repository entry points, capability levels, study methods, and completion templates. This positioning update is documentation work; experiment and module completion status retain their original records.

Main repository: [yubol-bobo/frontier_llm_handbook](https://github.com/yubol-bobo/frontier_llm_handbook), with `origin/main` as the main line going forward. The local study directory retains `frontier-llm-lab`; upstream clones are restored from the manifest and pinned SHAs.

<a id="当前状态"></a>

## Current status

| Item | Completed | Not completed |
|---|---|---|
| Learning repository | Local Git, main GitHub repository, study index, methods, and templates | Continue updating this main repository as study progresses |
| Interactive website | Chinese–English interface and 57 documents, terminology table, light/dark themes, course cards, reading tracking, dynamic knowledge connections, interactive mathematics experiments, local notes and backups, GitHub Pages deployment | More instructional units, actual model experiments, and learner feedback |
| Source acquisition | 21/21 shallow clones, pinned SHAs, source index | Full history, submodules, large LFS files, weights, and data have not been downloaded |
| Source reading | Initial reading of at least one implementation/configuration path for 21/21 projects | Whole-repository reading and runtime validation |
| Cross-project relationships | 5 connection topics, relationship index, conceptual knowledge tree v0.5 | Actual installation compatibility for dependency combinations |
| Full-process research | Overview + detailed chapters on data design, distributed operations, and post-training + Marin 535B case study | Complete training execution, independent reproduction of authors' results, and closed-source recipes |
| Public curriculum | 16 core modules M00–M15, 32 exercise designs, F01–F06 seminar designs | More complete lessons, exercise answers, and actual execution for each module |
| Introductory teaching | First lesson: probability, manually implemented gradients, masks, global token averaging | Instructional experiments for automatic differentiation and a complete decoder |
| Experiments | 4 local CPU experiments, with scripts and results saved | Model APIs, Docker, GPUs, complete RL training |

<a id="逐仓库进度"></a>

## Per-repository progress

`L2` and `L3` are limited to the targeted questions listed. Detailed evidence, proposed experiments, and questions are saved in the individual notes.

| Order | Project | Current reading level | Experiments |
|---:|---|---|---|
| 1 | [Pi](notes/repositories/pi.md) | L2: low-level loop, tool ordering, steering, transform boundaries | Controlled event trace pending |
| 2 | [DeepSeek Harness](notes/repositories/deepseek-harness.md) | L1: input → log → request and invariants | Keyless replay pending |
| 3 | [Harbor Cookbook](notes/repositories/harbor-cookbook.md) | L2: tasks, scoring, example interfaces | Targeted L3: Python multidimensional reward assertions executed |
| 4 | [Harbor](notes/repositories/harbor.md) | L1: Trial / verifier / lifecycle | Docker Trial pending |
| 5 | [Verifiers](notes/repositories/verifiers.md) | L1: v1 rollout, Task, Pi/Harbor adapters | Online/offline scoring comparison pending |
| 6 | [Prime RL](notes/repositories/prime-rl.md) | L1: orchestration, trace → batch, GRPO | Small-model loop pending |
| 7 | [APEX recipe](notes/repositories/apex-agents-skyrl-recipe.md) | L1: TITO, Harbor generator, async entry points | Original training data missing; substitute-task experiment pending |
| 8 | [SmolLM](notes/repositories/smollm.md) | L1: stage YAML, data and SFT examples | Token budget calculated; training experiment pending |
| 9 | [OLMo-core](notes/repositories/olmo-core.md) | L1: official recipe, data, train module | Small-model mechanism experiment pending |
| 10 | [Open Instruct](notes/repositories/open-instruct.md) | L1: ratio, rho, masks, synchronization | Tensor-level loss experiment pending |
| 11 | [Marin](notes/repositories/marin.md) | Targeted L2: artifact DAG, 535B configuration, and data/communication/recovery handoffs | Static budget accounting; execution and DAG/cache experiments pending |
| 12 | [TorchTitan](notes/repositories/torchtitan.md) | L1: training step and declarative sharding | Sharding/throughput experiment pending |
| 13 | [Megatron-LM](notes/repositories/megatron-lm.md) | L1: training entry points, PP, MoE dispatcher | Parallel-topology and communication experiments pending |
| 14 | [verl](notes/repositories/verl.md) | L1: driver, advantage, Megatron engine | Numerical and control-flow experiments pending |
| 15 | [slime](notes/repositories/slime.md) | L1: trajectory branches, shared prefixes, training backend | Mask/prefix experiment pending |
| 16 | [Miles](notes/repositories/miles.md) | L1: TITO, routing replay, engine | Consistency experiment pending |
| 17 | [SGLang](notes/repositories/sglang.md) | L1: scheduler, cache, DeepEP/GEMM entry points | Shared-prefix experiment pending |
| 18 | [DeepGEMM](notes/repositories/deepgemm.md) | L1: JIT, layouts, Mega MoE benchmark | Kernel experiment on supported hardware pending |
| 19 | [DeepEP](notes/repositories/deepep.md) | L1: V2 Buffer, dispatch/combine, streams | Multi-GPU communication experiment pending |
| 20 | [Claude Code / Agent SDK](notes/repositories/claude-agent-sdk.md) | L1: CLI transport, permission callbacks, input-stream lifecycle; targeted external historical snapshot reading | Original CPU state machine executed; real Claude, SDK integration, and recovery remain untested |
| 21 | [SoL-Pi](notes/repositories/sol-pi.md) | Targeted L2: four mechanisms, cache cost, Pi version, and failure boundaries | 8 upstream cost-function scenarios pass; Windows upstream tests pass 134/139, with 5 failures retained in the record |

<a id="下一次从这里继续"></a>

## Resume here next time

**Current content-development unit: turn the M01 decoder and sample path into the next complete lesson.**

The first lesson, [from one token to one update](lessons/01-one-token-to-update.md), is complete; Experiment 002's finite-difference, mask, and normalization counterexamples passed. It validates only limited CPU mathematics and does not automatically upgrade any upstream project to “complete training validated.”

1. Following the objectives of [M01](curriculum/01-foundations-to-pretraining.md#m01), write a decoder lesson with shapes, causal masks, and failure examples, keeping the path accessible on an ordinary computer.
2. Select one data → loader → labels/masks → loss → optimizer → checkpoint path from an official OLMo-core recipe; identify each object's shape, identity, and token-counting rules.
3. Extend the first lesson's numerical objective with actual decoder forward/backward checks; execute once an isolated environment is available and record the actual scope and differences from the existing experiment.
4. Revisit the same interface question in Marin / Megatron / TorchTitan, distinguishing research recipes from training implementations.

**Retained agent RL continuation point: Pi's events, state, and recovery.**

1. Verify the Pi SHA: `6160683a4a8012f0d1cd30c145df18b4ca6f5176`.
2. Move from the low-level loop in the [Pi notes](notes/repositories/pi.md) to `packages/agent/src/harness/runtime/drive/generation.ts`, `runtime/reducer.ts`, and the session JSONL path.
3. Choose one concrete question: when a user cancels after a tool has started but before its result is persisted, what determines whether the next recovery retries or ends? First read the call chain and existing targeted tests.
4. Produce a second session record; once dependencies are available, use a faux provider for a CPU event trace, avoiding real model costs.
5. Then move to actual reward loading in Harbor and compare it with the raw reward dict results of Experiment 001.

<a id="需要后续核实的连接"></a>

## Connections requiring later verification

- Prime RL pins a Verifiers submodule; the independent HEAD differs, so align dependencies before actual training.
- The npm release used by the Verifiers Pi adapter differs from the independent Pi HEAD.
- SoL-Pi locks Pi 0.84.2 for development; the independent Pi 0.85.1 snapshot has not been verified as a compatible pairing.
- Open Instruct pins an OLMo-core commit and has 4 undownloaded LFS test-data files.
- Cookbook harbor_rl uses a Harbor feature branch; main lacks the module of that name.
- APEX requires unpublished original training data and a specified SkyRL checkout; the material for a complete reproduction of the original recipe is not currently available.
- The reviewed Megatron DeepEP Buffer adapter and DeepEP V2 mainline need an explicit version relationship.

These do not prevent continued code reading or independent CPU experiments; they limit specific end-to-end reproductions.

<a id="已保存的学习记录"></a>

## Saved study records

- [SoL-Pi: four mechanisms, cache costs, and partial verification](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-11-sol-pi.md)
- [Claude Code harness: source, historical snapshot, and CPU state machine](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-claude-code-harness.md)
- [Bilingual website and terminology review](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-bilingual-website.md)
- [Website visual and interaction upgrade](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-website-redesign.md)
- [Interactive learning website development](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-website.md)
- [Public curriculum and first lesson development](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-curriculum.md)
- [Research on the full large-scale training process](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-end-to-end.md)
- [First overall study record](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08.md)
- [Training-group reading record](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-training.md)
- [RL-group reading record](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-rl.md)
- [Infrastructure-group reading record](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-infra.md)
- [Experiment 001](experiments/001-harbor-reward-contract/README.md)
- [Experiment 002](experiments/002-token-weighted-loss/README.md)

This repository has no scheduled tasks or background training configured. Resume from the current unit next time; do not change “planned” to “completed.”
