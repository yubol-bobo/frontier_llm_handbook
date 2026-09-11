# Frontier LLM Handbook

**This learning resource systematically covers end-to-end training for frontier LLMs, helping learners build engineering foundations and progressively develop the ability to implement, validate, diagnose, and improve training systems.**

It is for beginners entering the field and engineers and researchers seeking deeper training and systems skills, especially learners without access to a frontier lab. Drawing on public primary papers, source code, training recipes, and run records, it connects data, model design, pretraining, post-training, inference, and evaluation into a bilingual Chinese–English learning path with prerequisites, exercises, and completion criteria.

Created: 2026-09-08. Curriculum v0.1: **16 core modules, 6 frontier topics, and 21 pinned source repositories**, supported by external primary resources, five end-to-end explanatory chapters, and a runnable first lesson. Curriculum design, actual source review, and executed experiments are labeled separately; the full coverage map and remaining gaps are maintained publicly.

Main repository: [yubol-bobo/frontier_llm_handbook](https://github.com/yubol-bobo/frontier_llm_handbook), primary branch: `main`. Learning notes, experiments, and the knowledge tree will continue to accumulate here; the local working directory currently retains the name `frontier-llm-lab`.

**[Open the interactive learning website →](https://yubol-bobo.github.io/frontier_llm_handbook/)** Read by module, search the full text, explore connections between concepts, adjust training formulas interactively, and save personal notes and completion records. The website is generated from this repository and published to GitHub Pages as `main` changes; see the [website development guide](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/website/README.md).

[Chinese website](https://yubol-bobo.github.io/frontier_llm_handbook/?lang=zh#/learn) · [English website](https://yubol-bobo.github.io/frontier_llm_handbook/?lang=en#/learn). Switch the interface and all published reading materials using the selector at the top. Both languages share your personal records; see the [bilingual terminology table](GLOSSARY.md).

<a id="两条贯穿全程的学习目标"></a>

## Two learning goals throughout the course

- **End-to-end training knowledge:** Understand the relationships among objectives and budgets, data and tokenizers, architecture and scaling experiments, pretraining and staged training, SFT/preference optimization/RL, inference, evaluation, and release. Explain each stage's inputs, outputs, design rationale, and tradeoffs.
- **Engineering foundations and advancement:** Start with mathematics, tensors, gradients, Git, and reproducible experiments, then progressively master data pipelines, training loops, distributed training and MoE, performance analysis, failure recovery, and agent systems. Assess progress through implementation, measurement, debugging, and integrated projects.

Every module must answer both “Why train this way?” and “How do we implement and validate it?” See the [learning roadmap](ROADMAP.md) for learning artifacts and capability levels, and the [coverage map](COVERAGE.md) for the actual depth of the material developed so far.

<a id="从这里开始"></a>

## Start here

| Entry point | Purpose |
|---|---|
| **[Learning roadmap: start here](ROADMAP.md)** | Entry assessment, the M00–M15 sequence, prerequisite graph, and paths for different specializations and hardware |
| [Resource atlas](RESOURCE_ATLAS.md) | Why to read each resource, which module uses it, where to stop, openness, and snapshot status |
| [Knowledge coverage and gaps](COVERAGE.md) | Distinguish curriculum design, instructional explanations, source review, measurements, and material that is not public |
| [First lesson: from one token to one update](lessons/01-one-token-to-update.md) | Understand probability, loss, gradients, masks, and normalization across shards on an ordinary computer |
| [Training a very large LLM from scratch: the full process](handbook/00-end-to-end.md) | From objectives, data, and scaling pilots to distributed pretraining, agent RL, and release; includes four detailed chapters |
| [Repository study index](LEARNING_LIST.md) | Registration numbers and source notes for 21 pinned projects; the numbers do not indicate prerequisite order |
| [Knowledge tree](KNOWLEDGE_TREE.md) | Projects organized by concept: from tasks and data to training, scheduling, and GPU kernels |
| [Repository relationship map](REPO_RELATIONSHIPS.md) | Distinguish actual dependencies, optional backends, example integrations, project lineage, and conceptual correspondences |
| [Content development and maintainer progress](PROGRESS.md) | What this repository has actually completed; readers use a separate learning record template |
| [Source snapshots](SOURCE_INDEX.md) | Local directories, pinned commits, clone completeness, and version boundaries |
| [How to study](HOW_TO_STUDY.md) | How to read code, run experiments, take notes, and update the knowledge tree in each session |
| [Glossary](GLOSSARY.md) | Shared concepts connecting training and systems design |
| [Contributing guide](CONTRIBUTING.md) | How to add resources, lessons, and experiments with instructional value |

<a id="学习顺序"></a>

## Learning sequence

**Foundations and gradients → data and scaling → distributed training and MoE → kernels/operations/recovery → inference → SFT/preferences → synchronous RL → harnesses/tasks/trajectories → asynchronous agent RL → independent evaluation and an integrated project.** Evaluation principles apply from the data stage onward, with a final comprehensive assessment at the end.

| Course | Modules | Core deliverables |
|---|---|---|
| [Foundations to pretraining](curriculum/01-foundations-to-pretraining.md) | M00–M04 | Mathematical foundations, decoder, correct updates, data, and experimental design |
| [Training and inference systems](curriculum/02-training-systems.md) | M05–M09 | Parallel layouts, MoE, numerics/performance, recovery, and serving |
| [Post-training and agents](curriculum/03-posttraining-and-agents.md) | M10–M15 | Objectives, synchronous to asynchronous execution, environments/trajectories, evaluation, and capstone |
| [Frontier topics](curriculum/04-frontier-seminars.md) | F01–F06 | New architectures, distillation, multimodality, data research, deeper infrastructure, and continual learning |

Each core module includes prerequisites and a skip assessment, ordered reading, two exercises, conditions for GPU extensions, completion criteria, and artifacts. An ordinary computer is enough to start with mathematics, code, and small experiments; actual GPU performance and reproduction at scale require separate validation. See [ROADMAP](ROADMAP.md) for the full schedule and specialized paths.

<a id="当前成果"></a>

## Current results

- 21/21 repositories cloned, all with recorded origins and pinned SHAs.
- 21 initial project reading notes, each tracing at least one concrete code/configuration path; these are not whole-repository audits or claims of having “finished learning.”
- Further detail on Pi's low-level agent loop and selected task/scoring interfaces in Harbor Cookbook.
- Executed [Experiment 001: Harbor multidimensional rewards](experiments/001-harbor-reward-contract/README.md) and [Experiment 002: token loss and gradient normalization](experiments/002-token-weighted-loss/README.md), both limited CPU experiments; no full agent RL or large-model training run has been performed.
- Added five [chapters on the full training process](handbook/00-end-to-end.md), combining pinned source code with current primary reports, and following real decisions in [the ongoing Marin 535B training run](handbook/04-marin-535b-live-case-study.md). Source code, author reports, engineering synthesis, and unverified items are labeled separately.

<a id="目录"></a>

## Directory layout

The 16 core modules also provide 32 exercise designs; this repository has not executed all of them, and publishing a curriculum does not count as completing it.

```text
frontier-llm-lab/
├── ROADMAP.md                   Learning sequence, prerequisites, paths, and completion criteria
├── RESOURCE_ATLAS.md            Core and supplementary resources, purposes, and openness
├── COVERAGE.md                  Depth of knowledge coverage and remaining gaps
├── curriculum/                 16 core modules and 6 frontier topics
├── lessons/                    Instructional units with worked examples
├── website/                    Interactive learning website, content generation, and automated checks
├── .github/workflows/          GitHub Pages build and deployment
├── LEARNING_LIST.md              Study index
├── KNOWLEDGE_TREE.md             Conceptual knowledge tree
├── REPO_RELATIONSHIPS.md         Evidence-backed project relationships
├── PROGRESS.md                   Stage status and where to resume
├── HOW_TO_STUDY.md               Ongoing study workflow
├── GLOSSARY.md                   Terminology and interface semantics
├── repos.json                   Registry of 21 source repositories
├── sources.lock.json            Source snapshots recorded for this study
├── SOURCE_INDEX.md              Clickable source index
├── handbook/                    Full process, three detailed mechanism chapters, and Marin 535B case study
├── sources/                     Independent upstream clones, ignored by the outer Git repository
├── notes/repositories/          Per-repository source notes
├── notes/connections/           Cross-repository topics and evidence
├── notes/sessions/              Study session records
├── experiments/                 Our experiments, READMEs, and small results
├── templates/                   Reusable note templates
└── tools/                       Cloning, snapshot, and integrity-check tools
```

The sources use depth=1 shallow clones. Their current working trees are readable; full Git history, submodule contents, large LFS files, model weights, and training data have not been fetched. The main GitHub repository stores learning material, experiments, and the source version manifest. The 21 independent upstream checkouts under `sources/` are not bundled and uploaded again; restore them with the commands below. Local source links in notes require `sources/` to be restored first; GitHub links pinned to a SHA can be read directly online.

<a id="在另一台电脑恢复"></a>

## Restore on another computer

```powershell
git clone https://github.com/yubol-bobo/frontier_llm_handbook.git
cd frontier_llm_handbook
python tools/clone_repos.py --restore-lock
python tools/validate_learning_repo.py
```

Restoration covers the registered source snapshots. It does not automatically install training environments, download weights/data, or run upstream installation scripts.

<a id="常用命令"></a>

## Common commands

Run these in this directory; the tools use only the Python standard library and Git:

```powershell
python tools/clone_repos.py --restore-lock
python tools/validate_learning_repo.py
python experiments/001-harbor-reward-contract/run.py
```

By default, `clone_repos.py` fills in missing repositories and checks existing origins without updating or resetting existing sources. To restore recorded versions on a new computer, use `python tools/clone_repos.py --restore-lock`. After updating sources, existing notes continue to cite their original SHAs; create a record for the new study session, then run `snapshot_sources.py` as needed.

<a id="下一次学习"></a>

## Next study session

New readers should start with the [ROADMAP assessment](ROADMAP.md) and [M00](curriculum/01-foundations-to-pretraining.md#m00), and copy the [personal learning record template](templates/learner-progress.md). Use the assessments to skip material you already know; do not treat the maintainer's reading progress as your own completion record. The next content priority is a complete M01 decoder lesson; see [maintenance progress](PROGRESS.md#下一次从这里继续).

<a id="claude-code-harness-学习入口"></a>

## Claude Code harness study entry

Start with the [M12 advanced guide](handbook/05-claude-code-harness.md): Pi → the Claude Code execution loop, context, permissions, and recovery → trajectories and evaluation in Harbor/APEX. Use the [pinned source and historical snapshot notes](notes/repositories/claude-agent-sdk.md) and [CPU state-machine experiment](experiments/harness-state-machine/README.md). The official SDK is the 20th independent source project; historical mirrors have separate provenance limits. The experiment uses an original mock provider, does not call Claude, and does not reproduce its internal implementation.

<a id="sol-pi学习可验证的-harness-提效"></a>

## SoL-Pi: study verifiable harness efficiency

The new [SoL-Pi guide](handbook/06-sol-pi-efficient-harnesses.md) and [pinned source note](notes/repositories/sol-pi.md) follow Pi → four efficiency mechanisms → M09 cache costs → M14 paired evaluation. The [CPU experiment and upstream validation record](experiments/004-sol-pi-contracts/README.md) distinguish actual cost-function tests, Windows platform limits, and unexecuted model evaluations.
