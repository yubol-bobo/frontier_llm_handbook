<a id="持续学习流程"></a>

# Ongoing study workflow

The default learning sequence is defined in [ROADMAP.md](ROADMAP.md). New readers should complete the M00 assessment and copy the [personal progress template](templates/learner-progress.md); [PROGRESS.md](PROGRESS.md) records the repository maintainer's actual work, not readers' course grades. Original project numbers are only index entries.

<a id="每次学习的最小循环"></a>

## The minimum loop for each study session

1. Read the current course module and your own progress; maintainers also read [PROGRESS.md](PROGRESS.md). Choose one main question, such as “Why do masked targets not enter the global token denominator?”
2. Confirm that the repository SHA matches the session's notes; if the source has changed, preserve the old evidence and create a new record.
3. Read the caller, implementation, configuration, and one relevant test. Record input/output shapes, object ownership, and where side effects occur.
4. Separate facts, author claims, inferences, and unverified hypotheses. Cite pinned commits and line numbers, avoiding links only to a changing main branch.
5. Write a minimal falsifiable prediction before deciding whether an experiment is needed. Record controlled variables, metrics, actual commands, and results; complete the parts that do not need a GPU locally first.
6. Update the project note and one node in the knowledge tree; add one evidence-backed connection. Raise a learning level only for a path that was actually traced.
7. Record the session's artifacts and the next concrete action in a session note, then update progress.

Accumulate two types of evidence for every module: support knowledge understanding with mechanism explanations, derivations, or design diagrams; support engineering skills with inspectable implementations, configurations, run records, or failure analysis. Label unexecuted designs “to be validated,” and record the specific change from needing help to completing something independently in this session.

<a id="笔记标准"></a>

## Note standards

Use the [project note template](templates/repository-note.md) for source notes and the [decision template](templates/research-decision.md) for research choices. A useful note should answer:

- What does this system layer consume and produce?
- Who owns the state? Who can change it? What is persisted?
- How do normal, cancellation/timeout/failure paths differ?
- Which quantity must remain invariant, and which parameter trades one cost for another?
- How could an experiment reveal that your explanation is wrong?

“This project supports GRPO / MoE / async” is only an entry point and cannot replace these explanations.

<a id="关系类型"></a>

## Relationship types

| Type | Determining evidence | What must not be inferred |
|---|---|---|
| Direct dependency | Import, manifest, submodule/gitlink | Any two HEADs are compatible |
| Optional backend | Guarded import, backend branch, optional extra | The default path necessarily uses this backend |
| Example integration | Actual calls in an example; or explicitly labeled as only a pointer | The entire project is connected end to end by default |
| Project lineage | Official fork/derivation statement | Current APIs are identical |
| Conceptual correspondence | Independent implementations of the same problem, with side-by-side source evidence | The two software projects depend on each other |

<a id="实验规则"></a>

## Experiment rules

Place experiments in `experiments/NNN-topic/`, with at least a `README.md`, a run script, and small results. Record model/data/source versions, the environment, and sources of randomness. Do not commit model weights, datasets, or large checkpoints to the learning repository.

This round uses only local CPU experiments without paid APIs. For future experiments requiring containers, GPUs, cloud services, or model APIs, first document specific commands, resource needs, and expected artifacts, then run them when the corresponding environment is available. Citing a recipe does not mean its training data and dependencies are ready.

<a id="源码管理"></a>

## Source management

- The main repository is [yubol-bobo/frontier_llm_handbook](https://github.com/yubol-bobo/frontier_llm_handbook), with `origin/main` as the default synchronization target; further learning material continues to accumulate here.
- Each upstream manages its own Git repository under `sources/`; the outer Git repository ignores that directory.
- By default, clones have shallow history, submodules are not initialized, and LFS contents are not downloaded. When studying an older version as needed, explicitly fetch that version without resetting other existing changes.
- `sources.lock.json` is a study provenance record, not a dependency lock that makes all 20 projects runnable together.
- Reading records use GitHub commit permalinks; local links provide convenient navigation.
- An independent source HEAD may differ from a project's pinned dependency commit. Actual execution follows that project's dependency pins.

<a id="续学提示词"></a>

## Prompt for resuming study

You can continue in this learning repository with:

> Read PROGRESS.md and continue the current study unit. First verify the source SHA, read code around one specific implementation question, and record evidence; choose a small experiment suitable for the current environment; update project notes, the knowledge tree, relationships, and progress. Do not describe unexecuted training as reproduced.

This repository stores study state; it does not itself configure scheduled runs, background training, or automatic upstream updates.
