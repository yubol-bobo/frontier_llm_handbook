<a id="harbor-cookbook任务评分与-rl-接口如何连接"></a>

# Harbor Cookbook: Connecting tasks, grading, and RL interfaces

Date: 2026-09-08  
Stage: L2 focused mechanism reading; completed an L3 local CPU experiment on the Python grading portion of multi-reward.  
Source: [harbor-framework/harbor-cookbook](https://github.com/harbor-framework/harbor-cookbook) @ `e093c9a860b988d9d74901010ddddb9c7f124f92`  
Verification scope: Read task and training-adapter examples; directly executed upstream Python assertions. Did not run Harbor, Docker, the pytest CLI, Tinker, or RL.

<a id="核心问题"></a>

## Core question

How does a task become a training signal? Do the scores emitted by multiple verifiers inherently constitute the single reward a trainer needs?

<a id="源码阅读链路"></a>

## Source-reading chain

| Entry point | Observation | Pinned version |
|---|---|---|
| [multi-reward/tests/test.sh](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/recipes/multi-reward/tests/test.sh) | correctness / performance independently produce binary rewards and output reward.json | [L10–27](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/recipes/multi-reward/tests/test.sh#L10-L27) |
| [test_correctness.py](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/recipes/multi-reward/tests/test_correctness.py) | Assertions for case-insensitive email handling, retaining the largest id, sorting, and more | [L8–69](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/recipes/multi-reward/tests/test_correctness.py#L8-L69) |
| [multi-step/task.toml](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/recipes/multi-step/task.toml) | Per-step instructions, prerequisite health checks, reward thresholds, and artifact saving | [L31–98](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/recipes/multi-step/task.toml#L31-L98) |
| [harbor_rl/train.py](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/harbor_rl/train.py) | Sandbox → RLEnvironment → tool / grade → Tinker Env | [L42–130](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/harbor_rl/train.py#L42-L130) |

`make_envs()` creates a separate sandbox and session for each sample in a group, starts BashTool, and uses a renderer to generate initial messages. The tool wrapper calls `env.step()`, and the reward wrapper calls `env.grade()`; cleanup stops the environments created when execution ends. `group_size` affects both training sampling and the direct cost of concurrent environments.

<a id="真实接口边界"></a>

## Actual interface boundaries

- `sky_rl/README.md` explicitly states that this directory is a pointer and the full integration lives in the external SkyRL repository. [Pinned entry point](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/sky_rl/README.md#L1-L9)
- `prime_rl/README.md` points to Verifiers' `environments/opencode_harbor`; it cannot be described as a complete Prime trainer bundled with Cookbook. [Pinned entry point](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/prime_rl/README.md#L1-L5)
- The script dependency declaration in `harbor_rl/train.py` uses a specific Harbor feature branch and external `tinker-cookbook`. This example does not establish verified compatibility with the independently cloned Harbor HEAD in this study. [L1–7](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/harbor_rl/train.py#L1-L7)
- The current reward wrapper reads `result.rewards.get("reward", 0.0)`. Passing a raw dictionary containing only correctness/performance directly to this read would yield 0. Whether the actual environment normalizes the rewards, and how they should be aggregated, must be explicitly verified at the adapter layer; this alone does not establish a bug in a running Harbor system.

<a id="已运行一个错误实现也能获得性能分"></a>

## Executed: an incorrect implementation can still earn the performance reward

See the [experiment description](../../experiments/001-harbor-reward-contract/README.md), [rerunnable script](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/001-harbor-reward-contract/run.py), and [raw results](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/001-harbor-reward-contract/results.json).

| Candidate | correctness | performance |
|---|---:|---:|
| upstream reference solution | 1 | 1 |
| Incorrect version with only email `.lower()` removed | 0 | 1 |

The experiment actually loaded the pinned reference Python function and test assertions; it did not start the network installation procedure in upstream `test.sh`. The result shows that this performance test does not cover case-handling correctness. If training optimizes only the performance field, incorrect behavior can also be rewarded. Scalarizing a reward vector requires explicit product objectives, such as satisfying correctness first and then rewarding performance. Specific weights are a separate design decision that requires evaluation.

<a id="与其他项目的连接"></a>

## Connections to other projects

- **Direct use: Harbor.** Harbor provides Cookbook's task format and RL environment interfaces.
- **Documentation-linked integrations: SkyRL, Verifiers / Prime RL.** Continue into the target repositories' source; a link alone is not evidence of end-to-end reproduction.
- **Conceptual correspondence: APEX recipe.** Both pass task artifacts from a sandbox to a verifier and then associate the result with a rollout; APEX has separate data and version constraints.
- **Conceptual correspondence: Open Instruct / Verifiers.** Reward output does not guarantee a reliable optimization signal. Formats, missing values, failure types, and final metrics still need examination.

<a id="下一步"></a>

## Next steps

- [ ] Run this task in a real Harbor Trial to verify JSON reward reading and normalization, beyond executing Python assertions.
- [ ] After fixing scalarization, check the training-reward ordering of the incorrect candidate and the reference.
- [ ] Write two independent verifiers for a realistic task and retain held-out cases that are not used in training.
