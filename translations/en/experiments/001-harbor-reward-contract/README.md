<a id="实验-001奖励向量并不自动等于可靠训练信号"></a>

# Experiment 001: a reward vector is not automatically a reliable training signal

Date: 2026-09-08. Status: executed, limited local CPU experiment.

<a id="问题与方法"></a>

## Question and method

Uses Harbor Cookbook's multi-reward task, pinned to source `e093c9a860b988d9d74901010ddddb9c7f124f92`. After reading the reference solution and all Python correctness and performance assertions, the script extracts the Python function from the heredoc in `solve.sh` and executes those assertions directly. **No shell scripts were executed, packages installed, models called, or containers started.**

The control is the upstream reference; the experimental variant only changes the case-insensitive email key to a case-sensitive one. All other code and test inputs remain unchanged. Upstream code is read only from the cloned snapshot and is not copied into the learning repository; its original license and attribution remain in the source repository.

<a id="重跑"></a>

## Rerun

Run from the learning repository root:

```powershell
python experiments/001-harbor-reward-contract/run.py
```

The script requires the source SHA to match the initial reading and uses `git show SHA:path` to read the three files actually executed from pinned Git objects; uncommitted working-tree changes will not be misattributed to that version. The script rejects `python -O` / `PYTHONOPTIMIZE` to prevent false passes when assertions are removed. After a source update, review it before establishing a new experiment version. Results are written to [results.json](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/001-harbor-reward-contract/results.json), including input hashes, execution time, and each assertion's result and duration.

<a id="观测"></a>

## Observations

| Candidate | Correctness tests | Performance tests | Reward vector |
|---|---|---|---|
| Reference solution | 7/7 pass | 1/1 pass | correctness=1, performance=1 |
| Solution with a case-sensitivity error | 5/7 pass | 1/1 pass | correctness=0, performance=1 |

The performance assertion processes 100,000 records, requires 50,000 output records, and must finish in under 2 seconds; all input email addresses use the same case pattern. This test can therefore validate speed and counts for that workload, but cannot independently validate email case semantics. Timing is a result from this machine, not a cross-machine performance conclusion.

The experiment also confirms that the raw reward dictionary has no `reward` field; directly calling `.get("reward", 0.0)` returns zero. The script's `correctness * (0.5 + 0.5 * performance)` calculation merely illustrates explicit scalarization; it is neither an upstream formula nor a recommended general objective.

<a id="结论的边界"></a>

## Boundaries of the conclusions

- Validated: results of the pinned snapshot's Python correctness/performance assertions for these two candidates.
- Not validated: pytest plugins, container isolation, Harbor's complete reward.json loading process, training-adapter normalization, or RL learning effects.
- This experiment does not establish a specific upstream integration bug. When actually connecting a task to a trainer, confirm the reward fields and aggregation policy the consumer really receives.

<a id="与知识树的连接"></a>

## Connection to the knowledge tree

Task specification → verifier coverage → reward vector → scalarization → RL objective → held-out evaluation. The actual achievement of the first study round is validating these interfaces separately, rather than calling a successfully executed script successful training.
