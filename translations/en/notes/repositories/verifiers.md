<a id="verifiers任务harnessruntime-与评分怎样组成同一条-rollout"></a>

# Verifiers: How do tasks, harnesses, runtimes, and grading form one rollout?

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/PrimeIntellect-ai/verifiers @ `27bbd216df0af719a43705866b2cf6139bcc95de`  
Verification scope: Read the implementation, related configurations, and declarations in the pinned local snapshot; did not install upstream dependencies or start a model, sandbox, or training run. The control-flow assessments in this note come from static reading and do not represent a reproduction of the authors' performance or benchmarks.

<a id="核心问题"></a>

## Core question

How can an agent benchmark both evaluate different harnesses and provide the traces needed for training? The current pinned snapshot must be read starting from the **v1 Task / Harness / Runtime / Rollout** interfaces; describing current source solely through class names in older environment tutorials misses actual control boundaries.

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

- [Rollout open / step / close lifecycle](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/rollout.py#L178-L540); local: [`verifiers/v1/rollout.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/rollout.py).
- [Task grading hooks, runtime dependencies, and reward recording](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/task.py#L198-L269); local: [`verifiers/v1/task.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/task.py).
- [Actual optional Pi harness adapter](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/harnesses/pi/harness.py#L19-L192); local: [`verifiers/v1/harnesses/pi/harness.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/harnesses/pi/harness.py).
- [Optional integration with Harbor CLI and task models](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/tasksets/harbor/taskset.py#L402-L535); local: [`verifiers/v1/tasksets/harbor/taskset.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/tasksets/harbor/taskset.py).

`Rollout.open()` creates or borrows a Runtime → executes Task.setup → Harness.setup → starts interception and tool servers → passes reachable endpoint/secret/MCP addresses to the Harness session. Task owns task conditions and grading rules, Harness owns how the agent executes, and Runtime owns program-execution resources. Composability does not mean every combination is compatible.

`Rollout.step()` executes one harness segment, without guaranteeing exactly one LLM request. A user message takes the continuation path; timeout, model/tool errors, and whether a new turn exists all affect whether execution can continue. `close()` shuts down interaction services, performs finalize/artifact collection if opening succeeded and the rollout has not failed, concurrently calls task/harness score, and finally cleans up and closes the Runtime according to ownership.

`Task.score()` finds metric / reward hooks and plugged judges, injecting data by signature. Without a runtime, it skips grading items with a required runtime parameter; it records metrics and weighted rewards separately, rather than automatically treating every returned number as one training reward.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

- **Implementation fact:** A borrowed runtime is not closed by Rollout; lifecycle ownership is an interface contract. If the resource owner tears it down early, open raises an error. This directly affects correctness when reusing sandboxes concurrently.
- **Implementation fact:** Agent timeout is recorded as failure; a normal stop can grade a partial trajectory. close is idempotent, with final cleanup in finally, and transport-close failure does not necessarily erase results that were otherwise gradable. These distinctions change the definition of valid training samples.
- **Implementation fact:** Offline rescoring cannot conjure evidence that depends on runtime; `Task.score` skips such signals. Repeatable evaluation requires saving sufficient artifacts during finalize or reconstructing a verifiable environment.
- **Implementation fact:** The Pi adapter pins npm package `@earendil-works/pi-coding-agent` to 0.84.1 and connects through pi-acp, provider configuration, and an interception endpoint. This reading only examined installation strings; none of their commands was run.
- **Implementation fact:** The Harbor adapter downloads via CLI, parses through Harbor's Python task model, and defers imports until a Harbor task is actually loaded. Also read lines 1–93 of the file: Dockerfile-only environments are not built directly by default, and timeout/isolation options change evaluation conditions.
- **Reading assessment:** Reusable environments depend on explicit lifecycle, evidence, and grading contracts. A unified entry point does not automatically produce a fair benchmark or guarantee rewards without shortcuts.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Object | Relationship type | Evidence and boundaries in this reading |
|---|---|---|
| Pi | Actual integration of an optional harness | `harnesses/pi/harness.py:38–41,47–48,84–192`; its pinned npm release is not the independent local Pi HEAD. |
| Harbor | Optional taskset dependency | `tasksets/harbor/taskset.py:402–410,434–441,522–528`; there is code evidence for both CLI and Python API use. |
| Prime RL | Directly depended upon by Prime RL | Established by Prime RL's pyproject, submodule, and `import verifiers.v1`; see the Prime RL note. Verifiers' own lifecycle layer is not a GPU optimizer. |
| APEX recipe | Conceptual correspondence, with separate integrations through Harbor | Both need task execution, grading, and token traces; APEX is not treated as a direct dependency of Verifiers. |

<a id="动手实验"></a>

## Hands-on experiments

**Pending: online/offline grading differences for the same trace.** Create one reward that checks only a string in the trace and another that must read a runtime file. Fix the task, input, and harness output, and compare `score` with a live runtime against offline `score` after saving only the trace. Observe which rewards/metrics are recorded, which are skipped, and whether missing evidence is explicit. Expected: runtime-only items do not execute offline; missing items cannot default to passing. Actual result: **not executed**.

Phase one uses a synthetic trace and lightweight runtime; a CPU suffices. Phase two switches between Pi and another harness on the same small Harbor task, fixing model, budget, task image, and verifier, then compares traces and artifacts. Phase two involves a model API, Node, and sandbox, none of which has been installed/called.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Trace interception/session into token and request-rewrite records in `Trace`.
- [ ] Verify how the Task/Taskset layer combines multiple rewards to avoid treating logged metrics as the training objective.
- [ ] Read the Harbor adapter's separate-verifier execution path and compare grading-isolation overhead.
