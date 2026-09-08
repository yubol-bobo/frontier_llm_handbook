<a id="harness-状态机实验"></a>

# Harness state machine lab


<a id="目的"></a>

## Purpose

This small experiment makes the boundary between a provider's proposal and a harness's authority visible. A deterministic mock provider proposes tools; the harness owns permissions, a finite tool-cycle budget, call IDs, and session events. It is an original teaching model, **not a description or reproduction of Claude, Claude Code, or any proprietary runtime**.

<a id="运行"></a>

## Run

Requires Python 3.10 or later. From this directory:

```sh
python run.py
```

The script runs eight self-contained `unittest` tests. It uses only the Python standard library and CPU. It needs no package installation, credentials, paid API, network access, subprocess, or real shell/file tool. Running Python itself is the only command you need.

<a id="状态与执行"></a>

## State and execution

```text
requested → permission_checked → denied
                              ↘ started → result

restore with started but no result → unknown (no automatic retry)
```

`Session` appends immutable `Event` records to an in-memory tuple. Old snapshots retain their original event tuple. `Snapshot` includes a schema version, session ID, canonical constraints, and events; restore rejects unsupported versions. Call IDs combine the caller-supplied session ID and a counter derived from recorded requests. They remain unique within one linear session across restore; callers must choose distinct session IDs for different sessions.

`dispatch()` checks the trusted, runtime-supplied tool allowlist before recording `started` or invoking a tool. Unknown tools are denied. Re-dispatching the same ID reuses an existing denial or result without invoking the tool. A later policy change does not turn an old denied call into a new call. The two mock tools are `double` and `increment`; the latter changes only `MockWorld.counter` in memory.

The loop first checks for actions that started without a recorded result. If any exist, it returns `unknown` before querying the provider. Otherwise, the loop stops when its provider finishes or its tool-cycle budget is consumed. Each new proposal uses one cycle, including denials. Reaching the cap returns `budget_exhausted`, even if the provider might have finished on a subsequent query. This budget bounds this one loop invocation; it does not meter tokens, money, wall time, or later invocations.

<a id="测试"></a>

## Tests

| Test | Behavior checked |
|---|---|
| Provider loop and permission order | Two proposals finish; permission precedes tool start and result. |
| Denial reuse | Repeating a denied ID remains denied after a policy change. |
| Result reuse | Restoring and repeating a completed ID does not repeat its effect; the next ID advances. |
| Crash window | An effect occurs before the result is journaled; re-dispatch and a resumed loop report `unknown` without rerunning it. |
| Finite budget | A repeating provider stops after three tool proposals. |
| Compaction diagnosis | A deliberately lossy summary drops a constraint and the checker identifies it. |
| Versioned snapshot | Later events do not change an older snapshot; unknown versions are rejected. |
| Invalid tool/call | An unsupported tool and an unknown call ID cause no effect. |

<a id="压缩与约束"></a>

## Compaction and constraints

The compaction test summarizes history as “Result was 6.” and intentionally drops “Do not increment the counter.” `compact_context()` returns both the summary and the missing canonical constraint. The original constraint remains in the session snapshot.

This is an **exact-string retention diagnostic**, not semantic validation: a paraphrase may produce a false alarm, and a contradictory summary that contains the exact words could pass. The test demonstrates detection only; it does not repair the summary or make natural-language constraints executable. The dispatch allowlist is a separate policy input supplied by the caller.

<a id="运行范围与局限"></a>

## Runtime scope and limits

“Restart” means reconstructing a `Session` object from its in-memory snapshot while retaining the same mock world. The crash is an injected exception after the counter changes and before the result event is appended. There is no disk journal, process restart, concurrent worker, real model, streaming, sandbox, or external service. Snapshots are trusted local data, not authenticated or comprehensively validated input. Tuple appends favor readability over efficiency.

This is **not an exactly-once guarantee**. An action marked `started` without a result could have happened, partly happened, or never happened; the safe report is `unknown`. A fresh call ID, independent session, stale snapshot, or fork can still repeat an effect. A real system needs durable records and tool-specific reconciliation or idempotency support. This lab illustrates why blindly retrying an ambiguous side effect is unsafe; it does not implement those production mechanisms.

<a id="本次执行记录"></a>

## Execution record

On 2026-09-08, all 8 tests passed locally with Python 3.11.4. The script below also generated the [results and event traces](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/harness-state-machine/results.json), which can be compared with the [recording script](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/harness-state-machine/record.py). Counter values are 0 after denial, 3 at budget exhaustion, and 7 in the unknown-outcome case; recovery did not add 7 again. The compaction diagnostic reports the omitted constraint text.

```sh
python record.py
```

This writes only the experiment's JSON result file; the simulated tools operate in memory. A real disk journal, CLI integration, and model evaluation remain unexecuted.
