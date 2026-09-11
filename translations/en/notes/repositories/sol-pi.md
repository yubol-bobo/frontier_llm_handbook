<a id="sol-pi沿四种机制学习-agent-harness-提效"></a>

# SoL-Pi: learning agent harness efficiency through four mechanisms

First source study: 2026-09-11. Snapshot: `d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0`, version `0.1.0`, MIT, 64 tracked files. This study reaches targeted L2 reading of four mechanisms with partial execution evidence; it does not reproduce the entire project or paper results. [Pinned source](https://github.com/NVlabs/SoL-Pi/tree/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0).

<a id="阅读定位与配置边界"></a>

## Reading position and configuration boundaries

This standalone package uses Pi's public extension APIs and fits after the [Pi event-loop study](pi.md). It treats tool actions, tool results, and context restructuring as system components that can be changed. The package does not train model weights or publish the complete automated proposal, selection, and research-search pipeline. Follow the [learning guide](../../handbook/06-sol-pi-efficient-harnesses.md) for the sequence.

Project configuration takes precedence only when the current directory is allowed to be trusted. The selected project file replaces the user configuration file and is combined with built-in defaults; the two files are not merged field by field. All four switches default to off, and configuration is read when extensions are registered at session start. Development locks Pi 0.84.2, whereas this handbook's independent Pi snapshot is 0.85.1. That pair is not an already-verified dependency combination. [Configuration selection](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L48-L75), [defaults](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L27-L36), [returned configuration](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L115-L121).

| Mechanism | Interface changed | Overhead it aims to reduce | Behavior to check |
| --- | --- | --- | --- |
| Action Fusion | `edit/write` + `then_run` | Model round trips between mutation and validation | File state and ordering after a successful edit and failed check |
| ObservationPack | Context projected to the model | Repeated transmission of large outputs | Archiving, paginated recall, and fallback |
| Evidence-Preserving Reducer | Diagnostic tool results | Repetitive, verbose logs | Quote provenance, omitted evidence, and source readback |
| Online Context Compact | Plan boundaries and session lifecycle | Combined long-context and cache-rebuilding costs | Economic gates, summary quality, cancellation, and continuation |

<a id="本次实际验证"></a>

## Actual verification in this study

[Experiment 004](../../experiments/004-sol-pi-contracts/README.md) directly imports the pinned `economics.ts`: all eight deterministic scenarios and independent arithmetic assertions passed. A separate isolated checkout ran upstream checks: TypeScript passed; Vitest reported **134/139 passed, five failed, zero skipped**. Failures involved Windows npm subprocess resolution, symlink fixtures, and a POSIX file-mode assertion. Separately executed packaging checks and the Pi API probe passed. These results do not make the full upstream check successful; commands, failure locations, and scope are recorded with the experiment.

The sections below distinguish assertions inspected in source, executed fixtures, and proposed exercises. No live model was called; quality, billing, cache hits, and task latency were not measured.

<a id="action-fusion-与-observationpack顺序执行与可回取观察"></a>

## Action Fusion and ObservationPack: ordered execution and recallable observations

This reading is pinned to SoL-Pi commit `d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0`. Both mechanisms are disabled by default. The entrypoint reads configuration at `session_start` and registers Action Fusion, ObservationPack, EPR, then Online Context Compact. Development dependencies pin Pi 0.84.2; runtime packages remain peer dependencies. This establishes an integration baseline, not compatibility with every version. [Defaults L27–36](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L27-L36), [registration L13–36](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/index.ts#L13-L36), [dependencies L31–47](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/package.json#L31-L47).

<a id="action-fusion沿三个入口追踪"></a>

### Action Fusion: trace three entrypoints

1. [`createActionFusionExtension` L69–132](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/index.ts#L69-L132) composes Pi's public edit/write definitions, retaining their interfaces and adding optional `then_run`. Execution removes that argument, delegates the mutation to the built-in tool for `ctx.cwd`, and uses a shared helper for the follow-up command. The model chooses the command before the mutation; there is no intervening model decision.
2. [`executeMutationThenRun` L95–125](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/then-run.ts#L95-L125) mutates first and returns the original result when no command was requested. Otherwise it checks the file, invokes Pi's bash tool, and appends a `succeeded` marker and output. A mutation exception makes the command `skipped`; a command exception reports `failed` while retaining the written file. This provides no transactional rollback.
3. [`resolveToolPath` and `withFusedFileQueue` L17–74](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/file-queue.ts#L17-L74) normalize relative paths, `@`, `~`, and file URLs, then derive a queue key from a real path or an existing ancestor. The same-file queue covers both mutation and command; `finally` releases it. External writers are outside this lock. Two SHA-256 reads separated by an event-loop yield detect only changes visible across that interval; they do not establish exclusive access throughout the command. [Check interval L50–67](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/then-run.ts#L50-L67).

The command string, cancellation signal, and optional timeout in seconds are delegated to Pi's bash tool; this layer sets no default timeout. Distinguish failures by stage: an invalid encoded file URL can throw before mutation, before reaching the branch that generates a `skipped` marker. The test counts calls to establish that neither mutation nor command occurs on that path. [Arguments L16–30](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/then-run.ts#L16-L30), [invalid-path test L141–152](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/action-fusion-paths.test.ts#L141-L152).

<a id="observationpack沿三个入口追踪"></a>

### ObservationPack: trace three entrypoints

1. The [`context` callback L137–209](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/index.ts#L137-L209) copies the message array and changes only the provider projection. It archives before counting sends: the first two remain full, and later projections use a stable placeholder. Counts are scoped by session root and observation ID. After a restart, absent in-memory counts fall back to the number of subsequent assistant messages. Storage or ledger failures retain the original message, but session-root resolution occurs outside this catch block.
2. [`createObservation` and `ensureStored` L67–156](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/observation.ts#L67-L156) accept only successful, nonempty, entirely textual results whose joined text is strictly larger than 10 KiB. Errors, mixed content, and results containing an EPR receipt line pass through. IDs combine tool name, call ID, and content hash. Text blocks are joined with newlines and stored as UTF-8; existing objects must match size and hash. A placeholder retains metadata and complete head/tail lines within 512 bytes each, omitting the middle. The excerpt is lossy; exact retrieval depends on the separate archive. [Thresholds L12–17](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/observation.ts#L12-L17), [placeholder L178–196](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/observation.ts#L178-L196).
3. [`obs_recall` L41–103](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/index.ts#L41-L103) validates the ID, reads from a byte offset, and returns `next_offset` and `eof`. Output including headers is capped at 16 KiB and 400 lines. The reader trims page ends to avoid splitting UTF-8 characters. Start at zero and follow returned offsets to reconstruct the archived text. An arbitrary offset can start inside a character, and recall itself does not recheck the content hash. [Paging L207–250](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/observation.ts#L207-L250).

Archives live at `sol-pi/<sessionId>/observation-pack/objects/` under the current session directory; the provider does not need to retain the complete text. Projection replacement preserves the message's other fields, and stored session history is not edited in place. The object directory and object file have symlink checks; a corrupt existing object prevents packing. Recall reports errors for unknown IDs, missing files, and offsets beyond the end of the file. These failures do not fabricate a complete result. [Session directory L9–16](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/runtime-paths.ts#L9-L16).

<a id="练习与证据边界"></a>

### Exercises and evidence limits

- In a temporary copy, make the follow-up command fail. Predict the error marker, resulting file content, and order of the next same-file operation, then compare with the [failure and queue tests L249–342](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/action-fusion.test.ts#L249-L342).
- Construct a Chinese log spanning several pages and predict four projections. Remove each recall page's headers, concatenate payloads in order, and compare UTF-8 bytes. See the [projection test L110–128](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/observation-pack.test.ts#L110-L128) and [paging test L261–285](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/observation-pack.test.ts#L261-L285).
- Add one sample exactly at the threshold and another one byte larger. Explain why only the latter qualifies for packing. Restart the extension and vary subsequent assistant messages to inspect the fallback count, distinguishing the code's proxy for context projections from actual provider-billed requests.

These findings explain behavior from source; see Experiment 004 for actual execution results. The exercise variants remain unexecuted, and no performance result is established. In particular, the [fused-write test L303–325](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/observation-pack.test.ts#L303-L325) uses less than 10 KiB and therefore establishes pass-through only. Enlarge its output and move the marker outside the head/tail excerpt to check whether the placeholder still contains it.

<a id="可验证摘录与有条件的上下文压缩"></a>

## Verifiable excerpts and conditional context compaction

This section reads implementation and test source at fixed commit `d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0`. Test references describe assertions inspected in source; see Experiment 004 for actual execution scope and five failures. The authors' benchmark results were not reproduced.

<a id="1-reducer先存原文再核对摘录"></a>

### 1. Reducer: archive the text, then verify excerpts

Evidence-Preserving Reducer handles long diagnostic logs at `tool_result`, including `bash` and the `then_run` output of fused `edit/write` calls. Defaults are a 4,096-byte minimum and a 600,000-JavaScript-character maximum. Nonmatching commands, short/oversized output, and text matching the likely-secret regex are skipped. A full-output file is read only when it is a regular, nonsymlink `pi-bash-*.log` file in the system temporary directory; otherwise the inline text is used. Thus the archived source is the text actually obtained, which need not always be the complete process output. [Candidate extraction](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/candidate.ts#L34-L98) [Eligibility](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/index.ts#L64-L77) [Defaults](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/config.ts#L14-L29)

Eligible text is first stored by SHA-256 as `objects/<first two digits>/<hash>.txt` under session storage, then sent to the configured reducer model. The receipt contains the source path, source hash, byte count, quote locations, and quote hashes. The main model retains diagnosis, repair, reruns, and adjudication, and can read an explicit line or byte range when it needs exact context. Fused calls replace only the command-output portion, preserving mutation confirmation and the original `isError`. [Archive](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/archive.ts#L24-L52) [Receipt format](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/receipt.ts#L146-L176) [Fused projection](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/candidate.ts#L34-L98)

<a id="2-证据保留验证了什么"></a>

### 2. What “evidence preserving” verifies

The validator checks JSON, schema, source hash, a status matching the tool's `isError`, a Boolean `uncertain`, a maximum of 12 evidence items, and a maximum of 600 JavaScript characters per quote. Each quote must be an exact contiguous substring of the source. It deduplicates evidence and locates the first matching line. If a failed log matches the failure-keyword regex, at least one evidence item must be tagged `fatal` or `failure`. [Validator](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/receipt.ts#L82-L143) [Limits and regex](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/config.ts#L14-L29)

**These checks establish provenance, not semantic completeness.** The validator does not establish that a `kind` label is accurate, that other failures were retained, or that the real cause was selected. The failure guard does not check whether that particular quote contains a failure keyword. `uncertain=true` can still be accepted. “Lossless” is an instruction in the prompt, not a proof that the summary loses no information. For example, a log with two failures can yield a passing receipt that accurately quotes only one. This example is a code-derived inference, not an executed test. [Validator](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/receipt.ts#L82-L143)

When the model is unavailable, a call throws, a response reports an error, the receipt is invalid, or the receipt is not smaller than the source, the extension returns `undefined`, preserving the original tool result. It also skips when there is no persistent session directory. The parent cancellation signal is relayed to the model call, with a separate 90-second timeout. **Fail-open has limits:** `archiveBody()` runs before the model-call `try/catch`. An archive write error or mismatched existing object throws; this extension does not convert every storage failure into a fallback. [Fallback flow](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/index.ts#L77-L195) [Signal propagation](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/provider.ts#L74-L92) [Archive integrity](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/archive.ts#L24-L52)

<a id="3-compact在完成步骤的边界停下再续跑"></a>

### 3. Compact: stop at a completed-step boundary, then continue

Online Context Compact starts from an `update_plan` call that newly marks a step `completed`. At `turn_end`, it must find the corresponding successful tool result; the assistant must not have errored/aborted, and the context signal must not be aborted. An economic or window decision is followed by a check using Pi's `findCutPoint` that there is actually history to compact. It then calls `abort()`, waits until the whole agent is idle at `agent_settled`, and invokes native `compact()`. Window pressure therefore does not bypass this boundary sequence. [Boundary and selection](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L202-L311) [Native feasibility](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L99-L148)

After success, one hidden message with `triggerTurn: true` requests a fresh plan for remaining work. A settlement barrier waits for this continuation too, preventing print/JSON mode from returning prematurely. Cancellation or `AbortError` does not trigger a continuation; other compaction errors are thrown, and failure to start the continuation also throws. Tree navigation is blocked during compaction. Session start/tree changes restore the current branch's latest valid state and clear pending actions. `steer` or `CORRECTION:` clears stale plans, samples, and debt. These are recovery and cancellation safeguards, not a guarantee of automatic continuation after any crash. [Continuation and errors](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L314-L440) [Restore and correction](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L178-L258) [Persisted state](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/state.ts#L133-L207)

<a id="4-经济公式缓存成本的启发式"></a>

### 4. Economics: a heuristic for cache costs

Let `W=writeTokens` (the pre-compaction context estimate), `A=archiveTokens`, `M=memoTokens`, `r=cacheWriteReadRatio`, and `D=carriedDebtTokens`:

```text
S = A − M
c = max(0, r − 1)
B = Wc / S                 # This compaction's breakeven requests; requires S > 0
Bcombined = (D + Wc) / S   # Including carried debt
```

At runtime, `A=max(0,W−systemPromptEstimate−keepRecentTokens)`. The retained tail defaults to 20,000 tokens and the memo estimate to 1,000. W is the larger of the visible-message-plus-system-prompt estimate and a valid provider-reported count. The release entry gets r from configuration, defaulting to 12.5; switching models during the session does not recalculate it. Calling the factory directly without r produces null, leaving only window protection available. [Runtime estimates](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L202-L311) [Context estimate](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L178-L258) [Constants](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L35-L70) [Configured ratio](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L14-L35)

**Assumption inferred from the formula:** `r−1` compares incremental cache rebuilding with an otherwise readable warm cache, accounting in cache-read-token equivalents. The code does not observe actual hit rates or include the summarization model's input/output charges and added latency in B. It is a scheduling heuristic, not a complete bill. Cold caches, changing hit rates, or inaccurate plan forecasts can separate the estimate from realized savings. [Cost formula](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/economics.ts#L14-L235) [Configuration explanation](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/docs/configuration.md#L58-L65)

Let μ be the mean number of requests between completed boundaries and R the remaining boundary count. By default `k=0`, so μ is used directly. With nonzero k, fewer than three samples halve the mean; otherwise the lower estimate is `max(0,μ−k×sample standard deviation)`. Calling that adjusted mean L:

```text
Hraw = 1 + floor(L × max(0,R) × scale)       # Default scale=1
U = max(0, floor((window − context) / average positive growth))
H = min(Hraw,U)                            # Use Hraw when U is unavailable
Hfirst = min(2H,U)                         # No additional cap when U is unavailable
```

The first compaction requires `B≤Hfirst` and a positive forecast. Subsequent compactions require `B≤H`, `1.5B≤H`, and `Bcombined≤H`. At or above `window−16,384`, window protection can bypass the economic gates, but `S>0`, the boundary sequence, and native feasibility still apply. These multipliers are implementation constants, not proven global optima. After compaction, Wc is recorded as debt and S as repayment per request; each `before_provider_request` subtracts one repayment until zero. This is estimated accounting, not a record of payments. [Forecast and gates](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/economics.ts#L14-L235) [Repayment state](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/state.ts#L133-L207)

<a id="5-按真实公式手算"></a>

### 5. Work through the actual formula

Using the economic test's scale, set `W=80,000, A=60,000, M=1,000, r=12.5, D=0`. Completed-boundary counts are `[4,6,5]`, four boundaries remain, the window is 200,000, and average positive growth is 2,000 tokens per request. Then `S=59,000`, `Wc=920,000`, `B≈15.59`, `Hraw=21`, `U=60`, and `H=21`.

| Scenario | Result under the source rules |
| --- | --- |
| First compaction | `Hfirst=42`; 15.59≤42, so the economic gate passes. |
| Subsequent compaction with the same other inputs | `1.5B≈23.39>21`, producing `deferred_subsequent_margin`. |
| Subsequent compaction with six remaining boundaries and 2,000,000 carried debt | H=31 passes the margin, but `Bcombined≈49.49>31`, producing `deferred_carried_debt`. |

These are toy decision outcomes, not measured monetary or latency savings. Actual triggering also depends on the boundary, window, and native feasibility. [Fixture basis](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/online-context-compact-economics.test.ts#L12-L87) [Implemented formula](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/economics.ts#L14-L235)

<a id="6-测试证据的范围"></a>

### 6. What the tests establish

Reducer tests assert exact quotes, archive readback, preserved mutation confirmation in fused output, and fallbacks for invented quotes, model errors, and missing models/session directories. They do not prove semantic completeness for arbitrary logs. Economic tests cover horizon estimation, nonpositive savings, missing ratios, window protection, and carried debt. Real `AgentSession` tests check that the original `prompt()` waits for the final continuation reply after one/two compactions. They use a faux provider and a deterministic summary, however, so they verify lifecycle behavior rather than model-summary quality, real cache hits, or benchmark task performance. [Reducer tests](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/evidence-preserving-reducer.test.ts#L239-L542) [Economic tests](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/online-context-compact-economics.test.ts#L12-L87) [AgentSession tests](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/tests/online-context-compact-agent-session.test.ts#L47-L175)

<a id="与已有知识树的连接"></a>

## Connections to the existing knowledge tree

- **Pi → SoL-Pi is a direct implementation relationship**: public tool factories, extension events, session storage, and native compaction provide integration points. Read the underlying loop before tracing the four registration entrypoints, so extension behavior is not mistaken for default runtime behavior.
- **M09 inference → M12 harness**: prefix changes can affect prompt-cache reuse and cost. The cache write/read ratio here is a provider-pricing heuristic, related to but distinct from GPU KV-cache memory and scheduling in [SGLang](sglang.md). This is not an implemented integration.
- **M12 → M14 evaluation**: compare [DeepSeek Harness](deepseek-harness.md) request boundaries and [Harbor](harbor.md) tasks and grading to study success, evidence coverage, and total cost on the same tasks. SoL-Pi does not provide verified wiring to these projects.
- **M12 → M13 agent RL**: comparison with the [APEX recipe](apex-agents-skyrl-recipe.md) requires specifying how fused actions, projected results, and summaries change trajectories and reward-visible information. This is a future experiment, not an RL update implemented by SoL-Pi.
- **Claude Code / Agent SDK → SoL-Pi**: the former supports comparison of tool permissions, process transport, and event lifecycle; the latter exposes editable efficiency mechanisms. See the [Claude notes](claude-agent-sdk.md). This does not imply identical internals.

Next, test two currently unverified hypotheses: whether shortened logs still expose every task-relevant error, and whether the economic gate predicts lower total cost after actual cache misses and summarization charges are included. Establish quality adjudication and source-recall checks before scaling the task set.
