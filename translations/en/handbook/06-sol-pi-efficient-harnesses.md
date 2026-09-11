<a id="sol-pi-学习指南在保持任务质量的前提下改进-harness-效率"></a>

# SoL-Pi study guide: improve harness efficiency while preserving task quality

Date: 2026-09-11. Source baseline: `NVlabs/SoL-Pi@d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0`. Use the [source notes](../notes/repositories/sol-pi.md) and [Experiment 004](../experiments/004-sol-pi-contracts/README.md), keeping reading evidence, inference, and execution records separate.

<a id="1-学习位置与公开边界"></a>

## 1. Learning position and public scope

Follow **M01 → Pi foundations in M12 → the four mechanisms → M09 cache economics → M14 paired evaluation**. Pure harness reading in M12 can follow M01 early; this route does not replace the mathematical prerequisites for post-training. The goal is to explain execution, design counterexamples, and audit efficiency gains. Completing the reading does not establish mastery of a complete automated research system.

The package publishes four independent extensions. The private automated proposal/search loop that produced these designs is not shipped in this package. All four mechanisms are disabled by default and use Pi's public APIs. The project title does not establish access to a search engine, trainer, or complete research workflow. [Release scope L21–45](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/README.md#L21-L45), [defaults L27–36](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/config.ts#L27-L36).

The authors describe screening 152 candidate directions down to four mechanisms. On the held-out 51-task EdgeBench suite, they report roughly 94% of Pi's average score with 45–49% fewer tokens. These are author-reported results with a quality tradeoff, not a reproduction by this handbook. [Official research report](https://nvlabs.github.io/SoL-Pi/).

<a id="2-从-m01-进入-m12先追一条-pi-轨迹"></a>

## 2. From M01 to M12: trace one Pi trajectory first

Read the [Pi notes](../notes/repositories/pi.md), then draw the order of the user task, model request, tool execution, feedback, and termination. Separate raw output, durable session history, and the messages visible to the provider on this request. Number requests and explain why tool success and task completion require different evidence. If cancellation, resume, and failure returns remain unclear, complete the [M12 foundation exercises](../curriculum/03-posttraining-and-agents.md#m12) first.

Next trace SoL-Pi's `session_start`: it reads configuration using the trusted context, initializes once, and registers Action Fusion, ObservationPack, EPR, and Online Context Compact in that order. Registration order and the events each mechanism handles jointly determine composition; the four switches are not independent black boxes. [Entrypoint L13–36](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/index.ts#L13-L36).

<a id="3-四项机制每项都找一条正常路径和反例"></a>

## 3. Four mechanisms: find a normal path and a counterexample for each

Read each row in order. Submit a small “input → state change → model-visible result” diagram for every mechanism, marking where its guarantee ends. Understand each mechanism before examining composition. Recallability does not ensure that the model recalls; authentic quotations do not ensure complete evidence.

| Mechanism and source entrypoint | Contract to understand | Counterexample exercise |
|---|---|---|
| [Action Fusion L95–125](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/action-fusion/then-run.ts#L95-L125) | The same-file queue covers mutation and follow-up command; mutation failure skips the command, while command failure retains the mutation. | Fail the follow-up check and confirm that the file remains changed; two content hashes do not provide a global lock against external writers. |
| [ObservationPack L137–209](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/observation-pack/index.ts#L137-L209) | Eligible large text is projected in full twice, then represented by a stable reference; replacement follows archival, without editing session history in place. | Put a critical fact in the middle; recall from zero using `next_offset` and compare reconstructed bytes. |
| [EPR validation L82–143](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/receipt.ts#L82-L143) | [Archive and delegate receipt generation to a model L77–147](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/evidence-preserving-reducer/index.ts#L77-L147); validate schema, source hash, status, and quotations, preserving original output on validation failure. | Construct a receipt with authentic quotations that omits another critical failure, distinguishing authenticity from coverage. |
| [Online Context Compact L289–370](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L289-L370) | At a plan boundary, cost and compactability checks can stop the run; `agent_settled` then invokes Pi's native compaction. | Compare an economic approval with a context that cannot be natively compacted; only success [triggers continuation L381–407](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/extension.ts#L381-L407). |

<a id="4-回访-m09减少-token-不自动减少账单"></a>

## 4. Revisit M09: fewer tokens do not automatically mean a smaller bill

Compaction reduces subsequent replay but can introduce cache rewriting and summarization costs. Identify cache read/write units before explaining the upstream decision model. It uses a configured price ratio and estimated request count; it is not an actual billing calculator. [Economics L149–196](https://github.com/NVlabs/SoL-Pi/blob/d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0/src/sol-pi/extensions/online-context-compact/economics.ts#L149-L196).

```text
savingTokens = archiveTokens - memoTokens
incrementalRatio = max(cacheWriteReadRatio - 1, 0)
breakevenRequests = writeTokens * incrementalRatio / savingTokens
combinedBreakevenRequests = (carriedDebtTokens + writeTokens * incrementalRatio) / savingTokens
```

The last two expressions require `savingTokens > 0` and an available price ratio. Vary remaining requests, first versus subsequent compaction, outstanding cache cost, and window headroom in turn. Predict `reason` before execution. Window protection can bypass economic gates but still requires positive savings; the outer layer can also reject a context that cannot be natively compacted. Explaining each branch matters more than memorizing a single compaction threshold.

<a id="5-cpu-实验与本次已验证范围"></a>

## 5. CPU experiment and the scope verified in this study

[Experiment 004](../experiments/004-sol-pi-contracts/README.md) loads the actual upstream `economics.ts` and executes eight deterministic scenarios with Node `--experimental-strip-types`. Preserve inputs, expected results, actual values, and assertions. Do not substitute a separately implemented equivalent function and describe it as upstream execution. The isolated-checkout validation environment below was Windows, Node 22.19.0, and npm 10.9.3. Failures remain part of the results.

| Check | Result in this study | Supported conclusion and limit |
|---|---|---|
| Economics function | All 8 scenarios PASS | Verifies decision contracts for these inputs; no live compaction was run. |
| TypeScript | PASS | Type checking passed in the recorded dependency environment. |
| Complete Vitest suite | 139 cases: 134 PASS, 5 FAIL | 2 `spawnSync npm ENOENT`, 2 symlink `EPERM`, and 1 POSIX file-mode mismatch between expected `0600` and actual `0666`; the suite was not all green. |
| Package checks | Dry-run: 35 files; 9 package assertions PASS | Verifies the package manifest and assertions; no package was published. |
| Pi compatibility check | PASS | Limited to the public API and version environment checked in this run. |
| Live model and cost evaluation | Not executed | No online provider calls; no measurement of live compact-and-continue behavior, task success rate, or cost savings. |

Reading upstream tests explains their intended guarantees; execution records show what passed on this machine. Do not remove platform failures and report the original suite as passing, or describe function assertions as improved model capability. Preserve the trace, environment, and cause of a failure before deciding whether to rerun on a compatible platform.

<a id="6-m14-配对评估先过质量门再谈效率"></a>

## 6. M14 paired evaluation: pass the quality gate before discussing efficiency

The next step is a live-run plan and has not been executed. For each task, fix the model version, code snapshot, initial files, permissions, budget, and independent verifier. Compare an all-disabled baseline with variants enabling one mechanism at a time; test combinations after individual checks. Repeat runs and preserve every failure, rather than selecting successful samples.

- **Quality gate:** Predeclare completion criteria and acceptable degradation. Check omitted work, lost evidence, recovery errors, and final artifacts. Classify verifier failures separately from task failures.
- **Cost gate:** Aggregate actual usage and charges for the main model, reducer, summaries, retries, recall, and cache reads/writes, alongside end-to-end latency. Do not claim an efficiency improvement when quality fails the gate.
- **Comparison record:** Preserve paired quality differences, cost differences, request counts, and failure categories. Report uncertainty across repeated runs; state the tradeoff when a variant is cheaper but slower.

<a id="7-交付物与退出标准"></a>

## 7. Deliverables and exit criteria

Finish the reproducible CPU deliverables before choosing an API or local-model extension. Path tracing, function boundaries, and evaluation design need no GPU. GPU or online execution is a separate budgeted experimental stage whose results must not be filled in beforehand.

| Deliverable | Exit criterion |
|---|---|
| `reading-map.md` and four path diagrams | Each claim has a pinned source entrypoint; distinguish normal paths, errors, and uncovered boundaries. |
| `contracts.json` and raw execution logs | Reproduce the eight scenarios and explain at least one economic rejection and one window-protection decision. |
| `failure-analysis.md` | Accurately record the five upstream test failures and environmental factors, without inventing fixes or reruns. |
| `paired-eval-plan.md` | List tasks, baseline, variables, budget, quality/cost gates, and failure categories; freeze the plan before execution. |

Core completion requires explaining why a shorter observation can harm later work and demonstrating how to detect that harm. A positive efficiency result is not a passing requirement.

<a id="8-m13-的未来连接轨迹与奖励仍需另建契约"></a>

## 8. The future M13 connection: trajectories and rewards need another contract

Continue through the [post-training curriculum](../curriculum/03-posttraining-and-agents.md#m13) and [knowledge tree](../KNOWLEDGE_TREE.md) to connect harness behavior to trajectories and rewards. SoL-Pi's public package does not implement RL parameter updates, and session text is not a complete training sample. Training integration must additionally preserve actual sampled tokens, logprobs, loss masks, model/policy versions, branches, and verifier results, while explaining visibility before and after compaction. This is a future research connection, not a training loop executed in this study.
