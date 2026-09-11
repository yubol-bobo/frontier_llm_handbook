<a id="实验-004直接检验-sol-pi-的上下文压缩决策"></a>

# Experiment 004: directly checking SoL-Pi context-compaction decisions

Recorded: 2026-09-11. The aim is to turn “compaction saves money” into a calculable, falsifiable condition. This CPU experiment directly calls `decideCompaction` in the pinned upstream source. It calls no model, installs no Pi CLI, and performs no live session compaction. Its results support only the listed decision scenarios.

<a id="1-先预测再执行"></a>

## 1. Predict before running

Read the [SoL-Pi guide](../../handbook/06-sol-pi-efficient-harnesses.md) and [source notes](../../notes/repositories/sol-pi.md) first. Requirements are Git, Python, and Node 22.19.0 as used for this verification. Node reads upstream TypeScript with `--experimental-strip-types`. No GPU or model API is required.

Run from the handbook root; skip the first command if the pinned sources are already available:

```powershell
python tools/clone_repos.py --group harness --restore-lock
node --experimental-strip-types experiments/004-sol-pi-contracts/check.mjs
```

The first command restores the entire harness group, not only SoL-Pi. The tool does not reset existing checkouts; a mismatched existing HEAD produces an error and is preserved. The checker requires SoL-Pi HEAD to match `d7ecfc089944f0d04b80122a0a9a6ca0d786f3d0` exactly, with no changes to tracked files. It depends only on Node built-ins and the upstream pure function; no npm installation in `sources/sol-pi` is needed.

Implementation: [check.mjs](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/004-sol-pi-contracts/check.mjs). Recorded inputs, decision fields, and environment: [results.json](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/004-sol-pi-contracts/results.json). Rerunning refreshes that result file and its UTC date.

<a id="2-共同输入与独立算术"></a>

## 2. Shared inputs and independent arithmetic

Base inputs are `W=80000, A=60000, M=1000, r=12.5, D=0`; current context 80000, window 200000, average positive growth 2000; completed-boundary request counts `[4,6,5]`, four remaining boundaries, and zero prior compactions. r is an example configuration, not a current API price.

```text
S = A - M = 59000
incremental_cache_cost = W * max(0, r - 1) = 920000
breakeven_requests = 920000 / 59000 ≈ 15.59322
predicted_requests = 1 + floor(5 * 4) = 21
window_request_cap = floor((200000 - 80000) / 2000) = 60
first_effective_horizon = min(2 * 21, 60) = 42
```

The checker asserts upstream decisions and reasons, then independently checks this arithmetic and the ratio with carried debt. It does not substitute a copied implementation for the actual upstream function.

| Scenario | Changes from base inputs | Observed decision | Reason |
| --- | --- | --- | --- |
| first | None | Compact | `economic` |
| later_margin | One prior compaction | Defer | `deferred_subsequent_margin` |
| later_longer_horizon | One prior compaction, six remaining boundaries | Compact | `economic` |
| unpaid_debt | Previous row plus 2000000 debt | Defer | `deferred_carried_debt` |
| window_pressure | W/current context 190000, A=150000, r=100 | Compact | `window_protection` |
| no_saving | A=M=1000 | Defer | `non_positive_saving` |
| no_history | Completed-boundary samples null | Defer | `horizon_unavailable` |
| no_price_ratio | r=null | Defer | `cache_ratio_unavailable` |

**Observed result: 8/8 scenarios and independent arithmetic assertions passed.** The subsequent-compaction margin is `1.5 × 15.59322≈23.39`, so 21 predicted requests are insufficient. Increasing remaining boundaries to six predicts 31 requests and passes the margin. With carried debt, combined breakeven is about 49.49 requests, so 31 is still insufficient.

<a id="3-上游完整检查的真实结果"></a>

## 3. Actual outcome of the complete upstream check

A separate checkout outside the handbook used the same SHA, Windows NT 10.0.26200, Node 22.19.0, npm 10.9.3, and locked Pi 0.84.2 development dependencies to run these commands. They document completed work; they are not installation steps for the dependency-free CPU checker.

```powershell
npm ci --ignore-scripts
npm run check
npm pack --dry-run --json
node scripts/check-pi-compat.mjs
```

`npm ci` succeeded. `npm run check` **exited with code 1**: TypeScript passed; Vitest reported 134 of 139 tests passed, five failed, zero skipped, with 15 of 18 files passing. The combined command's packaging stage was not reached after the earlier failure. A separate dry run succeeded with 35 package entries and nine equivalent content assertions passing. The read-only Pi API probe also passed. [Machine-readable record](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/experiments/004-sol-pi-contracts/upstream-validation.json).

| Failure count | Test location | Observation and limit |
| --- | --- | --- |
| 2 | `tests/package.test.ts:15–19` | Shell-free `spawnSync("npm", …)` produced `ENOENT` in this Windows environment. A successful separate package run does not change the original test failures. |
| 2 | `tests/observation-pack.test.ts:254,292` | Symlink fixture creation failed with `EPERM`, before the intended guard assertions. |
| 1 | `tests/evidence-preserving-reducer.test.ts:296` | File-mode reporting returned `0666`, failing the `0600` assertion. This does not establish Windows ACL privacy. |

Tracked source remained clean in both checkouts. Tests were not changed or skipped, and system permissions were not adjusted to make results green. No Linux rerun has been performed. The upstream suite uses mock providers and some real temporary-file operations. A real `AgentSession` with deterministic summaries verifies lifecycle behavior, not actual model-summary quality.

<a id="4-学会解释失败再扩展实验"></a>

## 4. Explain failures before extending the experiment

- Change `remainingBoundaries` from four to six, write a prediction first, then call the function in a separate experiment script. Explain why equally long contexts can produce different decisions.
- Keep ample window space and set r to null, then increase window pressure. Explain unavailable economic mode versus window protection. Passing the pure function does not establish a successful session boundary or a feasible native cut point.
- Propose a cold-cache or early-task-completion counterexample: why does `compact=true` fail to establish a lower bill? List the actual read/write tokens, summarization charges, and final task outcomes that must be measured.
- Use source behavior for cancellation, archive errors, and omitted evidence to design another fixture set. Record only executed scenarios as execution evidence; do not mark exercise prompts complete.

Deliver your input table, predicted and actual decisions, failure explanations, and explicit applicability limits. Some conditions in the first two exercises reproduce recorded scenarios. Combining a null ratio with window pressure, and the new cancellation, archive, and omitted-evidence fixtures, remain unexecuted. This study did not reproduce quality evaluation, online compaction, cache hits, speed, or costs.
