<a id="claude-code--agent-sdk控制协议权限与-harness-的实现边界"></a>

# Claude Code / Agent SDK: control protocols, permissions, and harness implementation boundaries

Date: 2026-09-08. Stage: L1 targeted source reading. Official repository: [Claude Agent SDK for Python](https://github.com/anthropics/claude-agent-sdk-python), pinned SHA: `f1315c69a74db1c15fed2e5974918495d90b7d57`.

The local official SDK checkout contains 144 tracked files. The SDK was not installed, Claude was not started, no model API was called, and upstream tests were not run. Results of the curriculum's original CPU experiment are recorded in the [experiment guide](../../experiments/harness-state-machine/README.md). Historical mirror observations and provenance limits appear separately in the second part of this note.

<a id="先确定读的是哪一层"></a>

## Establish which layer you are reading

The Python SDK communicates with the Claude Code CLI through a transport. Its readable source includes configuration conversion, message handling, permission callbacks, hooks, and process lifecycle code; it is not the complete internal CLI source. Official distribution packages bundle the CLI, as described in the [README](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/README.md#L15-L21) and [CLI discovery implementation](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L247-L263). A shallow source checkout does not mean the CLI from the distribution package has been downloaded.

Study order: [Pi](pi.md) → [Claude Code advanced guide](../../handbook/05-claude-code-harness.md) → the call chain below → CPU failure experiments → [APEX trajectories and training interfaces](apex-agents-skyrl-recipe.md). The last two projects are connected pedagogically, not by an implemented integration.

<a id="按一次控制请求读源码"></a>

## Follow one control request through the source

1. **Configuration becomes CLI arguments.** Start with [`_build_command`](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L562-L586), including `stream-json` and system prompt arguments, then inspect [resume/session arguments](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L632-L645) and [fork arguments](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L693-L711). Argument forwarding establishes interface wiring, not the internal recovery algorithm or exactly-once external side effects.
2. **The CLI issues a permission request.** [`_handle_control_request`](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/query.py#L469-L529) dispatches by subtype. Its `can_use_tool` branch builds context containing tool/agent information, invokes the caller's callback, and encodes the Allow/Deny result as a control response. Allow can return modified input; Deny can carry interrupt. The `signal=None` here also means complete cancellation-signal support cannot be inferred from a callback type name.
3. **Responses require a live bidirectional channel.** The [input-stream lifecycle](https://github.com/anthropics/claude-agent-sdk-python/blob/f1315c69a74db1c15fed2e5974918495d90b7d57/src/claude_agent_sdk/_internal/query.py#L819-L861) waits for a result boundary before ending input when SDK MCP, hooks, or permission callbacks need bidirectional communication. Its comments explicitly warn that one result need not end the lifetime of all background tasks and record a limitation for multi-message input. This boundary case was not executed in this study.

<a id="能推出什么不能推出什么"></a>

## What the evidence establishes and what remains open

| Pinned source fact | Engineering implication and validation boundary |
|---|---|
| Callback results are encoded as allow/deny control responses | Permissions are part of the execution protocol; all CLI execution paths, argument checks, and operating-system sandbox behavior still need separate validation |
| Resume and fork are converted to CLI options | Session identity is part of the runtime contract; automatic restoration of external tool state does not follow |
| Control requests depend on the stdin/stdout lifecycle | Input cannot arbitrarily be closed after the first text output; real local process cancellation and background-task cleanup remain untested |
| SDK and CLI are separately versioned artifacts | Reproducible experiments must record SDK commit/package, CLI, model, settings, and permissions, not just one SDK SHA |

The curriculum's failure question is whether to replay a tool after it has changed the environment but before its result has been journaled. Our [state-machine experiment](../../experiments/harness-state-machine/README.md) marks this as an unknown outcome and stops automatic retry. This is a conservative teaching design, not a claim about Claude Code internals or a cross-process exactly-once guarantee.

<a id="与训练知识树连接"></a>

## Connect the case to the training knowledge tree

Compare log projection in [DeepSeek Harness](deepseek-harness.md) with tool events in Pi. Distinguish the complete event record, the next model context, and a trainable token sequence. Agent RL additionally requires actual sampled token IDs, logprobs, loss masks, policy versions, and branch information; SDK text messages are not direct substitutes for those quantities.

First-pass deliverables: draw the Python SDK / CLI / model / tool boundaries and mark where permission responses occur; produce a failure trace with a started action whose outcome is unknown; explain how compaction changes the next input. Then design a real SDK comparison with the same tasks, permissions, and budgets.

<a id="官方资料与阅读状态"></a>

## Official materials and reading status

- [Official agent loop documentation](https://code.claude.com/docs/en/agent-sdk/agent-loop): public interfaces and current behavior, checked on 2026-09-08; not a pinned historical implementation.
- [Context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): tradeoffs involving compaction, notes, retrieval, and subtasks.
- [Long-running harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents): the authors' experiments with initialization, progress artifacts, and independent verification.

<a id="claude-code-历史镜像有限范围证据笔记"></a>

## Historical Claude Code mirror: bounded evidence notes

Checked 2026-09-08. This is a static reading aid for a third-party historical mirror, not an authenticated Anthropic release or evidence of current production behavior. No mirror code or dependency was installed or executed; no source files are included here.

<a id="快照选择与来源边界"></a>

### Snapshot selection and provenance

Use mirror commit `f5a40b86dede580f6543bf8926c9af017eea9409`. GitHub's [history at this SHA](https://github.com/jaicorn/claude-code-source/commits/f5a40b86dede580f6543bf8926c9af017eea9409/) shows one March 31, 2026 commit, attributed to `realsigridjin`, titled “init: add source code from src.zip.” Its [root tree](https://github.com/jaicorn/claude-code-source/tree/f5a40b86dede580f6543bf8926c9af017eea9409) contains only `src/`.

The advertised `backup` route did not establish a usable snapshot: both `jaicorn/claude-code-source/tree/backup` and the README's actual target, `nirholas/claude-code/tree/backup`, returned 404. The inspected branch inventory listed only `main`. Do not label the selected SHA as the backup branch.

The SHA pins the mirror content and its local history. I did not obtain and compare an independently authenticated Anthropic artifact, signed manifest, or checksum. Consequently, neither the SHA nor the import message proves that the tree is byte-for-byte original, complete, or representative of the shipped feature configuration. The repository describes the material as Claude Code v2.1.88; that attribution remains a mirror claim.

<a id="三处小范围阅读入口"></a>

### Three small reading loci

All anchors below use physical GitHub source line numbers, cross-checked in the corresponding blame view; the web tool's normalized raw-text line numbers were not used.

| Locus | Static observation | Transferable question |
| --- | --- | --- |
| [query.ts, lines 826–862](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/query.ts#L826-L862) | Assistant tool-use blocks set a follow-up flag. A streaming executor receives tool blocks and contributes completed results normalized for the API. | How does the harness turn model-proposed actions and observed results into the next model input? |
| [autoCompact.ts, lines 28–49](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/compact/autoCompact.ts#L28-L49), [62–91](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/compact/autoCompact.ts#L62-L91), [257–265](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/compact/autoCompact.ts#L257-L265) | The effective context size reserves output space; an additional buffer controls automatic compaction. A consecutive-failure check prevents repeated compaction attempts. | How much room must recovery retain, and when should unsuccessful recovery stop? The numeric constants are historical implementation details, not general defaults. |
| [toolExecution.ts, lines 916–931](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/tools/toolExecution.ts#L916-L931), [995–999](https://github.com/jaicorn/claude-code-source/blob/f5a40b86dede580f6543bf8926c9af017eea9409/src/services/tools/toolExecution.ts#L995-L999) | The harness awaits a permission resolver using the tool, processed input, context, and permission callback, then branches on a non-allow decision. | Where is an action proposal converted into an authorization decision? This small reading does not establish complete permission correctness. |

<a id="镜像维护者增补与通知记录"></a>

### Maintainer additions and notice history

The [initial services tree](https://github.com/jaicorn/claude-code-source/tree/f5a40b86dede580f6543bf8926c9af017eea9409/src/services) omits an `x402` directory; the [later mirror tree at d43bd40](https://github.com/jaicorn/claude-code-source/tree/d43bd40690853fd323758e038cb686930d53a39f/src/services) includes it. This is a directory-level comparison, not a whole-tree authenticity audit.

In the [maintainer's correction, pinned at e37f8a9](https://github.com/nirholas/fresh-start/blob/e37f8a9ea1a7ab4d83432d93bd357a332549e924/HISTORY.md), nirholas claims authorship of x402 and identifies generated docs, web-terminal work, shims, build declarations, Docker, and the explorer MCP server as additions. His accounting is explicitly limited through 2026-03-31T12:43Z. Treat these as first-person provenance claims; do not attribute those additions to Anthropic.

The [March 31 notice](https://github.com/github/dmca/blob/e8211dc62ad9c3a23d3f5c2878a4557d6b9f16fd/2026/03/2026-03-31-anthropic.md) alleges infringement by the nirholas repository and identifies the work as not open-source licensed. GitHub says network-wide processing affected 8.1K repositories. The [April 1 partial retraction](https://github.com/github/dmca/blob/615484fed194981ef67a284ffe08c5625839e077/2026/04/2026-04-01-anthropic-retraction.md) retains the notice for the parent and 96 individually named fork URLs, while requesting reinstatement of the others. Retraction is not a source-code license or authentication statement.

For a public handbook, retain original analysis, small fixed references, and explicit provenance labels. Do not import this mirror, its generated documentation, or its prompts as handbook assets. These three loci support a historical harness case study; they cannot support claims about the current Claude Code implementation, active feature flags, or proprietary model internals.
