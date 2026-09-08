<a id="claude-code-harness-专题从公开接口到可验证状态机"></a>

# Claude Code harness study: from public interfaces to a verifiable state machine

Date: 2026-09-08. The goal is to understand how mechanisms outside the model affect reliability, cost, and usable training trajectories. This study combines official material with an original course CPU exercise. It does not reconstruct the complete Claude Code implementation or require leaked mirrors as reading.

<a id="1-学习位置与证据边界"></a>

## 1. Placement and evidence boundaries

Study this after [M12: tasks, harnesses, verifiers, and trajectories](../curriculum/03-posttraining-and-agents.md#m12), and revisit [M09: inference serving and rollout cost](../curriculum/02-training-systems.md). Distinguish model requests, tool execution, environment state, and grading evidence before discussing recovery. Distinguish token cost from end-to-end latency before comparing efficiency. Integrating trajectories into RL also requires M11's synchronous-update foundations.

Use the [Claude Agent SDK repository note](../notes/repositories/claude-agent-sdk.md) and label each claim: **publicly observable**, supported by official documentation, pinned SDK source, or recorded interface behavior; **reading inference**, a design explanation requiring further testing; or **unverified internals**, such as complete CLI algorithms, unpublished server behavior, or unauthenticated leak content. Separate documented capabilities from local observations: availability in documentation does not mean a feature was run here.

Public SDK code supports studying interfaces, message handling, and control callbacks. Its README describes downloading and bundling the Claude Code CLI; readable SDK source therefore does not establish publication of the complete core harness. Record source, documentation, and package versions. A changing webpage does not replace a pinned snapshot.

<a id="2-官方阅读顺序"></a>

## 2. Official reading order

1. [How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop): draw the order of prompts, model responses, tool results, and termination messages. Then locate permissions, budgets, compaction, and sessions. Understand the public contract before tracing individual options.
2. [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): compare retrieval on demand, compaction, persistent notes, and subagents. Record what each preserves and loses.
3. [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents): trace how initialization, feature lists, progress files, Git history, and end-to-end tests support work across sessions. This is one experimental design, not a universally optimal recipe.
4. [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps): study separate generation and evaluation, explicit completion criteria, and the need to reassess harness complexity as models change.
5. [Claude Agent SDK for Python](https://github.com/anthropics/claude-agent-sdk-python): finally inspect `query`, the client, message types, hooks, and the CLI connection boundary. Draw the SDK package and its bundled executable as separate boxes; interface code is not the entire internal implementation.

<a id="3-六个学习主题"></a>

## 3. Six study topics

<a id="31-agent-loop谁决定谁执行何时停止"></a>

### 3.1 Agent loop: decisions, execution, and termination

The documented loop lets the model respond or request tools, then executes those tools and feeds results back until termination. Distinguish a model request, a tool interaction cycle, and the complete user task. Draw normal completion, tool denial, budget exhaustion, and execution failure. A completion statement establishes that the loop produced final output; independent acceptance checks still determine whether the task is complete.

<a id="32-context可见上下文不等于完整历史"></a>

### 3.2 Context: visible input is not the entire history

Official material discusses retrieval, compaction, structured notes, and subagent isolation. Separate what storage contains from what the current request actually receives. Compaction frees space but can discard details needed later; notes and retrieval introduce selection and maintenance responsibilities. Record visible input before and after compaction instead of guessing from the final answer. Do not infer exact thresholds, retention algorithms, or hidden prompts from conceptual articles.

<a id="33-tools-and-permissions授权判断先于副作用"></a>

### 3.3 Tools and permissions: decide before causing effects

The SDK documents tool permissions and lifecycle hooks. Focus on the execution boundary: after a model request, who validates input, decides permission, and permits environment changes? Check that denial is auditable and the rejected operation caused no side effect. A prompt asking for no writes cannot replace an execution constraint. Recording an approval reason does not itself prove the operation safe.

<a id="34-sessions恢复记录不等于证明执行结果"></a>

### 3.4 Sessions: restored records do not prove an outcome

Official documentation describes session continuation, resumption, and forking. The course adds a recovery question: an operation was recorded as started and the process exited; did its external effect happen? Missing completion could mean no execution or completed execution without bookkeeping. Do not automatically classify this as failure and repeat a mutation. Reconcile artifacts, reliable operation identifiers, or human evidence first. These are exercise requirements, not claims about Claude's internal recovery algorithm.

<a id="35-subagents隔离上下文与汇总信息的代价"></a>

### 3.5 Subagents: context isolation and summary costs

Official material presents subagents as a way to work in separate contexts. Map the parent task, child task, returned summary, and shared workspace; separate context does not inherently isolate files. Compare giving the parent the entire work history with returning a summary. Which saves context, and which loses evidence? Record information loss and duplicated work without assuming multiple agents must improve success.

<a id="36-evaluation完成声明要由证据支持"></a>

### 3.6 Evaluation: completion needs evidence

The long-running articles emphasize feature lists, incremental progress, end-to-end tests, and a separate evaluation role. The course consequently separates execution state, artifact correctness, and successful grading. Fix tasks, permissions, budgets, and initial environments before comparing designs; retain failures. A grader exception must not silently become a wrong-answer label. A higher score obtained with more tool calls does not alone establish better model capability.

<a id="4-原创-cpu-实验可审计的-harness-状态机"></a>

## 4. Original CPU exercise: an auditable harness state machine

Run the course implementation from the repository root:

```bash
python experiments/harness-state-machine/run.py
```

See the [experiment instructions](../experiments/harness-state-machine/README.md) for inputs, output paths, and execution records. It uses a deterministic mock provider and in-memory state, with no Claude calls, official SDK installation, API key, or GPU. It checks course-designed state-machine contracts; passing does not reproduce real model behavior.

Read the scenarios, run the program, and inspect artifacts in this order:

1. **Denial causes no side effect.** The mock requests a write and the permission layer rejects it. Compare file contents or an operation counter before and after, and inspect the denial event. A printed rejection with an already-modified file is a failure.
2. **Tool cycles are bounded.** The mock repeatedly requests tools. Set the maximum allowed tool cycles and check that execution stays within it with a distinct termination reason. Report this exercise's counting definition without presenting it as the SDK's full budget semantics.
3. **Compaction loss is explicit.** Insert a distinctive fact and intentionally omit it from a simplified summary. Compare the actual context supplied to the mock before and after compaction. Identify preserved information, missing information, and a later task lacking evidence. Character or message counts are simplified measures, not real tokenizer costs or model accuracy.
4. **Interruption preserves an unknown outcome.** Journal an operation's start and simulate interruption between its effect and completion bookkeeping. On recovery, a started entry without completion requires reconciliation and must not trigger another write automatically. Explain why interruption before the effect could leave the same journal: that record alone cannot establish which occurred.

Predict each transition before reading the trace. Change one permission or budget and explain which check should change. For recovery, separately answer what evidence is available and what action it permits. Reading a session journal does not itself provide exactly-once execution guarantees.

<a id="5-与其他项目的连接"></a>

## 5. Connections to other projects

| Comparison | Shared question to trace | Boundary in this study |
|---|---|---|
| [Pi](../notes/repositories/pi.md) | Context transformations, model-visible projections, and tool/event order | Compare interfaces and traces without assuming a shared runtime. |
| [DeepSeek Harness](../notes/repositories/deepseek-harness.md) | Reconstructing requests from logs; separating live events from durable records | Request reconstruction does not establish that an external effect's outcome is known. |
| [Harbor](../notes/repositories/harbor.md) | Artifact collection, independent verification, and task-versus-grader failures | A harness completion message cannot replace an independent verifier. |
| [APEX / SkyRL recipe](../notes/repositories/apex-agents-skyrl-recipe.md) | Converting interaction into actual tokens, logprobs, masks, and rewards | Mock chat logs are not RL samples; retokenizing text cannot manufacture sampling probabilities, especially after compaction or branching. |

These are conceptual comparisons, not claims of direct software dependencies between Claude Code and these projects. RL integration separately requires provenance for model versions, actual sampled tokens, behavior probabilities, and training masks. This CPU exercise performs no weight updates.

<a id="6-交付物与退出标准"></a>

## 6. Deliverables and exit criteria

Deliver a state-transition diagram; a claim/source/version/evidence-level table; raw logs and check results for the four scenarios; and a one-page comparison across projects. Record the command, Python version, actual execution date, and unexecuted work. Use the output filenames specified by the experiment instructions.

To pass, trace one request through permission, environment changes, and termination; show the exact fact lost during compaction; explain why an incomplete journal leaves an unknown outcome; demonstrate recovery without automatic repeated effects; and identify model capabilities the mock does not measure. Proceed to real SDK experiments after all four checks pass with reviewable evidence. If a check fails, retain the failing trace and explain the fix instead of rewriting expectations to conceal the problem.

<a id="7-可选扩展真实-sdk推理与训练"></a>

## 7. Optional extensions: SDK, inference, and training

A paid SDK/API extension can repeat harmless tasks in an isolated temporary project with fixed SDK, CLI, model, permissions, and budgets, recording real messages and charges. Redesign deterministic fixture assertions for the real interface. This extension is optional and unexecuted; no success rate, cost, or run result is supplied in advance.

A GPU extension can replace the mock with a controllable local model to study tool selection and summary loss. That compares another model with a custom harness and does not reproduce Claude Code. For RL, return to M11/M12's token and update contracts and add independent evaluation. Complete one falsifiable, reviewable question before increasing models, tasks, or concurrency.
