<a id="pi从-agent-loop-追踪上下文工具与事件顺序"></a>

# Pi: Tracing context, tools, and event order through the agent loop

Date: 2026-09-08  
Stage: L2, mechanism tracing in the low-level agent loop; only a portion of the entire project has been read.  
Source: [earendil-works/pi](https://github.com/earendil-works/pi) @ `6160683a4a8012f0d1cd30c145df18b4ca6f5176`  
Verification scope: Actually read agent-loop.ts and types.ts; did not install dependencies, run Pi, or call a model.

<a id="本次问题"></a>

## Question for this reading

After a model returns multiple tool calls, which work happens first? Is tool completion order the same as the order presented to the next model request? At what boundary does a user's additional instruction enter during a run?

<a id="源码入口"></a>

## Source entry points

| File | Mechanism read in this pass | Pinned-version evidence |
|---|---|---|
| [agent-loop.ts](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts) | Outer follow-up loop and inner tool/steering loop | [L156–273](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L156-L273) |
| Same file | Context transformation, provider-message conversion, and streamed results entering context | [L279–369](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L279-L369) |
| Same file | Sequential/parallel tool dispatch and result ordering | [L409–560](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L409-L560) |
| [types.ts](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/types.ts) | Explicit contracts for extension hooks and events | [L149–293](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/types.ts#L149-L293) |

<a id="执行链路"></a>

## Execution chain

```text
runAgentLoop(prompts, context, config)
  → add prompts / agent_start / turn_start
  → runLoop
      → admit steering
      → streamAssistantResponse
          → transformContext(AgentMessage[])
          → convertToLlm(Message[])
          → streamFunction(model, context)
      → extract tool calls
      → prepare arguments / validate / beforeToolCall
      → tool.execute → afterToolCall
      → tool results enter context
      → turn_end → shouldStopAfterTurn
      → tools or steering require continuation: call model again after prepareNextTurn
      → otherwise check follow-up queue
  → agent_end
```

`AgentMessage` is an internal application message; `Message` is a message a provider accepts. `transformContext` prunes or injects at the application-message layer, followed by protocol conversion in `convertToLlm`. This abstraction helps separate UI events from what the model actually sees. The low-level interface documentation requires these callbacks not to throw, because doing so may interrupt the normal event sequence; automatic loop recovery from every extension exception cannot be assumed.

<a id="已确认的取舍"></a>

## Confirmed tradeoffs

1. **Parallel execution still preserves deterministic model-message order.** Preflight runs sequentially; only tools that pass checks are scheduled concurrently. `tool_execution_end` can be emitted in completion order, but after `Promise.all` collects results, tool-result messages are published in the original assistant-call order. The UI can update promptly while the next model input retains stable ordering.
2. **One tool requiring sequential execution sends the whole batch down the sequential branch.** Actual parallelism cannot be inferred solely from the global `toolExecution` setting.
3. **Output truncation is not an ordinary tool-call failure.** When `stopReason === "length"`, all tools in the batch receive error results; potentially truncated arguments are not executed, even if they appear parseable. After receiving error feedback, the model can resend a complete call. [Implementation L226–239, L379–403](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L226-L239)
4. **Steering enters at call boundaries.** The current assistant's tool calls finish before new messages are checked; this is not a mechanism for arbitrarily interrupting a tool with side effects. `prepareNextTurn` can perform longer operations such as compaction, and the code then fetches steering that has not yet been admitted again.
5. **Batch termination requires every completed call to return terminate.** The distinction between `some` and `every` changes control behavior with multiple tools. [L589–590](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L589-L590)

<a id="手工推演两个工具一条新消息"></a>

## Manual walkthrough: two tools and one new message

Suppose the assistant calls two tools that permit parallel execution in the order `[read_a, read_b]`, and `read_b` finishes first. The event order derived from the source is: two preflights → concurrent execution → `read_b` completion → `read_a` completion → tool-result messages written in `[read_a, read_b]` order → `turn_end` → admission of queued steering → next model request. This is a source-based walkthrough, not yet a runtime trace.

<a id="与其他项目的连接"></a>

## Connections to other projects

- **Conceptual correspondence: Pi ↔ DeepSeek Harness.** Both handle context projection, model calls, tool results, and input during execution. DeepSeek's log-reconstruction invariant can be compared with Pi's `transformContext` / `convertToLlm` interfaces; this reading neither found nor claims a direct dependency between them.
- **Conceptual correspondence: Pi ↔ slime/Miles.** Harness message transformations, parallel tools, and branching affect RL trajectory construction. Correct application messages/logs do not imply correct token IDs, logprobs, or loss masks; training integration still needs a dedicated adapter.
- **Actual optional integration: Verifiers → Pi.** Further cross-reading found an explicit Pi ACP/provider adapter in Verifiers, with npm pinned to 0.84.1, alongside a Harbor taskset adapter. Thus, harness/task combinations can be studied within the same environment framework; this does not mean Pi directly depends on Harbor, and not all combinations have been verified to run. [Verifiers pinned-version implementation](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/harnesses/pi/harness.py#L19-L192)

<a id="待执行实验"></a>

## Pending experiment

- Inputs: a faux model that always produces two tool calls; tools use controlled delays without accessing real files.
- Variables: parallel/sequential execution, one tool marked sequential, output length truncation, and steering arrival time.
- Observe: completion-event order, tool-result order, model-call count, and actual tool-execution count.
- Hardware: CPU; prepare Node dependencies following upstream instructions, then run only targeted faux-provider cases.
- Actual result: not executed. The manual walkthrough is not counted as runtime verification.

<a id="下一步"></a>

## Next steps

- [ ] Read `harness/runtime/drive/generation.ts` and the reducer to verify the relationship between the low-level loop and higher-level durable runtime.
- [ ] Trace one JSONL-session restoration and record which events persist and which serve only live UI updates.
- [ ] Implement the controlled event-order experiment above before increasing the corresponding mechanism's verification level.
