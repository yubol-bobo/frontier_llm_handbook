<a id="deepseek-harness如何让模型请求可以由日志重建"></a>

# DeepSeek Harness: Making model requests reconstructible from logs

Date: 2026-09-08  
Stage: L1, initial reading of the architecture and one key implementation chain.  
Source: [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) @ `c389f96bf3a9b6807cb71ed6bdad5849be0df6d8`  
Verification scope: Architecture documentation, the agent's preStep/turn/step/buildRequest, and the request invariant; did not start the harness.

<a id="本次问题"></a>

## Question for this reading

When plugins can modify context and model parameters, how can a saved session still explain exactly what the model saw at the time?

<a id="阅读路径"></a>

## Reading path

| Entry point | Content | Pinned version |
|---|---|---|
| [architecture.md](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/docs/architecture.md) | Cordis, event categories, and session/capability layering | [L55–127](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/docs/architecture.md#L55-L127) |
| [agent.ts](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/packages/core/agent-loop/src/agent.ts) | Input admission, turn/step, and request construction | [L236–361](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/packages/core/agent-loop/src/agent.ts#L236-L361) |
| Same file | request/header and the frozen request | [L522–595](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/packages/core/agent-loop/src/agent.ts#L522-L595) |
| [invariant.ts](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/packages/core/agent-loop/src/invariant.ts) | Checks at llm/stream whether the request can be reconstructed from logs | [L19–54](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/packages/core/agent-loop/src/invariant.ts#L19-L54) |

<a id="一次请求怎样产生"></a>

## How a request is produced

```text
inbox.claim
  → assemble prompt / context
  → agent/pre-step waterfall (enter / reject)
  → step/start
  → append model input as user/message
  → session.deriveMessages()
  → agent/request waterfall / provider prepareCall
  → log canonical request/header
  → freeze messages, request header, and request object
  → llm/stream
  → durable settlement of assistant/message or assistant/attempt
  → tool execution / tool/result
```

A turn can contain multiple steps; a step is one model request plus its tool operations. When the first batch of inputs is rejected or rewritten to be empty, a turn boundary can still occur without a model call. Counting turns therefore cannot be directly equated with counting model requests.

<a id="核心机制与取舍"></a>

## Core mechanisms and tradeoffs

**Model-visible content must be logged.** `buildRequest` saves not only chat text but also a request header containing model configuration, system prompt, tools, and more. The header's reason differs when resuming execution, changing configuration, or starting a new request sequence. The subsequently constructed request is frozen, reducing the risk of other plugins rewriting it after dispatch.

**The invariant is a concrete runtime hook.** `invariant.ts` registers as a companion plugin on `llm/stream`, checks the loop marker, frozen state, valid session, step/start, and request/header, and compares messages with `deriveMessages()` and key header fields. It makes reconstructibility a checkable constraint. Whether a specific profile enables this companion requires further configuration tracing; the existence of its source alone does not establish that all deployments perform this check.

**Durable events and live events are separate.** Live stream chunks update the UI; the full stream enters the log when a successful message or failed attempt is settled. The architecture document explicitly states that an unsettled attempt stream is not guaranteed to persist if the process exits abruptly before settlement. Logging a failed attempt also does not mean that the failed text automatically becomes context for the next model call.

**Plugins have explicit roles.** A Service Definition declares a capability, a Provider implements it, and a Consumer uses it. Moving local execution into a sandbox requires checking that the filesystem and subprocesses belong to the same execution environment; simply replacing the name of a “bash tool” is insufficient.

<a id="跨项目连接"></a>

## Cross-project connections

- **Conceptual correspondence: Pi.** Compare their input admission, context projection, and event ordering without assuming they share a runtime.
- **Conceptual correspondence: trajectory correctness in Miles/slime.** Log reconstruction establishes the provenance of application-layer messages; training must additionally demonstrate that tokens, sampled logprobs, model versions, and loss masks match the sampling process. These two forms of correctness are related but do not replace each other.
- **External dependency: Cordis.** This snapshot contains vendored Cordis; it is an internal architectural dependency of the harness, not one of the other cloned training projects.

<a id="待执行实验"></a>

## Pending experiment

Create a keyless provider fixture. Modify one context item before log admission and then after request construction, and check the request-reconstruction invariant's acceptance/rejection behavior. Then test recovery semantics when an abrupt exit occurs between live streaming and durable settlement. The required resources are Node and a local fixture; training GPUs are unnecessary. This study only read source and did not execute the experiment.

<a id="下一步"></a>

## Next steps

- [ ] Trace how one shipped profile loads the invariant companion.
- [ ] Trace `deriveMessages()` after a fork / compaction.
- [ ] Compare the same failure scenario in Pi, rather than comparing the two README feature lists.
