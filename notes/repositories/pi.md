# Pi：从 agent loop 追踪上下文、工具与事件顺序

日期：2026-09-08  
阶段：L2，低层 agent loop 机制追踪；整个项目只完成局部阅读。  
源码：[earendil-works/pi](https://github.com/earendil-works/pi) @ `6160683a4a8012f0d1cd30c145df18b4ca6f5176`  
验证范围：实际阅读 agent-loop.ts 与 types.ts；没有安装依赖、运行 Pi 或调用模型。

## 本次问题

模型返回多个 tool calls 后，哪些工作先发生？工具完成顺序与进入下一次模型请求的顺序是否相同？用户在运行中补充要求时，消息在哪个边界进入？

## 源码入口

| 文件 | 本次读到的机制 | 固定版本证据 |
|---|---|---|
| [agent-loop.ts](../../sources/pi/packages/agent/src/agent-loop.ts) | 外层 follow-up、内层工具/steering 循环 | [L156–273](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L156-L273) |
| 同一文件 | context transform、provider 消息转换、流式结果落入 context | [L279–369](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L279-L369) |
| 同一文件 | sequential/parallel 工具分派及结果排序 | [L409–560](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L409-L560) |
| [types.ts](../../sources/pi/packages/agent/src/types.ts) | 扩展钩子和事件的明确约定 | [L149–293](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/types.ts#L149-L293) |

## 执行链路

```text
runAgentLoop(prompts, context, config)
  → 添加 prompts / agent_start / turn_start
  → runLoop
      → 接纳 steering
      → streamAssistantResponse
          → transformContext(AgentMessage[])
          → convertToLlm(Message[])
          → streamFunction(model, context)
      → 提取 tool calls
      → 参数准备 / 校验 / beforeToolCall
      → tool.execute → afterToolCall
      → 工具结果进入 context
      → turn_end → shouldStopAfterTurn
      → 工具或 steering 需要继续：prepareNextTurn 后再调用模型
      → 否则检查 follow-up 队列
  → agent_end
```

`AgentMessage` 是应用内部消息；`Message` 是 provider 能接受的消息。`transformContext` 在应用消息层裁剪或注入，`convertToLlm` 再做协议转换。这里的抽象有助于把 UI 事件与模型实际看见的内容分开。低层接口文档要求这些回调不要抛错，否则可能打断正常事件序列；不能假设所有扩展异常都由循环自动恢复。

## 已确认的取舍

1. **并行执行仍保留确定的模型消息顺序。** Preflight 是顺序执行的，通过检查的工具才被并发调度。`tool_execution_end` 可按完成时间发出，但 `Promise.all` 收集后，tool-result message 按原 assistant 调用顺序发布。UI 可以及时更新，下一轮模型输入仍有稳定次序。
2. **某个工具要求 sequential 会让整个 batch 走串行分支。** 不能仅检查全局 `toolExecution` 就推断实际并行度。
3. **输出截断不是普通工具调用失败。** `stopReason === "length"` 时本批工具全部生成错误结果，不执行可能被截断的参数，即使参数表面上还能解析。模型得到错误反馈后可以重发完整调用。[实现 L226–239、L379–403](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L226-L239)
4. **Steering 在调用边界进入。** 当前 assistant 的工具调用先完成，再检查新消息；它不是随时中断一个有副作用工具的机制。`prepareNextTurn` 可用于压缩等较长操作，代码会再次取尚未接纳的 steering。
5. **批次终止要求所有已完成调用都返回 terminate。** `some` 与 `every` 的区别会改变多工具情况下的控制行为。[L589–590](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts#L589-L590)

## 手工推演：两个工具、一条新消息

设 assistant 按 `[read_a, read_b]` 调用两个允许并行的工具，`read_b` 先完成。源码推导的事件次序是：两个 preflight → 并发执行 → `read_b` completion → `read_a` completion → tool-result messages 按 `[read_a, read_b]` 写入 → `turn_end` → 接纳排队的 steering → 下一轮模型请求。此处是源码推演，尚非运行 trace。

## 与其他项目的连接

- **概念对应：Pi ↔ DeepSeek Harness。** 两者都处理上下文投影、模型调用、工具结果、运行中输入。DeepSeek 的日志重建 invariant 可以与 Pi 的 `transformContext` / `convertToLlm` 接口对照；本次没有发现或声称两者直接依赖。
- **概念对应：Pi ↔ slime/Miles。** Harness 的消息变换、并行工具与分叉会影响 RL trajectory 的构造。应用消息/日志正确不等于 token IDs、logprobs、loss mask 正确，训练接入仍需要专门适配。
- **真实可选集成：Verifiers → Pi。** 进一步交叉阅读发现 Verifiers 有明确的 Pi ACP/provider adapter，固定 npm 0.84.1，并且另有 Harbor taskset adapter。因此可以沿同一个环境框架研究 harness 与任务的组合；这不是 Pi 直接依赖 Harbor，也未验证所有组合能运行。[Verifiers 固定版本实现](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/harnesses/pi/harness.py#L19-L192)

## 待执行实验

- 输入：faux model 固定产生两个工具调用；工具使用受控延迟，不访问真实文件。
- 变量：并行/串行、单工具 sequential、输出 length 截断、steering 到达时刻。
- 观测：completion 事件顺序、tool-result 顺序、模型调用次数、工具实际执行次数。
- 硬件：CPU；先按上游说明准备 Node 依赖，只运行针对性 faux-provider 用例。
- 实际结果：未执行。没有将手工推演计为运行验证。

## 下一步

- [ ] 读取 `harness/runtime/drive/generation.ts` 与 reducer，验证低层 loop 与更高层 durable runtime 的关系。
- [ ] 追踪 JSONL session 的一次恢复，记录哪些事件持久化、哪些仅用于实时 UI。
- [ ] 编写上述受控事件顺序实验后，再提升对应机制的验证等级。
