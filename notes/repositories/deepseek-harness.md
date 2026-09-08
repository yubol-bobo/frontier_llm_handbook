# DeepSeek Harness：如何让模型请求可以由日志重建

日期：2026-09-08  
阶段：L1，架构与一条关键实现链路初读。  
源码：[deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) @ `c389f96bf3a9b6807cb71ed6bdad5849be0df6d8`  
验证范围：架构文档、agent 的 preStep/turn/step/buildRequest、request invariant；未启动 harness。

## 本次问题

插件可以修改上下文和模型参数时，怎样保证保存的 session 仍能解释模型当时到底看见了什么？

## 阅读路径

| 入口 | 内容 | 固定版本 |
|---|---|---|
| [architecture.md](../../sources/deepseek-harness/docs/architecture.md) | Cordis、事件分类、session 与 capability 分层 | [L55–127](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/docs/architecture.md#L55-L127) |
| [agent.ts](../../sources/deepseek-harness/packages/core/agent-loop/src/agent.ts) | 输入接纳、turn/step、请求构造 | [L236–361](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/packages/core/agent-loop/src/agent.ts#L236-L361) |
| 同一文件 | request/header 和冻结后的请求 | [L522–595](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/packages/core/agent-loop/src/agent.ts#L522-L595) |
| [invariant.ts](../../sources/deepseek-harness/packages/core/agent-loop/src/invariant.ts) | 在 llm/stream 检查请求能否由日志重建 | [L19–54](https://github.com/deepseek-ai/deepseek-harness/blob/c389f96bf3a9b6807cb71ed6bdad5849be0df6d8/packages/core/agent-loop/src/invariant.ts#L19-L54) |

## 一次请求怎样产生

```text
inbox.claim
  → assemble prompt / context
  → agent/pre-step waterfall（enter / reject）
  → step/start
  → 进入模型的输入追加为 user/message
  → session.deriveMessages()
  → agent/request waterfall / provider prepareCall
  → canonical request/header 记入日志
  → 冻结消息、请求头与请求对象
  → llm/stream
  → assistant/message 或 assistant/attempt 的持久化结算
  → 工具执行 / tool/result
```

一次 turn 可以包含多个 step；step 是一次模型请求加其工具操作。第一批输入被拒绝或改写为空时，仍可能有 turn 边界而不发生模型调用。因此统计 turn 数不能直接等同于统计模型请求数。

## 核心机制与取舍

**模型可见内容必须记入日志。** `buildRequest` 不只保存聊天文本，还保存模型配置、system prompt、tools 等 request header。恢复运行、更换配置或开始新请求序列时，header 的 reason 不同。随后构造的请求被冻结，降低派发后被其他插件改写的风险。

**Invariant 是具体运行钩子。** `invariant.ts` 以 companion plugin 注册到 `llm/stream`，检查 loop 标记、冻结状态、有效 session、step/start、request/header，并比较 messages 与 `deriveMessages()`、关键 header 字段。它让“能重建”变成可检查的约束；是否在某个具体 profile 启用该 companion，需要继续追踪配置，不能仅凭存在源码就声称所有部署都执行此检查。

**Durable events 与 live events 分离。** 实时 stream chunk 用于 UI 更新；完整流在成功消息或失败尝试结算时进入日志。架构文档明确指出，进程在结算前硬退出时，尚未结算的 attempt stream 不保证持久化。记录了失败尝试，也不意味着失败文本自动成为下一次模型的上下文。

**插件具有明确的角色。** Service Definition 声明能力，Provider 实现，Consumer 使用能力。把本地执行换到 sandbox，需要检查文件系统与子进程是否属于同一执行环境，不能仅替换一个“bash 工具”名称。

## 跨项目连接

- **概念对应：Pi。** 对照两者的输入接纳、上下文投影和事件排序，而非假设共享 runtime。
- **概念对应：Miles/slime 的 trajectory 正确性。** 日志重建解决应用层消息来源；训练侧还要证明 token、采样 logprob、模型版本及 loss mask 与采样过程一致。这两种正确性有联系，但并不互相替代。
- **外部依赖：Cordis。** 该快照包含 vendored Cordis；这是 harness 内部架构依赖，不是已经克隆的其他训练项目。

## 待执行实验

制作一个 keyless provider fixture，分别在日志接纳前与请求构造后修改一条上下文，检查请求重建 invariant 的接受/拒绝行为；再测试硬退出发生于 live stream 与 durable settlement 之间的恢复语义。所需资源是 Node 与本地 fixture，不需要训练 GPU。本次只读源码，未执行。

## 下一步

- [ ] 追踪一个 shipped profile 如何加载 invariant companion。
- [ ] 跟踪一次 fork / compaction 后的 `deriveMessages()`。
- [ ] 对照 Pi 的同一失败场景，而不是比较二者 README 功能列表。
