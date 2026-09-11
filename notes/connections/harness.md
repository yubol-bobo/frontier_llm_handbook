# Harness 与训练轨迹之间的连接

日期：2026-09-08。证据来自本地固定源码；除实验 001 的 Python 评分外，以下集成未执行。

## 两种请求正确性

[Pi](../repositories/pi.md) 在 agent message 层允许 context transform，再转换为 provider messages。[DeepSeek Harness](../repositories/deepseek-harness.md) 则把 loop-built request 与 session log 的重建结果进行比较。两者围绕相同问题提供不同接口：应用内部状态如何变成模型输入。这是概念对应，没有依赖关系证据。

RL 层还需要保存实际采样 token、概率、角色 mask 和模型版本。[slime](../repositories/slime.md) 与 [Miles](../repositories/miles.md) 的轨迹处理是下一层问题。即使一份 session 文本完全可读，也不能仅凭重新 tokenize 后的文本证明梯度对应真实采样。

## 已有 adapter 的连接

| 连接 | 类型 | 证据 |
|---|---|---|
| Verifiers → Pi | 可选 harness adapter | [Pi harness 实现](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/harnesses/pi/harness.py#L19-L192)；npm release 和 ACP/provider 设置都在这里 |
| Verifiers → Harbor | 可选 taskset adapter | [Harbor taskset](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/tasksets/harbor/taskset.py#L402-L535) |
| Cookbook → SkyRL | 文档 pointer | [sky_rl/README](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/sky_rl/README.md#L1-L9)；完整代码在范围外仓库 |
| Cookbook → Prime RL / Verifiers | 文档 pointer | [prime_rl/README](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/prime_rl/README.md#L1-L5)；目标路径可能随上游版本演变，应读目标快照 |
| Cookbook harbor_rl → Harbor feature branch / Tinker | 具体示例调用 | [依赖声明](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/harbor_rl/train.py#L1-L7) 和 [调用链](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/harbor_rl/train.py#L42-L130) |

上述关系提供一条学习路线：固定任务、模型和预算，对比 harness；再观察训练框架消费了哪些 trace 字段。但独立 HEAD、npm release、feature branch 不同，尚未形成一套经过运行验证的组合。

## 评分连接的一个已验证问题

[实验 001](../../experiments/001-harbor-reward-contract/README.md) 说明：同一候选可以 correctness=0、performance=1。Harbor 主分支默认 verifier 保留奖励字典，[解析路径](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/verifier/verifier.py#L255-L266) 不为任意字典自动选择训练目标。Cookbook 中特定 feature-branch 示例的 `grade()` 行为尚未在本地读取，因此不能把 raw dict 的观察扩展为该示例完整运行结果。

可复用问题是：**task 写出了什么，verifier 解析了什么，adapter 传递了什么，trainer 最后优化了什么？** 这四项需要分别留下证据。

## Claude Code：从回调协议到训练数据契约

[固定 SDK 笔记](../repositories/claude-agent-sdk.md) 展示 Python 回调如何通过控制通道返回 CLI；不能把 SDK 的 allow/deny 序列化当成整个 sandbox 的实现证明。[进阶讲解](../../handbook/05-claude-code-harness.md) 把它与 Pi 的执行事件和 DeepSeek 的日志投影并读。

[CPU 实验](../../experiments/harness-state-machine/README.md) 的未知结果场景说明：工具可能已经产生副作用，而日志尚无完成记录。自动重试可能重复动作；需要幂等键、外部核对或明确的人工处置。日志恢复、任务成功评分与 RL token 对齐分别验收。

## SoL-Pi → Pi：直接扩展；其他框架：机制对照

[SoL-Pi 源码笔记](../repositories/sol-pi.md) 追踪对 Pi 公开工具定义、事件和 native compaction 的直接调用。开发版本 0.84.2 与独立 Pi 快照 0.85.1 需要区分。与 Claude Code / DeepSeek 的连接是机制比较，不是直接依赖；与 Harbor/APEX 的连接是独立评分与训练轨迹的设计问题，未接成可运行栈。

[学习专题](../../handbook/06-sol-pi-efficient-harnesses.md) 把工具往返、输出重放、证据完整性、缓存重建和异步继续放到同一条执行链中。必须分别验证原始证据可取回、模型实际看到了什么以及最终任务是否完成。
