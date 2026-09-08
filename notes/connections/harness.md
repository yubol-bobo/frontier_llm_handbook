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
