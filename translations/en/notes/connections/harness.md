<a id="harness-与训练轨迹之间的连接"></a>

# Connections between harnesses and training trajectories

Date: 2026-09-08. Evidence comes from pinned local source; apart from the Python grading in experiment 001, the following integrations were not executed.

<a id="两种请求正确性"></a>

## Two kinds of request correctness

[Pi](../repositories/pi.md) allows context transformation at the agent-message layer before converting to provider messages. [DeepSeek Harness](../repositories/deepseek-harness.md) compares a loop-built request with its reconstruction from the session log. The two offer different interfaces around the same question: how does internal application state become model input? This is conceptual correspondence, without evidence of a dependency.

The RL layer additionally needs actual sampled tokens, probabilities, role masks, and model versions. Trajectory processing in [slime](../repositories/slime.md) and [Miles](../repositories/miles.md) addresses the next layer. Even perfectly readable session text cannot establish that gradients correspond to actual sampling merely by being retokenized.

<a id="已有-adapter-的连接"></a>

## Connections with existing adapters

| Connection | Type | Evidence |
|---|---|---|
| Verifiers → Pi | Optional harness adapter | [Pi harness implementation](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/harnesses/pi/harness.py#L19-L192); npm release and ACP/provider settings are both here |
| Verifiers → Harbor | Optional taskset adapter | [Harbor taskset](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/tasksets/harbor/taskset.py#L402-L535) |
| Cookbook → SkyRL | Documentation pointer | [sky_rl/README](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/sky_rl/README.md#L1-L9); full code is in a repository outside this scope |
| Cookbook → Prime RL / Verifiers | Documentation pointer | [prime_rl/README](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/prime_rl/README.md#L1-L5); target paths may evolve with upstream versions, so read the target snapshot |
| Cookbook harbor_rl → Harbor feature branch / Tinker | Concrete example calls | [Dependency declaration](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/harbor_rl/train.py#L1-L7) and [call chain](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/harbor_rl/train.py#L42-L130) |

These relationships provide a learning route: fix task, model, and budget to compare harnesses, then observe which trace fields the training framework consumes. However, independent HEADs, npm releases, and feature branches differ; they have not yet formed a combination verified through execution.

<a id="评分连接的一个已验证问题"></a>

## One verified issue in the grading connection

[Experiment 001](../../experiments/001-harbor-reward-contract/README.md) shows that the same candidate can have correctness=0 and performance=1. Harbor main's default verifier preserves a reward dictionary; the [parsing path](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/verifier/verifier.py#L255-L266) does not automatically select a training objective for arbitrary dictionaries. The `grade()` behavior in Cookbook's specific feature-branch example has not been read locally, so the raw-dictionary observation cannot be extended to a complete runtime result for that example.

The reusable question is: **what did the task write, what did the verifier parse, what did the adapter pass, and what did the trainer ultimately optimize?** Each of these four needs its own evidence.

<a id="claude-code从回调协议到训练数据契约"></a>

## Claude Code: from callback protocols to training-data contracts

The [pinned SDK note](../repositories/claude-agent-sdk.md) shows how Python callbacks return responses to the CLI over a control channel; SDK allow/deny serialization does not establish the implementation of the entire sandbox. The [advanced guide](../../handbook/05-claude-code-harness.md) compares this with Pi execution events and DeepSeek log projection.

The unknown-outcome case in the [CPU experiment](../../experiments/harness-state-machine/README.md) shows that a tool may have produced a side effect before its completion was journaled. Automatic retry may duplicate the action; idempotency keys, external reconciliation, or explicit human handling are needed. Validate log recovery, task-success scoring, and RL token alignment separately.
