# Harbor Cookbook：任务、评分与 RL 接口如何连接

日期：2026-09-08  
阶段：L2 局部机制阅读；multi-reward 的 Python 评分部分完成 L3 局部 CPU 实验。  
源码：[harbor-framework/harbor-cookbook](https://github.com/harbor-framework/harbor-cookbook) @ `e093c9a860b988d9d74901010ddddb9c7f124f92`  
验证范围：阅读任务与训练适配示例；直接执行 upstream Python 断言。未运行 Harbor、Docker、pytest CLI、Tinker 或 RL。

## 核心问题

一个任务怎样变成训练信号？多个 verifier 输出的分数，是否天然就是 trainer 要的单个 reward？

## 源码阅读链路

| 入口 | 观察 | 固定版本 |
|---|---|---|
| [multi-reward/tests/test.sh](../../sources/harbor-cookbook/harbor_cookbook/recipes/multi-reward/tests/test.sh) | correctness / performance 独立产生二值奖励，输出 reward.json | [L10–27](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/recipes/multi-reward/tests/test.sh#L10-L27) |
| [test_correctness.py](../../sources/harbor-cookbook/harbor_cookbook/recipes/multi-reward/tests/test_correctness.py) | 邮箱忽略大小写、保留最大 id、排序等断言 | [L8–69](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/recipes/multi-reward/tests/test_correctness.py#L8-L69) |
| [multi-step/task.toml](../../sources/harbor-cookbook/harbor_cookbook/recipes/multi-step/task.toml) | 每步指令、前置健康检查、奖励门槛、产物保存 | [L31–98](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/recipes/multi-step/task.toml#L31-L98) |
| [harbor_rl/train.py](../../sources/harbor-cookbook/harbor_cookbook/harbor_rl/train.py) | Sandbox → RLEnvironment → tool / grade → Tinker Env | [L42–130](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/harbor_rl/train.py#L42-L130) |

`make_envs()` 为 group 中每个样本建立独立 sandbox 与 session，启动 BashTool，利用 renderer 生成初始消息；tool wrapper 调用 `env.step()`，reward wrapper 调用 `env.grade()`；结束时 cleanup 停止已创建环境。`group_size` 既影响训练采样，也直接影响环境并发成本。

## 真实接口边界

- `sky_rl/README.md` 明确写着本目录是 pointer，完整集成在外部 SkyRL 仓库。[固定入口](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/sky_rl/README.md#L1-L9)
- `prime_rl/README.md` 指向 Verifiers 的 `environments/opencode_harbor`，不能描述为 Cookbook 自带完整 Prime trainer。[固定入口](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/prime_rl/README.md#L1-L5)
- `harbor_rl/train.py` 的脚本依赖声明使用 Harbor 特定 feature branch 和外部 `tinker-cookbook`。该示例不等于本次独立 Harbor HEAD 已验证兼容。[L1–7](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/harbor_cookbook/harbor_rl/train.py#L1-L7)
- 当前 reward wrapper 读取 `result.rewards.get("reward", 0.0)`。把仅有 correctness/performance 的原始字典直接交给这一读取方式，会得到 0；实际环境是否归一化、应该如何聚合，需要在适配层明确验证，不能据此直接宣称运行中的 Harbor 有 bug。

## 已运行：一个错误实现也能获得性能分

见 [实验说明](../../experiments/001-harbor-reward-contract/README.md)、[可重跑脚本](../../experiments/001-harbor-reward-contract/run.py) 和 [原始结果](../../experiments/001-harbor-reward-contract/results.json)。

| 候选 | correctness | performance |
|---|---:|---:|
| upstream reference solution | 1 | 1 |
| 仅移除邮箱 `.lower()` 的错误版本 | 0 | 1 |

实验实际加载了固定版本的 reference Python 函数与测试断言；没有启动 upstream `test.sh` 中的网络安装流程。结果说明此性能用例没有覆盖大小写正确性。若训练仅优化性能字段，错误行为也能得到奖励。如何将奖励向量汇总成标量，必须明确产品目标，例如先满足 correctness，再奖励性能；具体权重属于另一个需要评测的设计决策。

## 与其他项目的连接

- **直接使用：Harbor。** Cookbook 的任务格式与 RL 环境接口由 Harbor 提供。
- **文档指向的集成：SkyRL、Verifiers / Prime RL。** 需继续进入目标仓库源码，不能把链接本身当端到端复现证据。
- **概念对应：APEX recipe。** 两者都将 sandbox 中的任务产物交给 verifier，再把结果关联到 rollout；APEX 的数据与版本依赖另有约束。
- **概念对应：Open Instruct / Verifiers。** 有奖励输出不保证优化信号可靠，仍要考察格式、缺失值、失败类型和最终指标。

## 下一步

- [ ] 在真正 Harbor Trial 中运行该任务，验证 JSON 奖励的读取与归一化，不只执行 Python 断言。
- [ ] 固定 scalarization 后，检查错误候选与 reference 的训练奖励排序。
- [ ] 给一个现实任务写两个独立 verifier，并保留未参与训练的 held-out cases。
