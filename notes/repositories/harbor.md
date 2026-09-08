# Harbor：一次 agent 行为怎样成为可信的 reward

日期：2026-09-08  
阶段：L1 源码初读  
源码：`https://github.com/harbor-framework/harbor` @ `9a2e3b135cc8fb1e41131020f83370cb2f12ae93`  
验证范围：实际阅读本地单步 trial、共享/独立 verifier 生命周期、reward 解析实现；未运行容器、agent、评测或 RL。已阅读上游 `AGENTS.md`，没有修改上游代码。

## 核心问题

环境运行失败、agent 超时、答案错误和 reward 文件损坏，是四种不同的问题。Harbor 怎样保存足够的证据，并把它们送到正确的结果路径？这比“最后执行一次测试”更接近 agent RL 的环境工程。

## 源码路径与执行链路

本地路径均相对于学习库根目录。以下链接固定到实际克隆的提交。

| 证据 | 本地实现 | 固定源码 |
|---|---|---|
| H1 | `sources/harbor/src/harbor/trial/single_step.py` | [SingleStepTrial._run 与 _run_agent，38–87 行](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/trial/single_step.py#L38-L87) |
| H2 | `sources/harbor/src/harbor/trial/trial.py` | [shared / separate verifier，641–744 行](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/trial/trial.py#L641-L744) |
| H3 | `sources/harbor/src/harbor/verifier/verifier.py` | [reward 解析与 verify，67–266 行](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/verifier/verifier.py#L67-L266) |
| H4 | `sources/harbor-cookbook/pyproject.toml` | [Cookbook 声明依赖 harbor，11 行](https://github.com/harbor-framework/harbor-cookbook/blob/e093c9a860b988d9d74901010ddddb9c7f124f92/pyproject.toml#L11) |

主链路：`SingleStepTrial._run()` → `_run_agent()` → 上传 agent 日志 → 收集 artifacts → `_run_verifier()` → `VerifierFactory` → `Verifier.verify()` → `VerifierResult(rewards=...)`。独立 verifier 模式先停 agent 环境，再创建验证环境并传入 artifacts；共享模式在同一环境验证，结束后才停 agent 环境。

`Verifier.verify()` 先定位并上传测试，再合并 verifier 环境变量，通过环境接口执行测试脚本；非 mounted 环境还要下载验证日志。它优先读取 `reward.json`，其次读取 `reward.txt`，两个都没有时抛 `RewardFileNotFoundError`。返回值来源是 reward 文件协议，而非把测试进程的退出码直接当作 reward。

## 已确认的机制与取舍

- **源码事实：agent 超时仍可验证。** `_run_agent()` 捕获 `AgentTimeoutError` 与 `NonZeroAgentExitCodeError`，记录异常并在 finally 同步输出。后续链路仍可收集部分产物、执行 verifier。其他未捕获异常不能由这一小段代码推断为同样处理。
- **源码事实：环境边界是配置的一部分。** shared 路径复用 agent 环境；separate 路径传递 artifacts，且 `skip_tests_upload=True`，测试由 verifier 镜像提供。两条路径都设置实际 verifier 用户、阶段网络策略及超时。
- **源码事实：奖励内容有输入校验。** 文本解析拒绝空文件和非有限数；JSON 解析检查数值类型与有限性，防止 NaN/Infinity 进入后续数据。该检查不是答案正确性的保证，正确性仍由任务测试定义。
- **源码事实：默认 verifier 保留奖励字典。** H3 末尾直接构造 `VerifierResult(rewards=rewards)`；`models/verifier/result.py:4–5` 也只是字典字段，没有自动把多个维度转换成名为 `reward` 的标量。训练端必须明确约定读取哪一项或如何聚合。仓库中另有 Rewardkit 聚合机制，不能把它误认为默认 verifier 的无条件行为。
- **阅读推断：** 独立验证环境使“被评价的运行环境”与“评价执行环境”边界更清楚，但代价是镜像、产物协议和传输成本；产物清单漏项会改变可验证的信息。不能仅凭存在 separate 模式就宣称所有任务均已抗 reward hacking。

## 与其他项目的连接

- **直接依赖，方向是 Cookbook → Harbor：** H4 的依赖声明已确认；不是 Harbor 依赖 Cookbook。[Cookbook 学习笔记](harbor-cookbook.md) 可作为任务创建入口。
- **示例集成有版本边界：** Cookbook 的 `harbor_cookbook/harbor_rl/train.py:3` 单独指定 `feature/harbor-rl-4d0` 分支。本次 Harbor main 没有该示例引用的 `harbor.rl`；因此不能用 main 的 verifier 源码证明分支上 `RLEnvironment.grade()` 的完整行为。
- **概念对应：** [Verifiers](verifiers.md) 同样把环境交互变成反馈；本次未审阅其与 Harbor 的完整桥接路径，不把概念对应画成直接依赖。
- **概念对应：** [SGLang](sglang.md) 提供生成执行与吞吐机制；Harbor 在本次追踪链路中承担任务/环境/验证。能处于同一训练系统，并不证明任意版本开箱即用。

## 动手实验

**状态：待执行。** 创建一个只修改文本产物的小任务，固定 instruction、镜像内容和 verifier。分别让正常 agent、超时前写出部分产物的 agent、生成非法 reward 的测试脚本执行；再对 shared / separate 两种模式比较。

输入：相同初始文件、相同产物路径与验证规则。控制变量：资源上限、agent 预算、随机种子/模型采样设置、任务版本。观测：agent exception、artifact 完整性、verifier exception、reward 字段、阶段耗时。预期：部分产物可被评分；非法 reward 被区分为解析错误，而非静默变成零分。实际结果：未执行，没有成功率数据。算力：Docker 与普通 CPU 即可测试脚本 agent；使用 LLM agent 时另需模型 API 或本地推理。

## 下一步与疑问

- [ ] 继续追 `trial/multi_step.py`，确认中间 reward、提前停止与最终结果聚合。
- [ ] 读 `trial/artifact_handler.py`，追踪缺失产物、过滤规则与传输失败语义。
- [ ] 从 Cookbook 最小 recipe 开始，先验证任务协议，再接 RL 框架。
