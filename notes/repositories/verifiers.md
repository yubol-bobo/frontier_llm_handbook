# Verifiers：任务、harness、runtime 与评分怎样组成同一条 rollout？

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/PrimeIntellect-ai/verifiers @ `27bbd216df0af719a43705866b2cf6139bcc95de`  
验证范围：阅读本地固定快照的实现、相关配置与声明；未安装上游依赖，未启动模型、沙箱或训练。本文的控制流判断来自静态阅读，不代表复现了作者的性能或 benchmark。

## 核心问题

一个 agent benchmark 如何既能评价不同 harness，又能提供训练需要的 trace？当前固定快照要从 **v1 Task / Harness / Runtime / Rollout** 接口读起；只凭旧环境教程中的类名描述当前源码会遗漏实际控制边界。

## 源码路径与执行链路

- [Rollout 的 open / step / close 生命周期](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/rollout.py#L178-L540)；本地：[`verifiers/v1/rollout.py`](../../sources/verifiers/verifiers/v1/rollout.py)。
- [Task 评分 hook、runtime 依赖与 reward 记录](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/task.py#L198-L269)；本地：[`verifiers/v1/task.py`](../../sources/verifiers/verifiers/v1/task.py)。
- [Pi 的真实可选 harness 适配](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/harnesses/pi/harness.py#L19-L192)；本地：[`verifiers/v1/harnesses/pi/harness.py`](../../sources/verifiers/verifiers/v1/harnesses/pi/harness.py)。
- [Harbor CLI 与任务模型的可选集成](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/tasksets/harbor/taskset.py#L402-L535)；本地：[`verifiers/v1/tasksets/harbor/taskset.py`](../../sources/verifiers/verifiers/v1/tasksets/harbor/taskset.py)。

`Rollout.open()` 建立或借用 Runtime → 执行 Task.setup → Harness.setup → 启动 interception 与 tool servers → 将可达 endpoint/secret/MCP 地址交给 Harness session。Task 负责任务条件与评分规则，Harness 负责 agent 执行方式，Runtime 负责程序运行资源。它们能组合不意味着所有组合都兼容。

`Rollout.step()` 执行一个 harness segment，而非保证只做一次 LLM request。传入用户消息时走续接路径；超时、模型/工具错误与是否有新 turn 都影响能否继续。`close()` 关闭交互服务，成功打开且未 failed 时执行 finalize/artifact collection，再并发调用 task/harness 的 score，最后 cleanup 并按所有权关闭 Runtime。

`Task.score()` 找出 metric / reward hooks 和 plugged judges，按签名注入数据。无 runtime 时跳过要求必传 runtime 的评分项；分别记录指标与带权 reward，而不是把所有返回数值自动当单一训练 reward。

## 已确认的机制与取舍

- **实现事实：** borrowed runtime 不由 Rollout 关闭；生命周期所有权是接口契约。如果资源所有者提早 teardown，open 会报错。这直接影响并发复用沙箱的正确性。
- **实现事实：** agent timeout 被记录成失败；正常 stop 则可对 partial trajectory 评分。close 幂等，最终清理放在 finally，传输关闭失败不必然抹掉本来可评分的结果。这些差异会改变“训练有效样本”的定义。
- **实现事实：** 离线重评分不能凭空得到 runtime 所依赖的证据；`Task.score` 会跳过该类信号。要可重复评估，需在 finalize 中保存足够 artifact 或重建可验证环境。
- **实现事实：** Pi 适配器 pin npm package `@earendil-works/pi-coding-agent` 的 0.84.1，并通过 pi-acp、provider config 和 interception endpoint 连接；本次仅阅读了安装字符串，没有运行其中任何命令。
- **实现事实：** Harbor adapter 通过 CLI 下载、通过 Harbor Python task model 解析，并将导入延迟到真正加载 Harbor task 时。额外阅读该文件 1–93 行：Dockerfile-only 环境默认不直接构建，timeout/isolation 选项会改变评价条件。
- **阅读判断：** 环境可复用的关键是明确生命周期、证据和评分契约；统一入口不自动产生公平 benchmark，也不自动保证 reward 没有捷径。

## 与其他项目的连接

| 对象 | 关系类型 | 本次证据与边界 |
|---|---|---|
| Pi | 可选 harness 的真实集成 | `harnesses/pi/harness.py:38–41,47–48,84–192`；其固定 npm release 不等于本地独立 Pi HEAD。 |
| Harbor | 可选任务集依赖 | `tasksets/harbor/taskset.py:402–410,434–441,522–528`；CLI 与 Python API 都有代码证据。 |
| Prime RL | 被直接依赖 | 由 Prime RL 的 pyproject、submodule 和 `import verifiers.v1` 证明，详见 Prime RL 笔记。Verifiers 自身生命周期层不是 GPU optimizer。 |
| APEX recipe | 概念对应，经 Harbor 分别集成 | 两者都需要任务执行、评分和 token trace；没有把 APEX 当 Verifiers 的直接依赖。 |

## 动手实验

**待执行：同一 trace 的在线/离线评分差别。** 建一个仅在 trace 上检查字符串的 reward 与一个必须读取 runtime 文件的 reward；固定 task、输入和 harness 输出，比较 live runtime 下 `score` 与只保存 trace 后离线 `score`。观察哪些 reward/metric 被记录、哪些跳过、缺失证据是否显式可见。预期 runtime-only 项离线不运行，不能把缺失项默认为通过。实际结果：**未执行**。

第一阶段用合成 trace 与轻量 runtime，CPU 足够；第二阶段对同一 Harbor 小任务切换 Pi 与另一个 harness，固定模型、预算、任务镜像与 verifier，比较 trace 和 artifact。第二阶段涉及模型 API、Node 与 sandbox，尚未安装/调用。

## 下一步与疑问

- [ ] 追 interception/session 到 `Trace` 的 token 与 request rewrite 记录。
- [ ] 验证 Task/Taskset 层如何合并多个 reward，避免把日志 metric 当训练目标。
- [ ] 阅读 Harbor adapter 的独立 verifier 环境执行路径，并比较评分隔离开销。
