# Prime RL：异步 rollout 为什么还需要严格的入队、出队与版本边界？

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/PrimeIntellect-ai/prime-rl @ `04a61d3b75c3c99f263b2c133e822f998909adf7`  
验证范围：阅读本地固定快照的实现、相关配置与声明；未安装上游依赖，未启动模型、沙箱或训练。本文的控制流判断来自静态阅读，不代表复现了作者的性能或 benchmark。

## 核心问题

完全异步并不是“生成越多越好”。已经完成、仍在队列里等待训练的 rollout 会继续变旧；被取消的请求也必须进入组完成计数。本次追 `Episode → group → TrainingSample → microbatch → sender`，重点看吞吐控制怎样影响实际训练分布。

## 源码路径与执行链路

- [结果分流、版本门控、packing 与发送](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/orchestrator.py#L513-L685)；本地：[`src/prime_rl/orchestrator/orchestrator.py`](../../sources/prime-rl/src/prime_rl/orchestrator/orchestrator.py)。
- [组完成计数、陈旧样本清理与样本转换](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/train_sink.py#L131-L325)；本地：[`src/prime_rl/orchestrator/train_sink.py`](../../sources/prime-rl/src/prime_rl/orchestrator/train_sink.py)。
- [本快照 GRPO credit 的具体实现](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/algo/grpo.py#L16-L45)；本地：[`src/prime_rl/orchestrator/algo/grpo.py`](../../sources/prime-rl/src/prime_rl/orchestrator/algo/grpo.py)。
- [Verifiers 本地依赖及 TorchTitan pin](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/pyproject.toml#L20-L250)；本地：[`pyproject.toml`](../../sources/prime-rl/pyproject.toml)。

额外实际阅读 `src/prime_rl/orchestrator/train_source.py:17–85`：按环境 ratio 选任务，为每个环境建立 curriculum，并保存 RNG 与各 curriculum 状态。调度器交回 Episode、GroupCancellation 或 DispatchFailure，`Orchestrator.main_loop()` 显式区分这些事件；真正的 Episode 在到达时先写入 `all` artifact，再交 train/eval sink。

`TrainSink.add()` 先执行 rollout-local algorithm work，然后将 Episode 按 group 收集。group 的完成数是已到 Episode + failure + cancellation count，避免等待永远不会回来的成员。组完成后执行 `finalize_group()` / curriculum admission，调用 `trace_to_samples()`，设置 temperature 与 loss routing，按 trace 放入 pending batch。

每次切 batch 前 `_drop_stale()` 清理过旧数据；组刚入队也再检查一次。`finalize_train_batch()` 等待 inference 至少应用指定 policy version，再 pack、`sender.send()`、推进 step。这里观察到发送接口，不宣称已审计 trainer 接收、梯度和广播全链。

## 已确认的机制与取舍

- **实现事实：** 若正在收集 step 的 batch、训练 policy v(step−1)，来自 v(k) 的样本陈旧度按 `(step−1)−k` 解释。dispatcher 提前取消只是省计算；sink 清理 queued trace 才是训练数据陈旧度的最后边界。frozen-source 无 live-policy span，不按此规则变旧。
- **实现事实：** stale cancellation 会使同组已回来的 Episode 一起失效，并跳过 curriculum：pipeline 过期事件不等于任务答错。这保留了环境难度统计与系统事件的区分。
- **实现事实：** top-p/top-k 截断采样需要 sampling mask；缺少时 `process_group():286–293` 抛错，因为 rollout logprob 的归一化概率空间与 trainer 必须相符。不能把相同 token ID 视作概率已经一致。
- **实现事实：** 此快照 `GRPOAlgorithm.score_group()` 默认是 `reward - group mean`，并未在该函数除组标准差；可选 length shaping 也显式计算。这与 verl 默认 GRPO 的 std 归一化不同，做算法对比要核对实现和配置。
- **阅读判断：** 丢弃过旧数据、拒收无训练信号的数据和限定发送版本会降低表面 rollout 利用率，但维护训练语义；好指标应同时看 all/effective、stale drops、有效 tokens 与 wall time。

## 与其他项目的连接

| 对象 | 关系类型 | 本次证据与边界 |
|---|---|---|
| Verifiers | 直接依赖 + Git submodule | `pyproject.toml:30,244` 和实际 `import verifiers.v1`；`.gitmodules` 指向官方仓库。 |
| TorchTitan | 平台限定直接依赖/组件复用 | `pyproject.toml:60` Linux dependency、250 的 git pin；另定位 `trainer/utils.py:16` 导入其 `clip_grad_norm_`。不是“整个 Prime trainer 建在 TorchTitan trainer 上”的证明。 |
| Harbor | Verifiers 路径中的可选任务集能力 | `verifiers[harbor]` 是明确声明；具体任务适配见 Verifiers 笔记，不能替代版本验证。 |
| verl / slime / Miles | 概念对应 | 都涉及 rollout logprob、mask、policy update；本次没有把它们视作 Prime RL 必需依赖。 |

**复现边界：** `git ls-files --stage deps/verifiers` 得到 gitlink `828488fffe31aa3332b9d1bd4bd9ee320e375cf1`，而本学习库独立 clone 的 Verifiers 是 `27bbd216df0af719a43705866b2cf6139bcc95de`。两个快照不是已经测试的组合。此次 shallow clone 没初始化 submodules，不能直接执行仓库 example 就假设依赖齐全。

## 动手实验

**待执行：完成后变旧的样本。** 构造两个相同 reward/token 的 rollout group，第一组立即返回，第二组延迟返回或先入队后推迟训练；控制 group size、curriculum、采样模型，仅改变版本推进和延迟。观测 stale drop、pending token 计数、group cancellation 计数、effective cohort 与发送 step。预期已入队样本也能过期；stale 事件不进入 curriculum 的任务结果。实际结果：**未执行**。

先用匹配依赖的合成 Episode 测试状态机，无需模型推理；完整闭环实验另用 Linux/CUDA 小模型，官方基础例子通常按 trainer/inference 两张 GPU 分配，不能把本次 clone 视作单卡训练环境已就绪。

## 下一步与疑问

- [ ] 追 `trace_to_samples()` 到 token loss routing，确认子 trace 及 sampling mask 转换。
- [ ] 追 sender → trainer → weight watcher，核实版本原子性和 checkpoint 恢复边界。
- [ ] 复现实验前先初始化作者锁定的 submodules，而非替换成旁边独立 clone 的 HEAD。
