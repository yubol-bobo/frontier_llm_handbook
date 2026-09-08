# Miles：MoE rollout 的离散路由怎样在训练中重放？

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/radixark/miles @ `3de96596f16b9e6d23ba550c4c47de3479c9f14c`  
验证范围：阅读本地固定快照的实现、相关配置与声明；未安装上游依赖，未启动模型、沙箱或训练。本文的控制流判断来自静态阅读，不代表复现了作者的性能或 benchmark。

## 核心问题

同样的模型权重、token 和温度是否足以让 rollout 与 trainer 计算同一条 MoE 路径？本次集中阅读 routing replay，而不是泛读 Miles 的全部优化。关键问题是离散 expert index 如何跨 microbatch、并行分片和重计算保持对齐。

## 源码路径与执行链路

- [Replay 缓存、阶段与 top-k 包装](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/utils/replay_base.py#L14-L241)；本地：[`miles/utils/replay_base.py`](../../sources/miles/miles/utils/replay_base.py)。
- [把 rollout replay 张量对齐到训练 microbatch / CP / SP](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/training_utils/replay_data.py#L30-L134)；本地：[`miles/backends/training_utils/replay_data.py`](../../sources/miles/miles/backends/training_utils/replay_data.py)。
- [训练 actor 中的 replay 状态切换](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/megatron_utils/actor.py#L490-L621)；本地：[`miles/backends/megatron_utils/actor.py`](../../sources/miles/miles/backends/megatron_utils/actor.py)。
- [Megatron MoE 层布局映射](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/megatron_utils/replay_utils.py#L1-L24)；本地：[`miles/backends/megatron_utils/replay_utils.py`](../../sources/miles/miles/backends/megatron_utils/replay_utils.py)。

`train_actor()` 先创建用于 logprob 与 training 的 data iterator；启用 rollout replay 时，对各 replay manager 调用 `fill_replay_data()`。该函数要求每条样本提供 `[num_tokens - 1, num_streams, topk]` 张量，以和训练相同的 microbatch 次序重放 iterator，再补最后一个 loss-masked token，对齐 CP、SP 和 padding，最后映射到每个模型层的 Replay queue。

`register_replay_list_moe()` 使用 Megatron 的 layer offset 与每个 virtual pipeline stage 的本地层数量，跳过 dense layer，只给本 rank 的 MoE 层装载正确 stream。`Replay.record()` 将 index 放到 pinned CPU buffer；forward/backward 各有独立读指针，需要时搬回当前 GPU。

在 actor 中，reference/teacher 计算阶段是 fallthrough；旧 actor/logprob 阶段根据配置选择 record 或 replay_forward；随后清 forward 游标，训练阶段设为 replay_backward，完成后清空队列。这里只确认调度和 replay wrapper 实现，未逐层验证所有模型 patch 的安装。

## 已确认的机制与取舍

- **实现事实：** top-k wrapper 重放的是选中的 index；`return_probs=True` 时仍从当前 `scores.gather()` 取对应 score，不是冻结整份概率或梯度。这是“控制离散选择”而非“复用旧 logits”。
- **实现事实：** padding 用 -1 标记，并在实际 gather 前将全无效行替换成合法 index；shape 断言检查 token 数与 topk。索引正确还依赖 CP/SP 分片和 microbatch 次序，不能只校验总长度。
- **实现事实：** `IndexerReplayManager` 与 `RoutingReplayManager` 共用机制，但 indexer 的索引指 token/KV 位置，需要在 packed 序列中 rebase。MoE expert id 不应套用 token position 偏移。
- **实现事实：** overlap checker 可比较重算 top-k 与 replay top-k；它有阈值与开关。默认有 checker 接口不等于我们实测了零 mismatch。
- **阅读判断：** pinned CPU 缓存节约 GPU 常驻空间，同时引入搬运/同步成本。replay 能约束一部分 train–infer 不一致，但不能单独保证所有低精度 kernel、softmax、weight version 都一致，更不能据此宣称整个训练“true on-policy”。

## 与其他项目的连接

| 对象 | 关系类型 | 本次证据与边界 |
|---|---|---|
| slime | 项目谱系 | 本地 `README.md:109–112` 明确声明 fork 自 slime；不等于当前两者文件/API 仍可直接互换。固定链接见关系笔记。 |
| Megatron-LM | 训练后端的直接依赖 | `replay_utils.py:1–2` 真实导入 Core 的层布局函数，不只是 README 提名。 |
| SGLang | 直接推理集成 | 本地 `miles/backends/sglang_utils/sglang_engine.py:8,68` 导入 ServerArgs 并构造 `python -m sglang.launch_server`；本次只定位该连接，未跟踪服务内部。 |
| DeepEP / DeepGEMM | 概念对应，待进一步追依赖 | 此次阅读的是 routing 选择重放；通信/矩阵计算 kernel 是下游不同层。不能据模型都叫 MoE 就画成本仓库直接依赖。 |

## 动手实验

**待执行：路由记录与重放的对齐检查。** 第一阶段读现有 `tests/fast/utils/test_replay_base.py` 与 `tests/fast/backends/training_utils/test_replay_data.py`，使用合成 top-k/score、不同长度样本和 padding 验证指针与分片。保持 scores/token 固定，只改变 replay index、batch packing 和 CP/SP 配置；观测 shape、layer stream、有效 token index、overlap 与梯度对应位置。

先做 host-side 数据排布验证；实际 `Replay` 使用 pinned memory 与 `torch.cuda.current_device()`，完整 wrapper 路径需要匹配的 PyTorch/CUDA，不能笼统说全链 CPU 可跑。第二阶段在小 MoE 上比较 replay 开/关的 logprob 差异、路由 mismatch、显存和每步耗时。实际结果：**均未执行**；不提供速度提升预估。

## 下一步与疑问

- [ ] 从 model provider 追至 router patch 的安装位置，验证每个支持模型的 hook。
- [ ] 追 SGLang 如何导出 `rollout_routed_experts`，核实 token shift 与 mask。
- [ ] 单独研究 weight version 和 precision；不能将 replay 一个机制等同于完整数值一致性。
