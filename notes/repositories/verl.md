# verl：一个 rollout batch 如何变成正确分组的策略更新？

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/verl-project/verl @ `7cb65014d3a6c84f59458367df768999e4f36c67`  
验证范围：阅读本地固定快照的实现、相关配置与声明；未安装上游依赖，未启动模型、沙箱或训练。本文的控制流判断来自静态阅读，不代表复现了作者的性能或 benchmark。

## 核心问题

“支持 GRPO”隐藏了两类工作：同一道题的多个回答如何保持正确分组，以及分布式 rollout、重算 logprob、训练和权重同步如何组成闭环。首读沿 `RayPPOTrainer.fit()` 追踪一个 batch，不把所有 backend 同时展开。

## 源码路径与执行链路

- [生成、分组、优势计算与权重同步](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/ray_trainer.py#L1481-L1716)；本地：[`verl/trainer/ppo/ray_trainer.py`](../../sources/verl/verl/trainer/ppo/ray_trainer.py)。
- [GRPO outcome advantage 的实际数学实现](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/core_algos.py#L267-L331)；本地：[`verl/trainer/ppo/core_algos.py`](../../sources/verl/verl/trainer/ppo/core_algos.py)。
- [Megatron engine 的真实导入与接口边界](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/workers/engine/megatron/transformer_impl.py#L15-L84)；本地：[`verl/workers/engine/megatron/transformer_impl.py`](../../sources/verl/verl/workers/engine/megatron/transformer_impl.py)。
- [SGLang / vLLM / mcore 的可选依赖声明](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/setup.py#L50-L77)；本地：[`setup.py`](../../sources/verl/setup.py)。

1. `fit()` 在训练前恢复 checkpoint 并同步 rollout 权重（1405–1430）。每个原始 prompt 分配 `uid`，随后按 `rollout.n` 做 interleaved repeat（1481–1491）。
2. `async_rollout_manager.generate_sequences()` 返回结果；随后让 inference replicas sleep，把结果与同样 repeat 的训练 batch 合并（1513–1544）。这里的 async 是调用的 rollout 管理器特征，不能仅凭名字把此 `fit()` 宣称为完全无屏障流水线。
3. 可选 token 数平衡重排 batch；group membership 依赖 `uid`，不依赖行位置。reward、旧策略 logprob、reference logprob、critic values 按配置加入 `DataProto`。
4. `compute_advantage()` 在 driver 执行，GRPO 分支把 `uid` 传给 `core_algos.compute_grpo_outcome_advantage()`。后者把逐 token reward 求和成 response score，按 uid 求组均值/标准差，再把同一个 response advantage 广播回有效 response token。
5. 完成可选 critic update 后更新 actor；checkpoint 与训练到推理权重同步是显式步骤（1662–1716）。Megatron engine 通过其自己的实现导入 Megatron Core 的 parallel state、pipeline forward/backward 等能力。

## 已确认的机制与取舍

- **实现事实：** 同组优势不是“本 batch 减一个总体均值”。它按 `uid` 聚合；GRPO 的 `norm_adv_by_std_in_grpo` 决定是否除组内标准差。单元素组特别处理为 mean=0、std=1，不能用正常多样本组的直觉解释它。
- **实现事实：** `_balance_batch` 附近明确说明：重排不影响按 uid 计算 advantage，但可能通过 minibatch 构成影响 loss。数据顺序与学习行为并非完全无关。
- **实现事实：** 重算旧 logprob 与直接使用 rollout logprob 是分支选择。decoupled 模式可先做 importance sampling / rejection correction，bypass 路径的指标比较对象不同（1647–1660）。存在该机制不等于任意精度/推理后端组合已正确校正。
- **阅读判断：** 本仓库最值得学的是 control flow 与 compute engine 的边界。先追清 batch 里每个字段的来源，比先记算法名更能理解系统正确性。

## 与其他项目的连接

| 对象 | 关系类型 | 本次证据与边界 |
|---|---|---|
| Megatron-LM | 可选训练后端的直接代码依赖 | `transformer_impl.py:23–25` 导入 `megatron.core`；不表示每个 verl 作业都会用 Megatron。 |
| SGLang | 可选 rollout 后端 | `setup.py:57–61,73` 声明 SGLang extra；该快照 pin 为 0.5.8，不能直接拿独立克隆的最新 SGLang 替换而声称兼容。 |
| vLLM | 可选 rollout 后端，外部项目 | `setup.py:55,72`；不在本次 19 库独立 clone 范围。 |
| prime-rl / slime | 概念对应 | 都处理采样数据到策略更新的边界；这里未发现需要把它们画成 verl 的直接依赖。 |

## 动手实验

**待执行：组标识与排序实验。** 输入三个 prompt，每题四个手工 response score，以及带 padding 的 response mask。先固定 uid 并打乱行序，再故意使用错误 uid，比较还原排序后的 advantage。控制 reward、组大小、mask，仅改变排列与标识；观测 per-response advantage、组均值和 padding 位置。预期正确 uid 下重排不改同一 response 的优势，错误 uid 会改变比较组；这是待验证假设。先用 CPU + 匹配版本 PyTorch/NumPy 跑实际函数，不需要模型/GPU。本次实际结果：**未执行**。

后续 GPU 实验再比较开启 batch balancing 前后的每 rank 有效 token、step time 与 minibatch loss，不把 CPU 数学实验外推为系统吞吐结论。

## 下一步与疑问

- [ ] 阅读 `rollout_corr_helper.py`，把 rollout / old / current 三种策略概率的定义画清楚。
- [ ] 从 `_update_actor()` 追至选定 engine 的 `train_batch`，核实 mask、temperature 和 loss denominator。
- [ ] 为一个选定 recipe 建独立环境，遵守其版本约束；当前只完成 L1。
