# Megatron-LM：MoE 的 router、expert compute 与通信后端如何接起来

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/NVIDIA/Megatron-LM @ `c9b53d0a87cb926f47115259593ecfeb351ca29f`  
验证范围：本地 MoE forward、TopKRouter、Flex/DeepEP dispatcher 与 fused_a2a 的 Buffer 接口；未构建 CUDA 扩展、未安装 DeepEP、未启动训练或 benchmark。

## 核心问题

一个 token 只去少数 experts，看似节省 FLOPs，为什么反而需要复杂的 metadata、permutation、all-to-all 和通信缓冲区？router 的算法选择怎样影响系统行为？

## 源码路径与执行链路

1. [moe_layer.py](../../sources/megatron-lm/megatron/core/transformer/moe/moe_layer.py) 的 [`MoELayer.forward`，L637–742](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/moe_layer.py#L637-L742)：shared experts / route → preprocess → dispatch → routed_experts_compute → combine → postprocess。还读了同文件 L466–618 对这些子步骤的实现，确认不是只从 docstring 猜流程。
2. [router.py](../../sources/megatron-lm/megatron/core/transformer/moe/router.py) 的 [`TopKRouter.routing`，L750–841](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/router.py#L750-L841)：logits 展平 → z-loss → top-k / 可选 balancing → capacity dropping → attach auxiliary loss → expert bias bookkeeping，输出 probs 与 routing_map。
3. [token_dispatcher.py](../../sources/megatron-lm/megatron/core/transformer/moe/token_dispatcher.py) 的 [`_DeepepManager`，L1225–1335](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/token_dispatcher.py#L1225-L1335)：将 token×experts 概率转换为 top-k indices/probs，丢弃位置标 -1，调用 fused_dispatch，保留供 combine 使用的 handle。另读 L1815–1866，确认 Flex dispatcher 显式选择 deepep / hybridep / ncclep。
4. [fused_a2a.py](../../sources/megatron-lm/megatron/core/transformer/moe/fused_a2a.py) 的 [L11–95](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/fused_a2a.py#L11-L95)：guarded import `deep_ep.Buffer`，按 group 与 hidden bytes 获取 dispatch/combine buffer 需求，再建立或复用缓冲区，FusedDispatch 为自定义 autograd Function。此轮未完整读它的 backward。

## 已确认的机制与取舍

**router 输出是数据布局计划。** 输入逻辑 shape 为 sequence×batch×hidden；router 将 logits 转为 tokens×experts，返回稀疏选择对应的概率/布尔映射。expert 不是对完整 batch 各跑一遍：dispatcher 把 tokens 按 destination rank 与 local expert 整理，expert compute 获取 `tokens_per_expert`，combine 需要恢复原 token 位置。局部 GEMM 的高效与通信布局是同一个问题的两面。

**capacity 是算法和系统共同的开关。** 配置 capacity factor 后调用 token dropping，可配置 drop policy / pad to capacity；这能改变固定形状和负载，但也会改变哪些 token 被处理。负载均衡 aux loss 分别有 batch、sequence、global 路径，attach 在 probs 的 autograd 链上。不能只将 aux loss 当成一个无关的记录指标，也不能把 balancing、dropping、padding 当成同义词。

**EP / TP 组合存在具体约束。** 训练中 attention TP group>1 且 sequence_parallel=False，MoELayer.forward 直接报错；源码理由是性能可能退化。Flex DeepEP 路径要求 TP×EP>1，并将组、top-k、专家数映射进 manager。此处约束不是理论上 MoE 必须多卡，而是选定训练/dispatch 路径的实现要求。

**DeepEP 是可选后端，不是所有 MoE 的硬依赖。** 构造器按 `moe_token_dispatcher_type` 选择 allgather/alltoall/flex；Flex 再选通信 manager。`fused_a2a.py` 的 import guard 与 `_DeepepManager` 的 ImportError 说明只有启用这条路径才要求安装 DeepEP。真正的 bridge 包含 dtype 转换：DeepEP probabilities 需 FP32，BF16/FP16 会警告并转换。

**通信状态不可丢。** manager 保留 `handle`、dispatched indices/probs、tokens_per_expert；这些元数据用于后续恢复和通信配对。buffer 根据 group、NVLink/RDMA size hints 判断能否复用；不是每次临时分配即可获得同样性能。`async_finish` / comm stream 参数要求进一步审计 event 生命周期，不能根据 async 名字宣称计算通信已经充分重叠。

**一个 forward 还能被分段捕获或重算。** `fwd_execution_map` 和 intermediate_tensors 允许 route/expert/postprocess 分段；MoE recompute 在 FP8/FP4 下走 TE checkpoint，其他情况走 tensor_parallel checkpoint。读取主链时必须分清正常完整 forward 与 CUDA graph partial capture，不应把提前返回误读成漏掉 combine。

## 与其他项目的连接

| 关系 | 证据与边界 |
|---|---|
| DeepEP：明确可选后端 | `fused_a2a.py:11–17` 导入其 Buffer；`token_dispatcher.py:1835–1843` 选择 manager。可以沿这条边去读 [DeepEP 笔记](deepep.md)，但独立 clone HEAD 的 API/构建兼容尚未验证。 |
| slime / verl / miles：需由各自入口确认的后端关系 | 它们可能使用 Megatron 训练组件；本笔记不凭项目名替调用方作证。见相应 repo 笔记和 [训练连接](../connections/training.md) 的关系标注。 |
| TorchTitan：概念对应 | 对比张量/专家布局、activation checkpointing、通信与计算边界；不是直接依赖。 |
| DeepGEMM：概念对应 | grouped expert GEMM 与 token dispatch 的数据布局相邻；本次读取的代码不足以证明本 snapshot 的此条 expert compute 路径调用 DeepGEMM，不能画成已接入的依赖。 |

## 动手实验

**状态：待执行。** 将数值正确性和性能分别验证。

- 问题：相同 routing indices/probs，通信后端是否保留 token→expert→token 映射与梯度；不同负载形态如何影响吞吐？
- 输入：微型 MoE、固定模型权重与输入；两类 routing：均匀分布、集中到少数 experts；初始关闭 token dropping / recompute / CUDA graphs。
- 控制变量：专家数、top-k、dtype、TP/EP 布局、有效 token 数和初始化；只切换支持的 dispatcher backend。
- 指标：output/gradient 数值差异、token counts、丢弃率、峰值显存、dispatch/combine 时间、expert compute 时间、rank 间负载偏差。
- 预期：语义相同的路径应在预设数值容差内一致；负载倾斜可能使通信或少数 expert 成为瓶颈。没有证据保证 fused backend 在所有小 batch 都更快。
- 算力：路由逻辑可先用 CPU 张量级案例；真正 DeepEP 多卡测试需满足其 CUDA、互连、buffer 等要求。当前 Windows 工作区只是源码库，尚未准备该运行环境。
- 实际结果：无；没有把 README benchmark 当成此机器的测量。

## 下一步与疑问

- [ ] 继续 `FusedDispatch.backward` / `FusedCombine`，解释 forward dispatch 为什么与 backward combine 配对。
- [ ] 追踪 `experts.py` 的 grouped MLP backend 选择，区分 TransformerEngine、其他 GEMM 路径与可选优化。
- [ ] 对照一个真实 Megatron RL caller，核对所 pin commit、模型 state dict、权重同步及训练/推理概率一致性。
