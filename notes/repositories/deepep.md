# DeepEP：MoE token 路由为什么需要 handle、stream 与带宽模型

日期：2026-09-08  
阶段：L1 源码初读  
源码：`https://github.com/deepseek-ai/DeepEP` @ `01dc3aaac82068020353dce2c302e38153c0bfaa`  
验证范围：实际阅读 EPv2 Python buffer、SM/QP 估计、dispatch/combine、事件等待、NCCL communicator 生命周期及测试片段；未编译、未运行 GPU/RDMA。当前重点是 V2，不能套用 V1 的接口描述。

## 核心问题

一个 token 被送给多个 expert 后，返回结果怎样对应到原 token？缓存路由布局、避免 CPU 同步、占用较少 SM 与数值/确定性之间是什么关系？

## 源码路径与执行链路

本地路径相对于学习库根目录。

| 证据 | 本地实现 | 固定源码 |
|---|---|---|
| E1 | `sources/deepep/deep_ep/buffers/elastic.py` | [带宽模型估计 SM，729–834 行](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L729-L834) |
| E2 | 同上 | [dispatch 与 handle/event 构造，855–1033 行](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L855-L1033) |
| E3 | 同上 | [combine 使用路由元数据，1046–1107 行](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L1046-L1107) |
| E4 | `sources/deepep/deep_ep/utils/event.py` | [EventOverlap 等待与钩子，38–96 行](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/utils/event.py#L38-L96) |

数据链路：`x / (x, scales) + topk_idx + topk_weights` → `ElasticBuffer.dispatch()` → C++ runtime → `recv_x、recv_topk_*、EPHandle、EventOverlap` → 各 expert GEMM → `combine(expert_output, handle)` → 按原 token 归并的 BF16 输出。

handle 保存路由索引、接收 token 数与各级 prefix sums、源元数据、目标 slot 等。它不是普通日志对象：combine 使用这些信息反转路由；cached dispatch 也复用它避免重新计算布局。另读 `deep_ep/utils/comm.py`：communicator cache 以 process group 为键，能复用 PyTorch communicator；不可复用或要求新建时，交换 NCCL unique ID 并显式管理生命周期。

## 已确认的机制与取舍

- **源码事实：cached handle 有约束。** 复用时不能重新给 `topk_idx`，且不能要求 CPU sync；专家数、alignment、最大 token 配置要与 handle 一致。默认可复制 top-k 索引以防调用者后续修改；取消复制需要调用方自己维护所有权和不可变性。
- **源码事实：SM 估计有适用前提。** E1 用组合数估计随机 balanced gate 命中多少 rank，再比较 RDMA/NVLink 瓶颈流量与每 SM 读写能力。源码明确不适用于 V3.0 group-limited gate，并留有 expand/multiple-reduction 的 TODO。它不是对所有路由分布成立的最优解。
- **源码事实：通信速度与计算可用资源共同优化。** 推荐 SM 数有余量、对齐和硬件上界；不偏好与计算重叠时至少尝试 64 SM。只报告 dispatch 最低延迟，会遗漏它占用了多少原本可跑 GEMM 的资源。
- **源码事实：等待和确定性排序关联。** 异步路径返回 `EventOverlap`；确定性 epilogue 可注册在 `current_stream_wait()` 之后运行。其 context manager 在退出时等待，所以 scope 内适合安排独立工作，不应把尚未就绪的数据当成已完成。
- **作者宣称与验证边界：** README 称 V2 改为 NCCL Gin 并减少 SM 占用；本次只确认相应代码接口、资源模型及 NCCL wrapper，尚未验证 kernel 带宽或跨节点扩展表现。

## 与其他项目的连接

- **可选消费方：[SGLang](sglang.md) → DeepEP。** 已在 SGLang 的 `token_dispatcher/deepep_v2.py` 实读 `ElasticBuffer` 导入和 dispatch/combine。它的 adapter 自己实施容量、dtype、dummy token 和 handle 生命周期约束；库接口与服务框架接口不是一层东西。
- **可选基准集成：[DeepGEMM](deepgemm.md) → DeepEP。** Mega MoE 的分离对照使用本库完成 dispatch/combine，证据在 DeepGEMM 笔记 G4。并非两个库互相硬性依赖。
- **概念对应：** MoE expert parallel 与训练框架的 token dispatcher；本次没有审阅 Megatron 的具体对接，关系留待训练笔记确认。

## 动手实验

**状态：待执行。** 基于 `tests/elastic/test_ep.py` 固定 token 数、hidden size、top-k 和路由，先验证 dispatch/combine 对参考实现的正确性，再比较自动 SM 与手动 SM 扫描、首次与 cached dispatch。

控制变量：同一 dtype/scale layout、GPU/互联、rank 数、输入种子、是否 expand、CPU sync 和 correctness 开关。观测：dispatch/combine 延迟、逻辑/物理字节口径、数值误差、SM 数；加入独立 GEMM 时记录总流水线耗时。预期：独立通信最低延迟的配置不一定使整个 MoE 层最快。实际结果：未执行。算力：相符 Hopper/Blackwell 多 GPU 与 NVLink；跨机才需要 RDMA。不能在本次 Windows 文档环境宣称已复现 GPU benchmark。

## 下一步与疑问

- [ ] 从 `runtime.dispatch` 追到 `deep_ep/include/deep_ep/impls/dispatch.cuh` 和 hybrid dispatch，解释何时走 NVLink/跨机路径。
- [ ] 将 balanced gate 改成 skewed/group-limited 分布，检验 E1 模型的偏差，而非仅验证理想输入。
- [ ] 比较确定性排序、复制 handle 和异步等待对完整 MoE 层的成本。
