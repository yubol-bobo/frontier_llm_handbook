# DeepGEMM：低精度 GEMM 的性能来自哪些数据与编译契约

日期：2026-09-08  
阶段：L1 源码初读  
源码：`https://github.com/deepseek-ai/DeepGEMM` @ `559d79fb6994a58b8a15b4b93bf13ccc16edf247`  
验证范围：实际阅读 C++ GEMM API、SM100 调度/生成入口、JIT compiler/cache、Mega MoE 基准实现；尚未逐指令读 CUDA 内核，也未编译或测量性能。子模块未初始化。

## 核心问题

“使用 FP8/FP4”为什么不足以获得高性能？本次追踪输入布局、量化 scale、硬件分派、模板专门化与编译缓存，理解 Python 算子调用背后的约束。另用 Mega MoE 的分离基准理解通信—计算边界。

## 源码路径与执行链路

本地路径相对于学习库根目录。

| 证据 | 本地实现 | 固定源码 |
|---|---|---|
| G1 | `sources/deepgemm/csrc/apis/gemm.hpp` | [fp8_fp4_gemm_nt，73–130 行](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/gemm.hpp#L73-L130) |
| G2 | `sources/deepgemm/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp` | [生成、选配置和启动，18–171 行](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp#L18-L171) |
| G3 | `sources/deepgemm/csrc/jit/compiler.hpp` | [Compiler.build，100–155 行](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/jit/compiler.hpp#L100-L155) |
| G4 | `sources/deepgemm/tests/test_mega_moe.py` | [DeepEP 分离基准及正确性比较，223–328 行](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/tests/test_mega_moe.py#L223-L328) |

GEMM 链路：`fp8_fp4_gemm_nt()` 检查 A/B major layout、dtype 和 M/N/K → 转换 scaling factors → 依据 SM 架构与 scale dtype 分派 → SM100 实现构造 `GemmDesc` → `get_best_config<SM100ArchSpec>()` → 创建 A/B/scale/output 的 TMA descriptors → 生成模板代码 → `compiler->build()` → 启动 kernel。

另读 `csrc/jit/cache.hpp`：先查进程内 `KernelRuntime` cache，再检查磁盘条目有效性。源码保留“考虑 cache capacity”的 TODO，因此不能假定进程内缓存已有容量上限。

## 已确认的机制与取舍

- **源码事实：dtype 与 scale/layout 必须一起解释。** API 分支中 SM90 浮点 scale 与 SM100 整数 packed scale 走不同实现；shape 对、dtype 对，并不代表 strides、scale granularity 和对齐契约都对。
- **源码事实：专门化参数覆盖执行结构。** 生成模板带入 block M/N/K、pipeline stages、线程数、cluster、SM 数、major layout、dtype 和可编译维度。实际 shape 与 `compiled_dims` 共同决定代码生成；不能把每次 Python 调用都等同于一次编译。
- **源码事实：编译缓存也是分布式工程。** 缓存键含 kernel 名、compiler signature、flags、生成代码。miss 时编译到唯一临时目录，fsync 后整体 rename 发布；另一 rank 先发布时清理自己的临时目录并使用已有结果。这解释多进程共享缓存为何不能只“写出一个 cubin 文件”。
- **源码事实：Mega MoE 对照组把边界写得很清楚。** G4 的 `run_baseline()` 是 DeepEP dispatch → grouped GEMM 1 → SwiGLU/scale 处理 → grouped GEMM 2 → combine。fused 与 baseline 有数值和 expert stats 比较，但只有依赖成功导入且测试次数启用时执行这些检查。
- **阅读推断：** 融合的收益可能来自减少中间读写、launch 和通信等待；本次没有读完整 Mega MoE CUDA pipeline，不能据此声称已验证每个阶段具体重叠比例，更没有复现 README 的性能数值。

## 与其他项目的连接

- **可选基准依赖：DeepGEMM 测试 → [DeepEP](deepep.md)。** G4 使用 `ElasticBuffer`；`test_mega_moe.py:15–33` 的 `import_baseline()` 还尝试加载 TileLang 辅助实现，失败会跳过 baseline。这不是 DeepGEMM 所有 GEMM 调用都依赖 DeepEP。
- **被选用的算子后端：[SGLang](sglang.md) → DeepGEMM。** 具体可选导入与调用证据在 SGLang 笔记 S4；不能把后端存在写成所有模型默认使用。
- **概念对应：** MoE expert routing、grouped GEMM 与 Megatron/其他训练框架中的 expert parallel 可对应；本次没有逐条审计这些训练框架的 DeepGEMM 依赖。

## 动手实验

**状态：待执行。** 第一项用 `tests/test_fp8_fp4.py` 中一个受支持的 dense shape，比较冷 JIT、同 shape 热缓存、变化 M 后的调用。第二项才运行 `test_mega_moe.py` 的 fused/baseline 对照。

输入：固定随机张量、量化 recipe、M/N/K、路由 top-k。控制变量：硬件、CUDA/compiler、SM budget、采样/预热次数及输入分布。观测：首次调用耗时、热态 kernel 时间、缓存条目、数值误差；Mega MoE 再观察总时间与通信/算子 trace。预期：冷启动与稳定吞吐分离，不能把 JIT 时间混进 steady-state TFLOPS。实际结果：未执行。算力：dense 实验需受支持 GPU；本次阅读的 SM100 与 Mega MoE 路径应在相符 Blackwell 环境验证，多卡实验还需匹配互联与子模块。

## 下一步与疑问

- [ ] 沿 `get_best_config` 读 heuristic，解释 block shape 的选择而非只记录配置值。
- [ ] 读生成代码指向的 `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`，追 TMA、barrier 与 accumulator 生命周期。
- [ ] 安装前检查子模块、支持矩阵与 benchmark baseline 是否真正启用，防止只跑 fused 路径却误报复现对照。
