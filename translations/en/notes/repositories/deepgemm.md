<a id="deepgemm低精度-gemm-的性能来自哪些数据与编译契约"></a>

# DeepGEMM: Which data and compilation contracts determine low-precision GEMM performance?

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: `https://github.com/deepseek-ai/DeepGEMM` @ `559d79fb6994a58b8a15b4b93bf13ccc16edf247`  
Verification scope: Actually read the C++ GEMM API, SM100 scheduling/generation entry point, JIT compiler/cache, and Mega MoE benchmark implementation; have not read CUDA kernels instruction by instruction, compiled, or measured performance. Submodules have not been initialized.

<a id="核心问题"></a>

## Core question

Why is “using FP8/FP4” insufficient for high performance? This reading traces input layouts, quantization scales, hardware dispatch, template specialization, and compilation caching to understand the constraints behind Python operator calls. It also uses the separate-operator Mega MoE benchmark to understand the communication–computation boundary.

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

Local paths are relative to the learning repository root.

| Evidence | Local implementation | Pinned source |
|---|---|---|
| G1 | `sources/deepgemm/csrc/apis/gemm.hpp` | [fp8_fp4_gemm_nt, lines 73–130](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/gemm.hpp#L73-L130) |
| G2 | `sources/deepgemm/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp` | [Generation, configuration selection, and launch, lines 18–171](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp#L18-L171) |
| G3 | `sources/deepgemm/csrc/jit/compiler.hpp` | [Compiler.build, lines 100–155](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/jit/compiler.hpp#L100-L155) |
| G4 | `sources/deepgemm/tests/test_mega_moe.py` | [DeepEP separate-operator baseline and correctness comparison, lines 223–328](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/tests/test_mega_moe.py#L223-L328) |

GEMM chain: `fp8_fp4_gemm_nt()` checks A/B major layout, dtype, and M/N/K → transforms scaling factors → dispatches by SM architecture and scale dtype → SM100 implementation constructs `GemmDesc` → `get_best_config<SM100ArchSpec>()` → creates TMA descriptors for A/B/scale/output → generates template code → `compiler->build()` → launches the kernel.

Also read `csrc/jit/cache.hpp`: it first checks the in-process `KernelRuntime` cache, then checks the validity of disk entries. The source retains a TODO to consider cache capacity, so an in-process cache capacity limit cannot be assumed to exist.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

- **Source fact: dtype must be explained together with scale/layout.** API branches use different implementations for floating-point scales on SM90 and integer packed scales on SM100. Correct shape and dtype do not imply that strides, scale granularity, and alignment contracts are all correct.
- **Source fact: specialization parameters cover execution structure.** Generated templates incorporate block M/N/K, pipeline stages, thread count, cluster, SM count, major layout, dtype, and compilable dimensions. The actual shape and `compiled_dims` jointly determine code generation; each Python call cannot be equated with a compilation.
- **Source fact: compilation caching is also distributed engineering.** Cache keys include the kernel name, compiler signature, flags, and generated code. On a miss, compilation targets a unique temporary directory, followed by fsync and a whole-directory rename to publish it. If another rank publishes first, the rank removes its own temporary directory and uses the existing result. This explains why a shared multiprocess cache cannot simply write out a cubin file.
- **Source fact: the Mega MoE control makes the boundaries explicit.** G4's `run_baseline()` is DeepEP dispatch → grouped GEMM 1 → SwiGLU/scale processing → grouped GEMM 2 → combine. There are numerical and expert-stats comparisons between fused and baseline implementations, but these checks execute only when dependencies import successfully and the test count enables them.
- **Reading inference:** Fusion may gain from fewer intermediate reads/writes, launches, and communication waits. This reading did not cover the complete Mega MoE CUDA pipeline, so it cannot claim to have verified the specific overlap fraction of each stage, much less reproduced the README performance figures.

<a id="与其他项目的连接"></a>

## Connections to other projects

- **Optional benchmark dependency: DeepGEMM tests → [DeepEP](deepep.md).** G4 uses `ElasticBuffer`; at `test_mega_moe.py:15–33`, `import_baseline()` also tries to load a TileLang helper implementation and skips the baseline on failure. This does not mean every DeepGEMM GEMM call depends on DeepEP.
- **Selected operator backend: [SGLang](sglang.md) → DeepGEMM.** Specific optional-import and call evidence appears in S4 of the SGLang note; the presence of a backend must not be described as default use by all models.
- **Conceptual correspondence:** MoE expert routing and grouped GEMM correspond to expert parallelism in Megatron/other training frameworks. This reading did not audit those frameworks' DeepGEMM dependencies individually.

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** First use one supported dense shape from `tests/test_fp8_fp4.py` to compare cold JIT, a warm cache for the same shape, and a call after changing M. Only then run the fused/baseline comparison in `test_mega_moe.py`.

Inputs: fixed random tensors, quantization recipe, M/N/K, and routing top-k. Controls: hardware, CUDA/compiler, SM budget, measurement/warmup counts, and input distribution. Observe: first-call duration, warm kernel time, cache entries, and numerical error; for Mega MoE, also observe total time and communication/operator traces. Expected: separate cold start from steady throughput; do not mix JIT time into steady-state TFLOPS. Actual result: not executed. Compute: the dense experiment requires a supported GPU; the SM100 and Mega MoE paths read here should be verified in a compatible Blackwell environment, with matching interconnects and submodules for multi-GPU experiments.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Follow `get_best_config` into its heuristic to explain block-shape selection rather than merely recording configuration values.
- [ ] Read `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`, which the generated code targets, and trace the lifecycles of TMA, barriers, and accumulators.
- [ ] Before installation, check submodules, the support matrix, and whether the benchmark baseline is actually enabled, to avoid running only the fused path and incorrectly reporting reproduction of the comparison.
