<a id="deepepmoe-token-路由为什么需要-handlestream-与带宽模型"></a>

# DeepEP: Why MoE token routing needs handles, streams, and a bandwidth model

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: `https://github.com/deepseek-ai/DeepEP` @ `01dc3aaac82068020353dce2c302e38153c0bfaa`  
Verification scope: Actually read the EPv2 Python buffer, SM/QP estimation, dispatch/combine, event waits, NCCL communicator lifecycle, and test excerpts; did not compile or run on GPUs/RDMA. The current focus is V2; V1 interface descriptions cannot be applied to it.

<a id="核心问题"></a>

## Core question

After a token is sent to multiple experts, how are the returned results matched to the original token? How do caching routing layouts, avoiding CPU synchronization, and using fewer SMs relate to numerical behavior and determinism?

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

Local paths are relative to the learning repository root.

| Evidence | Local implementation | Pinned source |
|---|---|---|
| E1 | `sources/deepep/deep_ep/buffers/elastic.py` | [Bandwidth-model SM estimation, lines 729–834](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L729-L834) |
| E2 | Same as above | [dispatch and handle/event construction, lines 855–1033](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L855-L1033) |
| E3 | Same as above | [combine uses routing metadata, lines 1046–1107](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L1046-L1107) |
| E4 | `sources/deepep/deep_ep/utils/event.py` | [EventOverlap waits and hooks, lines 38–96](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/utils/event.py#L38-L96) |

Data path: `x / (x, scales) + topk_idx + topk_weights` → `ElasticBuffer.dispatch()` → C++ runtime → `recv_x、recv_topk_*、EPHandle、EventOverlap` → GEMM for each expert → `combine(expert_output, handle)` → BF16 output combined by original token.

The handle stores routing indices, received token counts and prefix sums at each level, source metadata, destination slots, and more. It is not an ordinary logging object: combine uses this information to reverse the routing, and cached dispatch reuses it to avoid recomputing the layout. Also read `deep_ep/utils/comm.py`: the communicator cache is keyed by process group and can reuse PyTorch communicators. When reuse is unavailable or a new communicator is requested, it exchanges an NCCL unique ID and explicitly manages the lifecycle.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

- **Source fact: cached handles have constraints.** Reuse cannot supply `topk_idx` again or request CPU sync; the expert count, alignment, and maximum-token configuration must match the handle. By default, top-k indices can be copied to protect against subsequent caller modification. Disabling the copy requires callers to maintain ownership and immutability themselves.
- **Source fact: SM estimation has applicability assumptions.** E1 uses combinatorics to estimate how many ranks a random balanced gate reaches, then compares RDMA/NVLink bottleneck traffic with per-SM read/write capacity. The source explicitly states that it does not apply to the V3.0 group-limited gate and leaves TODOs for expand/multiple-reduction. It is not an optimal solution for every routing distribution.
- **Source fact: communication speed and resources available for computation are optimized together.** The recommended SM count includes margin, alignment, and a hardware upper bound; when compute overlap is not preferred, it tries at least 64 SMs. Reporting only the lowest dispatch latency omits how many resources it occupies that could otherwise run GEMM.
- **Source fact: waits are associated with deterministic ordering.** The asynchronous path returns `EventOverlap`; a deterministic epilogue can be registered to run after `current_stream_wait()`. Its context manager waits on exit, so the scope is suitable for independent work; data that is not yet ready must not be treated as complete.
- **Author claim and verification boundary:** The README says V2 switches to NCCL Gin and reduces SM usage. This reading confirmed only the corresponding code interfaces, resource model, and NCCL wrapper; kernel bandwidth and cross-node scaling have not been verified.

<a id="与其他项目的连接"></a>

## Connections to other projects

- **Optional consumer: [SGLang](sglang.md) → DeepEP.** Actually read SGLang's `token_dispatcher/deepep_v2.py`, including the `ElasticBuffer` import and dispatch/combine. Its adapter enforces its own constraints on capacity, dtype, dummy tokens, and handle lifecycle; the library interface and serving-framework interface are different layers.
- **Optional benchmark integration: [DeepGEMM](deepgemm.md) → DeepEP.** The separate-operator comparison for Mega MoE uses this library for dispatch/combine; evidence is G4 in the DeepGEMM note. The two libraries do not have a mandatory mutual dependency.
- **Conceptual correspondence:** MoE expert parallelism and token dispatchers in training frameworks. This reading did not review Megatron's specific integration; that relationship remains for the training notes to confirm.

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** Based on `tests/elastic/test_ep.py`, fix token count, hidden size, top-k, and routing. First verify dispatch/combine correctness against a reference implementation, then compare automatic SM selection with a manual SM sweep, and first dispatch with cached dispatch.

Controls: the same dtype/scale layout, GPU/interconnect, rank count, input seed, expand setting, CPU sync, and correctness switches. Observe: dispatch/combine latency, definitions of logical/physical bytes, numerical error, and SM count; when adding an independent GEMM, record total pipeline time. Expected: the configuration with the lowest standalone communication latency may not produce the fastest complete MoE layer. Actual result: not executed. Compute: compatible Hopper/Blackwell multi-GPU hardware with NVLink; RDMA is needed only across machines. GPU benchmark reproduction cannot be claimed from this Windows documentation environment.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Trace `runtime.dispatch` into `deep_ep/include/deep_ep/impls/dispatch.cuh` and hybrid dispatch to explain when the NVLink/cross-machine paths are used.
- [ ] Replace the balanced gate with skewed/group-limited distributions to test the bias of the E1 model, rather than validating only ideal inputs.
- [ ] Compare the costs of deterministic ordering, handle copying, and asynchronous waits for a complete MoE layer.
