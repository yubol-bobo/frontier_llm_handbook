<a id="megatron-lmmoe-的-routerexpert-compute-与通信后端如何接起来"></a>

# Megatron-LM: Connecting the MoE router, expert compute, and communication backend

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/NVIDIA/Megatron-LM @ `c9b53d0a87cb926f47115259593ecfeb351ca29f`  
Verification scope: Local MoE forward, TopKRouter, Flex/DeepEP dispatcher, and the Buffer interface in fused_a2a; did not build CUDA extensions, install DeepEP, or start training or benchmarks.

<a id="核心问题"></a>

## Core question

Sending a token to only a few experts appears to save FLOPs. Why does it instead require complex metadata, permutations, all-to-all communication, and communication buffers? How do router algorithm choices affect system behavior?

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

1. [moe_layer.py](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/moe_layer.py), [`MoELayer.forward`, L637–742](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/moe_layer.py#L637-L742): shared experts / route → preprocess → dispatch → routed_experts_compute → combine → postprocess. Also read the implementations of these substeps at L466–618 in the same file to confirm the flow beyond the docstring.
2. [router.py](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/router.py), [`TopKRouter.routing`, L750–841](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/router.py#L750-L841): flatten logits → z-loss → top-k / optional balancing → capacity dropping → attach auxiliary loss → expert bias bookkeeping, outputting probs and routing_map.
3. [token_dispatcher.py](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/token_dispatcher.py), [`_DeepepManager`, L1225–1335](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/token_dispatcher.py#L1225-L1335): converts token×experts probabilities to top-k indices/probs, marks dropped positions -1, calls fused_dispatch, and retains the handle for combine. Also read L1815–1866 to confirm that the Flex dispatcher explicitly selects deepep / hybridep / ncclep.
4. [fused_a2a.py](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/fused_a2a.py), [L11–95](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/fused_a2a.py#L11-L95): guarded import of `deep_ep.Buffer`, determines dispatch/combine buffer requirements from the group and hidden bytes, then creates or reuses a buffer. FusedDispatch is a custom autograd Function. Its full backward was not read in this pass.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

**Router output is a data-layout plan.** The logical input shape is sequence×batch×hidden; the router transforms logits into tokens×experts and returns probabilities/Boolean mappings for sparse selections. Each expert does not run once over the full batch: the dispatcher arranges tokens by destination rank and local expert, expert compute receives `tokens_per_expert`, and combine must restore original token positions. Efficient local GEMM and communication layout are two sides of the same problem.

**Capacity is both an algorithmic and a system switch.** Configuring a capacity factor invokes token dropping, with configurable drop policy / pad to capacity. This can change fixed shapes and load, but also changes which tokens are processed. Load-balancing auxiliary loss has separate batch, sequence, and global paths and attaches to the autograd chain of probs. Auxiliary loss cannot be treated as an unrelated logging metric, and balancing, dropping, and padding are not synonyms.

**Combining EP / TP has concrete constraints.** During training, when the attention TP group>1 and sequence_parallel=False, MoELayer.forward raises an error directly; the source gives possible performance degradation as the reason. The Flex DeepEP path requires TP×EP>1 and maps the group, top-k, and expert count into the manager. These are implementation requirements of the selected training/dispatch path, not a theoretical requirement that MoE use multiple GPUs.

**DeepEP is an optional backend, not a mandatory dependency of every MoE.** The constructor selects allgather/alltoall/flex using `moe_token_dispatcher_type`; Flex then selects a communication manager. The import guard in `fused_a2a.py` and ImportError in `_DeepepManager` indicate that DeepEP is required only when this path is enabled. The actual bridge includes dtype conversion: DeepEP probabilities require FP32, and BF16/FP16 trigger a warning and conversion.

**Communication state must be retained.** The manager retains `handle`, dispatched indices/probs, and tokens_per_expert; these metadata support subsequent restoration and communication pairing. Buffer reuse is determined by group and NVLink/RDMA size hints; allocating temporary buffers each time does not necessarily provide the same performance. `async_finish` / comm-stream parameters require further auditing of event lifecycles; the name async does not establish sufficient overlap of computation and communication.

**A forward pass can also be captured or recomputed in segments.** `fwd_execution_map` and intermediate_tensors allow route/expert/postprocess segmentation; MoE recompute uses TE checkpoint under FP8/FP4 and tensor_parallel checkpoint otherwise. Reading the main chain requires distinguishing a normal complete forward from CUDA graph partial capture; an early return must not be mistaken for an omitted combine.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Relationship | Evidence and boundaries |
|---|---|
| DeepEP: explicit optional backend | `fused_a2a.py:11–17` imports its Buffer; `token_dispatcher.py:1835–1843` selects the manager. Follow this edge into the [DeepEP note](deepep.md), but API/build compatibility with the independently cloned HEAD has not been verified. |
| slime / verl / miles: backend relationships requiring confirmation at their own entry points | They may use Megatron training components; this note does not vouch for callers based on project names. See the corresponding repo notes and relationship labels in [training connections](../connections/training.md). |
| TorchTitan: conceptual correspondence | Compare tensor/expert layouts, activation checkpointing, and communication/computation boundaries; this is not a direct dependency. |
| DeepGEMM: conceptual correspondence | Grouped expert GEMM and token dispatch have adjacent data layouts. The code read here does not suffice to prove that this expert-compute path in this snapshot calls DeepGEMM; it cannot be drawn as an integrated dependency. |

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** Verify numerical correctness and performance separately.

- Question: with identical routing indices/probs, does the communication backend preserve the token→expert→token mapping and gradients, and how do different load patterns affect throughput?
- Inputs: a tiny MoE with fixed model weights and inputs; two routing patterns: uniform distribution and concentration on a few experts. Initially disable token dropping / recompute / CUDA graphs.
- Controls: expert count, top-k, dtype, TP/EP layout, valid token count, and initialization; switch only among supported dispatcher backends.
- Metrics: output/gradient numerical differences, token counts, drop rate, peak GPU memory, dispatch/combine time, expert-compute time, and load imbalance across ranks.
- Expected: paths with the same semantics should agree within a predefined numerical tolerance; skewed load may make communication or a few experts the bottleneck. No evidence guarantees that a fused backend is faster for every small batch.
- Compute: routing logic can first use CPU tensor-level cases; actual multi-GPU DeepEP testing requires its CUDA, interconnect, buffer, and other requirements. The current Windows workspace is only a source library and has not prepared that runtime environment.
- Actual result: none; README benchmarks are not treated as measurements on this machine.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Continue into `FusedDispatch.backward` / `FusedCombine` to explain why forward dispatch pairs with backward combine.
- [ ] Trace grouped MLP backend selection in `experts.py`, distinguishing TransformerEngine, other GEMM paths, and optional optimizations.
- [ ] Compare against an actual Megatron RL caller, checking its pinned commit, model state dict, weight synchronization, and training/inference probability consistency.
