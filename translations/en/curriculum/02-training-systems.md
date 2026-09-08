<a id="训练系统课程m05m09"></a>

# Training Systems Curriculum: M05–M09

Complete M05 first, then study M06/M07 in parallel; M08 integrates all three. M09 can begin after M05 and must be completed before the agent RL systems module.

**All exercises are designs awaiting execution and do not represent personal course completion.** Without a GPU, you can follow the control-flow, numerical, state-machine, and trace tracks; CPU simulation provides no evidence of real GPU performance. Workload estimates cover core reading plus paper/CPU exercises, excluding installation, queuing, and full training. GPU extensions require additional time; none of these estimates is a guarantee.

Source code is pinned to the recorded snapshots; newly added official entry points were checked on 2026-09-08. Each module proceeds through three groups of core reading, followed by exercises and self-assessed completion checks. The curriculum design is an engineering synthesis; label source facts, the authors' benchmarks, and your own measurements separately.

<a id="m05"></a>

<a id="m05--分布式与并行基础"></a>

## M05 — Distributed Computing and Parallelism Fundamentals

**Prerequisites: M01, M02, M04. Diagnostic for skipping material:** draw gradient averaging across two GPUs, explain the global objective for unequal-length microbatches, and list what DP/TP/PP/CP partition and their communication costs. If all answers are correct, skim the principles group; the source reading and completion checks remain required.

**Topics in detail.** Ranks, process groups, physical topology, and logical meshes; sum/average, all-reduce, reduce-scatter, all-gather, and point-to-point communication; ownership of parameters/gradients/optimizer states; DDP, fully sharded data parallelism (FSDP)/ZeRO, and hybrid sharded data parallelism (HSDP); column/row tensor partitioning and non-TP parameters; pipeline parallelism (PP) schedules, microbatches, bubbles, and recomputation; the difference between context parallelism (CP) and sequence parallelism used with tensor parallelism (TP). Finally, place the GPU memory ledger, effective batch, and gradient normalization on the same diagram.

**Core reading, in order:**

1. Read the data/tensor/pipeline/context parallelism diagrams in the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), then compare the inputs and outputs in [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html). The goal is to write what each rank ultimately receives for small tensors; do not pursue the benchmark tables on this pass.
2. Read the [TorchTitan notes](../notes/repositories/torchtitan.md) and [parallel_dims.py:106–240](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/distributed/parallel_dims.py#L106-L240), locating world size and the batch/sparse meshes. Then read `parallelize_llama` at `models/llama3/parallelize.py:23`: it calls the declarative `model.parallelize`, then handles AC/compile/FSDP. Document mesh responsibilities and transformation order.
3. Compare [Megatron gradient finalization:560–713](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/distributed/finalize_model_grads.py#L560-L713) with [operations chapter §3](../handbook/02-distributed-pretraining-operations.md). Mark where the code waits for gradient communication, reduces special parameters, and scales by valid target tokens; explain why the two loops cannot simply be spliced together.

**Optional reading:** overlap/recomputation in the Playbook and Megatron pipeline schedules. Understand basic updates before pursuing interleaved PP, CUDA graphs, and specialized recomputation meshes.

**Draw/derive:** a parameter/sample diagram for 8 ranks with `DP=2,TP=2,PP=2`; `B_sequence=D×b×m`, where m is the total number of microbatches in one update. Calculate bytes for states, activations, and temporary buffers separately, explaining which term recomputation/sharding changes and retaining dtype and peak-memory assumptions.

**Exercise A: CPU gradient reconciliation.** Use a small linear model to compare full-batch token averaging, groups weighted by valid target tokens, and the incorrect average of local averages. Cover equal lengths, unequal lengths, and fully masked inputs; compare loss, gradients, and one update, with a predefined tolerance and zero-valid-target policy. You may write gradients manually in NumPy or use CPU autograd. **GPU extension:** after the comparisons pass, verify real DP scaling on two compatible GPUs; summing lists is not a communication test.

**Exercise B: A layout ledger on paper.** Propose two legal layouts for the same model, draw a timeline with 2 PP stages and 4 microbatches, and mark synchronization, stage imbalance, and logical communication bytes. **GPU extension:** fix the global batch and pass numerical comparisons before collecting traces for the two layouts; do not attribute the effects of simultaneously changing the batch to the parallelism method.

**Self-assessed completion:** why does CP not add samples? Why might FSDP be slower? How does the global norm exclude replicas? Why can EP not be arbitrarily multiplied into the device count? Answers must point to source code; “more GPUs must mean a larger batch” and “all-reduce always happens at the end of backward” are common misconceptions.

**Deliverables:** `M05-layout.md`, state/communication ledgers, a numerical reconciliation script and results (label as a design if unexecuted), and a pipeline diagram. **Workload:** 16–26 hours, plus 4–8 hours for GPU extensions. **Next modules:** M06 and M07; M09 can also begin.

<a id="m06"></a>

<a id="m06--moe-架构路由与分布式执行"></a>

## M06 — MoE Architecture, Routing, and Distributed Execution

**Prerequisite: M05. Diagnostic for skipping material:** write the forward and backward mappings for 6 tokens, 4 experts, and top-2 routing, explaining expert replicas, capacity, and padding. If you pass, skim the routing definitions, but still read about handles and asynchronous ownership.

**Topics in detail.** Total parameters/active parameters/shared experts; router logits, top-k, routing probabilities, and their gradients; capacity, dropping, padding; the different roles of auxiliary balancing objectives and router z-loss; expert DP/TP/expert parallelism (EP) groups; token permutation, prefix sums, and grouped GEMM; backward pairing of dispatch/combine; stream/event, buffer, and handle lifetimes; uniform and skewed loads, and routing within/across nodes. The balancing strategies of different recipes must not all be described as the same kind of congestion prevention.

**Core reading, in order:**

1. Start with the [Megatron notes](../notes/repositories/megatron-lm.md), then enter [router.py:750–841](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/router.py#L750-L841). Record shapes from logits to the routing map/probabilities, capacity branches, and where auxiliary losses enter the gradient; distinguish training objectives from logged metrics.
2. Follow route, dispatch, expert compute, and combine in [moe_layer.py:637–742](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/moe_layer.py#L637-L742). Use the notes to trace manager selection in `token_dispatcher.py`; draw only verified adapter edges. The presence of grouped GEMM does not establish a DeepGEMM integration.
3. Read the [DeepEP notes](../notes/repositories/deepep.md) and [dispatch:855–1033](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L855-L1033), then `combine:1046–1107` in the same file. Explain what the handle stores, why cached dispatch constrains new routes, and which event the consumer must wait for.

**Optional reading:** DeepEP `EventOverlap` and SM estimation; the isolated benchmarks in the [DeepGEMM notes](../notes/repositories/deepgemm.md). The Megatron integration path and DeepEP V2 are not automatically compatible; separate version testing is required.

**Draw/derive:** a token → expert → original-order diagram and backward-gradient diagram, marking shapes, indices, weights, and owners. Distinguish logical duplication, bytes crossing ranks/nodes, and padding, and explain the impact of the busiest expert.

**Exercise A: A minimal CPU MoE.** Fix top-k assignments/weights and use four linear transformations as experts, comparing per-token execution with a pack/group/restore implementation. Cover empty experts, repeated destination ranks, skew, and capacity truncation. Check outputs and fixed-route gradients; this does not validate differentiability of the discrete top-k selection. **GPU extension:** after comparisons pass and versions are compatible, test a multi-GPU dispatcher, checking correctness before latency.

**Exercise B: CPU handle and load simulation.** Deliberately combine route B with a handle from route A and construct a detectable error. Generate uniform, concentrated, and node-constrained routes and tabulate expert/communication ledgers. **GPU extension:** after understanding events/ownership, sweep the SM budget and compare communication time with full-layer time. The simulator does not predict RDMA throughput or the best SM count.

**Self-assessed completion:** which routing changes alter the objective? Which expert gradients may be averaged? When does a handle become invalid? Why might the shortest dispatch not produce the fastest whole layer? Use counterexamples to correct “active parameters determine all costs,” “no drops means balanced,” and “asynchronous means fully overlapped.”

**Deliverables:** `M06-token-journey.md`, an executable reference implementation, forward/backward reconciliation, three routing tables, and diagrams of actual dependencies and conceptual relationships. **Workload:** 18–30 hours, plus 6–12 hours for GPU extensions. **Next module:** complete M07 before entering M08; M09 may run in parallel.

<a id="m07"></a>

<a id="m07--gpu-内核低精度与-profiling"></a>

## M07 — GPU Kernels, Low Precision, and Profiling

**Prerequisite: M05; may run in parallel with M06. Diagnostic for skipping material:** explain different speeds at equal FLOPs, calculate tiled softmax by hand, and give a counterexample with the correct dtype but an incorrect scale/stride. If you pass, skim memory hierarchy material while retaining the numerical and measurement exercises.

**Topics in detail.** HBM, on-chip storage, registers, warps/thread blocks; arithmetic intensity, coalesced memory access, tiling, fusion, occupancy, and resource contention; attention intermediate matrices and online softmax; low-precision formats, scale granularity, amax, accumulation precision, and master-state precision; shape/stride/alignment and architecture dispatch; JIT/autotune/graph cold starts and steady state; timelines for CPU launches, GPU kernels, and communication overlap. Distinguish model FLOPs utilization (MFU) from hardware FLOPs utilization (HFU), which includes additional hardware work; recomputation must not quietly count as useful model FLOPs.

**Core reading, in order:**

1. Read §2.3 and §3 of the [FlashAttention-2 authors' paper](https://tridao.me/publications/flash2/flash2.pdf), following only tiled reads/writes, online softmax, recomputation, and work partitioning. Do not memorize specific hardware speedups yet. It explains IO optimization for exact attention; “exact” does not promise bitwise identity across different floating-point reduction orders.
2. Read the official Triton [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) and [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) tutorials in order. Identify program instances, tiles, masks, and address calculations individually. Understand why boundary masks are needed before examining what autotune searches.
3. Read the [DeepGEMM notes](../notes/repositories/deepgemm.md) and [gemm.hpp:73–123](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/gemm.hpp#L73-L123), then follow configuration generation and compilation caching in G2/G3 of the notes. Write the shape/dtype/layout/scale/architecture contract for one call; reading the entire CUDA kernel on your first pass is not required.

**Optional reading:** [TE low-precision principles](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html), schedule/export trace in [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), and FlashAttention implementations for newer architectures. Kernels for different GPU generations are not directly interchangeable.

**Draw/derive:** memory flow for Q/K/V tiles; combining softmax exponential sums and outputs using a common maximum; the critical path of a trace containing host work, kernels, and events. Do not add all overlapping intervals together.

**Exercise A: CPU numerical reference.** Compare full-row and online softmax, covering ordinary inputs, large magnitudes, masks, and fully masked inputs. Use simplified quantization/dequantization to compare outlier effects on per-tensor/per-block scales, explicitly noting that this is not complete hardware FP8/FP4. **GPU extension:** once the reference and gradient comparisons are defined, test supported Triton/TE paths and record errors, outliers, shapes, and execution times.

**Exercise B: Measurement audit.** Add range labels to a CPU process and parse its trace; identify incorrect synchronization, duplicate timing, and JIT contamination on a hypothetical GPU timeline. **GPU extension:** fix model/shapes/versions, measure cold and warm calls separately using synchronization or device events, and report sample counts, quantiles, and profiler overhead. Change only the tile or one fusion boundary at a time, comparing performance after numerical checks pass; a CPU trace does not provide GPU bandwidth.

**Self-assessed completion:** why might writing fewer matrices matter more than performing fewer multiplications? Why might a larger tile be slower? Why does kernel correctness not guarantee convergence? Correct “FP4 support means FP4 throughout the entire pipeline,” “peak performance equals training speed,” and “summing profiler rows gives wall-clock time.”

**Deliverables:** `M07-kernel-contract.md`, a numerical reference/error table, a memory-flow diagram, and a trace audit with measurement definitions; attach device and warmup records for actual measurements. **Workload:** 20–34 hours, plus 6–12 hours for GPU extensions. **Next module:** M08. Interested readers may also complete a single-kernel capstone, but cannot use it to skip training-state recovery.

<a id="m08"></a>

<a id="m08--训练运行存储checkpoint-与容错"></a>

## M08 — Training Operations, Storage, Checkpoints, and Fault Tolerance

**Prerequisite: M05; applies M06/M07. Diagnostic for skipping material:** list five classes of state beyond weights, explain staging versus persistent writes, and design a test that detects incorrect data order after recovery. If you pass, skim the checklist but retain the fault exercises.

**Topics in detail.** Cluster health and GPU/NIC/NUMA topology; data shards, shuffle, packing, prefetch, and reader state; shared-storage throughput, metadata pressure, and tail latency; checkpoint shards, manifests, commit points, and asynchronous staging; optimizer/scheduler/RNG/low-precision state; same-topology recovery and resharding; overflow skips, failure replay, slow ranks, communication timeouts, and silent corruption; checkpoint-interval tradeoffs between save overhead and lost work that must be recomputed.

**Core reading, in order:**

1. Read [operations chapter §7–8](../handbook/02-distributed-pretraining-operations.md), comparing with [TorchTitan dcp.py:580–650](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/components/checkpointer/dcp.py#L580-L650). Trace load, the staging future, and the saving future. Note the current rank-local RNG TODO; a library name is not a guarantee of exact recovery.
2. Read [Megatron RNG state:451–516](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/training/checkpointing.py#L451-L516), then locate `maybe_save_dataloader_state` in the same file. Together with the update-success/skip branches in `training.py`, distinguish consumed samples, attempted iterations, and successful updates, and explain why these may advance separately under deterministic rules.
3. Read Design and implementation, File system interfaces, and the FUSE limitations in [3FS Design Notes](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md). Draw the four roles—cluster manager, metadata, storage, and client—and their read/write paths. Separate CRAQ storage consistency from the application-level requirement that all training states belong to the same step. This is external reading on principles; deploying 3FS is not required.

**Optional reading:** [NCCL Tests metrics](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md), TorchTitan checkpoint documentation, and the 3FS native client. The authors' benchmarks do not promise throughput on your machine.

**Draw/derive:** update, staging, disk-write, and publication timelines, with recoverable state at each boundary; owners of parameters, optimizer, RNG, and low-precision states; the difference between a dataloader's prefetch cursor and committed cursor.

**Exercise A: CPU interruption and recovery.** Compare K uninterrupted steps of a small trainer with random sampling against saving at step j and resuming in a new process. Omit the optimizer, scheduler, sample order, and RNG/counters one at a time, and verify that differences are detectable. Use only an experiment-specific temporary directory; resetting a seed does not restore the random stream. **GPU extension:** after the CPU comparison passes, test same-topology recovery on a small task, then resharding; assess bitwise identity and tolerance-based agreement separately.

**Exercise B: A paper/CPU fault matrix.** Simulate shards, checksums, and a final manifest with small files. Inject missing files, truncation, unpublished state, an advanced cursor, and slow writes, verifying that incomplete states are rejected. Write separate counting rules for overflow skips and fail-fast/replay. **GPU extension:** with a test cluster and rollback point, measure concurrent data reads/saves and termination of a specified process. Local simulation does not establish real 3FS/RDMA fault-tolerance performance.

**Self-assessed completion:** why are complete weights insufficient to guarantee resumable training? Why does strongly consistent storage not guarantee consistent training state? Why might a timeout conceal an error? Distinguish loadability, statistical reproduction, and bitwise identity; distinguish overflow skips from failure replay, and consumed tokens from successful updates.

**Deliverables:** `M08-recovery-contract.md`, a state checklist, fault matrix, uninterrupted/resumed comparison records, and an operations and rollback manual. **Workload:** 18–30 hours, plus 8–16 hours for real-cluster extensions. **Next module:** complete M09; these operating conventions continue into M13 asynchronous agent RL and the M15 integrated project.

<a id="m09"></a>

<a id="m09--推理-serving-与-rollout-成本"></a>

## M09 — Inference Serving and Rollout Cost

**Prerequisites: M01, M02, M05; required before agent RL. Diagnostic for skipping material:** explain prefill/decode, TTFT/ITL/throughput, estimate MHA/GQA KV memory, and explain invalidation after a weight change. If you pass, skim the terminology while retaining scheduling and version boundaries.

**Topics in detail.** Autoregressive generation, prefill, and token-by-token decode; continuous batching, chunked prefill, preemption/queuing; KV pages, radix prefixes, active references, and eviction; cache namespaces for model/adapter/weight versions; TP/EP inference layouts; TTFT, ITL, TPOT, tail latency, and request success rate; cold/warm caches, arrival rate, and concurrency; long trajectories, tool waits, cancellation, and policy staleness in rollouts. SGLang serves as the generation system here; it does not perform base-model optimizer updates.

**Core reading, in order:**

1. Read the [SGLang notes](../notes/repositories/sglang.md) and pinned [Bench Serving metrics:230–255](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/docs/docs/developer_guide/bench_serving.mdx#L230-L255). Write each metric's start/end times, token denominator, and treatment of failed samples; a single tokens/s value cannot describe service quality.
2. Compare the normal and overlap loops in [scheduler.py:1893–2022](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/managers/scheduler.py#L1893-L2022), then use the notes to trace prefill merging. Draw the dependencies among GPU execution for the current batch, processing results from the previous batch, and admitting the next batch, rather than interpreting overlap as unlimited concurrency.
3. Read [radix_cache.py:400–680](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/radix_cache.py#L400-L680), tracing match, split, lock references, and eviction. Record the relationship among keys, KV indices, and active requests. Measurement is still needed to determine how much work a token-prefix hit actually saves.

**Optional reading:** SGLang adapters, speculative decoding, weight updates, and the 3FS KV cache. Understand scheduling before reading about prefill/decode disaggregation; the existence of an implementation does not mean it is enabled by default.

**Draw/derive:** for uncompressed MHA/GQA with equal K/V dimensions, `KV bytes≈2×number of layers×cached tokens×number of KV heads×head_dim×bytes per element`. Account separately for sharding, page waste, and metadata; do not apply this formula to MLA. Draw a radix tree with shared prefixes, and separate model, queuing, and tool time within a trajectory.

**Exercise A: CPU cache/scheduling model.** Use an integer-token prefix tree to test sharing, cancellation, release, eviction, and version switching. Check that active KV entries are not reclaimed and cross-version lookups do not produce false hits; compare two admission orders under assumed service times. **GPU extension:** after the invariants pass, use a small model that fits on the device to measure cache on/off and request order. A simulated hit rate does not equal real acceleration.

**Exercise B: A cost experiment on paper.** Design a “shared/random prefixes × cold/warm caches × load” matrix, fixing the model, tokenization, output budget, and sampling, and distinguishing arrival rate from the concurrency limit. A CPU can parse a trace with documented provenance; label fabricated data as such. **GPU extension:** after the numerical baseline passes, record successes/failures, tokens, TTFT/ITL, wall-clock time, and queuing. Do not count dropping slow requests as acceleration; cost per successful trajectory also includes tools and unusable samples.

**Self-assessed completion:** why might faster prefill slow down decode? Why might the same text produce different cache keys? Why does a serving benchmark not establish policy/logprob correctness for rollouts? Correct “higher hit rates are always better,” “cheap tokens mean cheap tasks,” and “an HTTP API is a rollout.”

**Deliverables:** `M09-serving-budget.md`, a KV ledger, scheduling/reference diagrams, an experimental matrix, and a cost table with metric definitions; explicitly leave unexecuted items blank. **Workload:** 16–26 hours, plus 4–10 hours for GPU extensions. **Next module:** enter the [M10–M14 post-training and agent curriculum](03-posttraining-and-agents.md). In particular, complete M10/M11 first, then apply this module to the M12 harness and M13 asynchronous agent RL; finally connect the end-to-end evidence in M15.
