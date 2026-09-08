<a id="sglang缓存调度和-moe-后端怎样共同决定-rollout-成本"></a>

# SGLang: How caching, scheduling, and MoE backends jointly determine rollout cost

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: `https://github.com/sgl-project/sglang` @ `30e7a3072d3f1e9bd70cd5e44146ca27c80522c4`  
Verification scope: Actually read the two local scheduler loops, batch-merging excerpts, radix matching/eviction/lock references, the DeepEP v2 adapter, and the DeepGEMM wrapper; did not start a service, install dependencies, or run GPU experiments.

<a id="核心问题"></a>

## Core question

Why is a rollout engine more than an HTTP wrapper around `model.generate()`? The key is reusing KV across multiple ongoing generation requests while scheduling the next GPU batch, processing the previous batch's results, and keeping distributed MoE data layouts and synchronization boundaries consistent.

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

Local paths are relative to the learning repository root.

| Evidence | Local implementation | Pinned source |
|---|---|---|
| S1 | `sources/sglang/python/sglang/srt/managers/scheduler.py` | [Normal/overlap loops, lines 1893–2022](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/managers/scheduler.py#L1893-L2022) |
| S2 | `sources/sglang/python/sglang/srt/mem_cache/radix_cache.py` | [Matching, cache lifecycle, eviction, and references, lines 400–680](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/radix_cache.py#L400-L680) |
| S3 | `sources/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py` | [Optional import, buffer, and dispatch/combine, lines 38–418](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py#L38-L418) |
| S4 | `sources/sglang/python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` | [Optional import and masked GEMM call, lines 18–100](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py#L18-L100) |

Normal loop: `ingest_requests()` → `get_next_batch_to_run()` → `run_batch()` → `process_batch_result()`. The overlap loop first schedules the current forward, puts its result in `result_queue`, and then processes the previous batch's results; sampling portions that depend on previous-batch state run after result processing.

Also read `scheduler.py:3500–3609`: completed prefill batches merge into `running_batch`; unfinished chunked requests are excluded and their existing KV cached when appropriate, before a new prefill plan is built. On the cache side, `match_prefix()` returns KV-pool indices and tree nodes; active requests hold node references, which become evictable capacity again only after release.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

- **Source fact: a prefix key contains more than text.** In `RadixKey`, token IDs and `extra_key` jointly determine the namespace; contexts such as different adapters must not incorrectly share KV. Matching also aligns to page size and may split tree nodes.
- **Source fact: hit rate is not the only state.** `inc_lock_ref/dec_lock_ref` maintain protected/evictable token counts along ancestor paths; `evict()` releases KV segments from a priority queue of evictable leaves. Cache capacity must serve both executing requests and reusable history.
- **Source fact: overlap has a latency tradeoff.** A special branch for consecutive prefills can disable overlap; the source comment explains that this improves TTFT for the preceding batch but may reduce throughput. With DP attention, synchronized batch information unifies the decision to avoid ranks taking different synchronization paths.
- **Source fact: the MoE adapter handles data contracts.** The DeepEP v2 path uses expanded/masked layouts for decode to avoid CPU synchronization and contiguous layout for extend. It checks hidden size, top-k, and per-rank token capacity; an empty rank inserts a zero-weight dummy token when CPU synchronization is needed. The dispatch handle must be consumed by combine and is released on failure as well.
- **Reading inference:** Multiturn agent requests often share prefixes, so KV and scheduling affect sampling cost. The effect also depends on context truncation/rewriting, models, request distributions, and concurrency; the existence of code does not imply a fixed speedup.

<a id="与其他项目的连接"></a>

## Connections to other projects

- **Optional backend: SGLang → [DeepEP](deepep.md).** S3 explicitly imports `ElasticBuffer`; a missing library retains an error and raises it when that path is selected. This is a substantive adapter, beyond a README compatibility claim. Note that it was not installed here, and ABI compatibility between the two HEADs is unproven.
- **Optional backend: SGLang → [DeepGEMM](deepgemm.md).** S4 imports and calls GEMM under `ENABLE_JIT_DEEPGEMM`. Also read `deep_gemm_wrapper/configurer.py:18–36`: the switch checks the device, importability, and environment configuration together.
- **Conceptual correspondence: SGLang ↔ [Harbor](harbor.md).** One manages generation/caching/operators; the other manages task environments/verification. Check specific training-framework connections in the [RL notes](../connections/rl.md); the three cannot be drawn as an unconditional direct-dependency chain.

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** Select a model that fits entirely on one GPU and generate two groups of equal-token-length requests: A shares a long token prefix; B has the same length but randomly different prefixes. Enable and disable radix caching for each, forming four groups.

Controls: the same model/weights, tokenization, maximum output length, sampling settings, concurrency, and hardware; distinguish cold-cache from warmed measurements. Observe: TTFT p50/p95, output tokens per second, end-to-end time, cache-hit tokens, and KV usage. Expected: larger benefits for A; measurements must establish whether they materialize. Actual result: not executed. Compute: one GPU suffices for this experiment; cross-machine MoE communication is outside its scope.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Trace `get_new_batch_prefill()` into PrefillAdder and diagram the actual token-budget and chunk-size constraints.
- [ ] Manually diagram radix splits and lock_ref changes for a shared-prefix example.
- [ ] Check cache invalidation and pause/resume implementation during RL weight updates to avoid directly applying serving correctness to training rollout.
