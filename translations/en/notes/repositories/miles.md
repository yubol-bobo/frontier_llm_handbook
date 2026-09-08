<a id="milesmoe-rollout-的离散路由怎样在训练中重放"></a>

# Miles: How is discrete MoE rollout routing replayed during training?

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/radixark/miles @ `3de96596f16b9e6d23ba550c4c47de3479c9f14c`  
Verification scope: Read the implementation, related configurations, and declarations in the pinned local snapshot; did not install upstream dependencies or start a model, sandbox, or training run. The control-flow assessments in this note come from static reading and do not represent a reproduction of the authors' performance or benchmarks.

<a id="核心问题"></a>

## Core question

Are identical model weights, tokens, and temperature enough for rollout and trainer to compute the same MoE path? This reading focuses on routing replay rather than surveying all Miles optimizations. The key question is how discrete expert indices stay aligned across microbatches, parallel shards, and recomputation.

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

- [Replay cache, stages, and top-k wrapper](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/utils/replay_base.py#L14-L241); local: [`miles/utils/replay_base.py`](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/utils/replay_base.py).
- [Aligning rollout replay tensors to training microbatches / CP / SP](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/training_utils/replay_data.py#L30-L134); local: [`miles/backends/training_utils/replay_data.py`](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/training_utils/replay_data.py).
- [Replay-state transitions in the training actor](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/megatron_utils/actor.py#L490-L621); local: [`miles/backends/megatron_utils/actor.py`](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/megatron_utils/actor.py).
- [Megatron MoE layer-layout mapping](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/megatron_utils/replay_utils.py#L1-L24); local: [`miles/backends/megatron_utils/replay_utils.py`](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/megatron_utils/replay_utils.py).

`train_actor()` first creates data iterators for logprob computation and training; with rollout replay enabled, it calls `fill_replay_data()` on each replay manager. This function requires each sample to supply a `[num_tokens - 1, num_streams, topk]` tensor. It replays the iterator in the same microbatch order as training, appends the final loss-masked token, aligns CP, SP, and padding, and finally maps to the Replay queue of each model layer.

`register_replay_list_moe()` uses Megatron's layer offset and the number of local layers in each virtual pipeline stage, skips dense layers, and loads the correct stream only for MoE layers on the current rank. `Replay.record()` puts indices in a pinned CPU buffer; forward/backward have independent read pointers and move data back to the current GPU when needed.

In the actor, reference/teacher computation uses fallthrough; the old-actor/logprob stage selects record or replay_forward according to configuration. It then clears the forward cursor, sets the training stage to replay_backward, and clears the queues after completion. This confirms only scheduling and replay-wrapper implementation, not layer-by-layer installation of every model patch.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

- **Implementation fact:** The top-k wrapper replays selected indices; with `return_probs=True`, it still retrieves corresponding scores from the current `scores.gather()`, rather than freezing the entire probability array or gradients. This controls discrete selection rather than reusing old logits.
- **Implementation fact:** Padding is marked -1, and fully invalid rows are replaced with valid indices before the actual gather; shape assertions check token count and topk. Index correctness also depends on CP/SP sharding and microbatch order, so checking total length alone is insufficient.
- **Implementation fact:** `IndexerReplayManager` and `RoutingReplayManager` share mechanisms, but indexer indices refer to token/KV positions and must be rebased in packed sequences. Token-position offsets must not be applied to MoE expert ids.
- **Implementation fact:** The overlap checker can compare recomputed top-k with replay top-k; it has thresholds and switches. Having a checker interface by default does not mean zero mismatches were measured here.
- **Reading assessment:** Pinned CPU caching saves resident GPU space but introduces transfer/synchronization costs. Replay can constrain some train–inference inconsistency, but cannot alone ensure agreement across all low-precision kernels, softmax, and weight versions, much less establish that the entire training process is “true on-policy.”

<a id="与其他项目的连接"></a>

## Connections to other projects

| Object | Relationship type | Evidence and boundaries in this reading |
|---|---|---|
| slime | Project lineage | Local `README.md:109–112` explicitly states that Miles is forked from slime; this does not mean their current files/APIs remain interchangeable. See the relationship note for the pinned link. |
| Megatron-LM | Direct dependency of the training backend | `replay_utils.py:1–2` actually imports Core layer-layout functions, beyond a README mention. |
| SGLang | Direct inference integration | Local `miles/backends/sglang_utils/sglang_engine.py:8,68` imports ServerArgs and constructs `python -m sglang.launch_server`; this reading only located the connection and did not trace serving internals. |
| DeepEP / DeepGEMM | Conceptual correspondence; dependency tracing remains | This reading concerns replay of routing choices; communication/matrix-compute kernels are different downstream layers. They cannot be drawn as direct dependencies of this repository merely because the models are all called MoE. |

<a id="动手实验"></a>

## Hands-on experiments

**Pending: alignment checks for routing recording and replay.** In phase one, read the existing `tests/fast/utils/test_replay_base.py` and `tests/fast/backends/training_utils/test_replay_data.py`, and use synthetic top-k/scores, variable-length samples, and padding to verify pointers and sharding. Keep scores/tokens fixed and change only replay indices, batch packing, and CP/SP configuration; observe shapes, layer streams, valid-token indices, overlap, and corresponding gradient positions.

Start with host-side data-layout verification. The actual `Replay` uses pinned memory and `torch.cuda.current_device()`, so the full wrapper path requires matching PyTorch/CUDA; the entire chain cannot broadly be described as CPU-runnable. In phase two, compare logprob differences, routing mismatches, GPU memory, and per-step time with replay enabled/disabled on a small MoE. Actual results: **neither phase executed**; no speedup estimate is provided.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Trace from the model provider to the router-patch installation point and verify the hook for each supported model.
- [ ] Trace how SGLang exports `rollout_routed_experts` and verify token shift and masks.
- [ ] Study weight versions and precision separately; the replay mechanism alone is not complete numerical consistency.
