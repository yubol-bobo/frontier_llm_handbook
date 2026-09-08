<a id="prime-rl异步-rollout-为什么还需要严格的入队出队与版本边界"></a>

# Prime RL: Why asynchronous rollout still needs strict enqueue, dequeue, and version boundaries

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/PrimeIntellect-ai/prime-rl @ `04a61d3b75c3c99f263b2c133e822f998909adf7`  
Verification scope: Read the implementation, related configurations, and declarations in the pinned local snapshot; did not install upstream dependencies or start a model, sandbox, or training run. The control-flow assessments in this note come from static reading and do not represent a reproduction of the authors' performance or benchmarks.

<a id="核心问题"></a>

## Core question

Fully asynchronous does not mean that generating more is always better. Completed rollouts still waiting in the training queue continue to age; canceled requests must also contribute to group-completion counts. This reading traces `Episode → group → TrainingSample → microbatch → sender`, focusing on how throughput control affects the actual training distribution.

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

- [Result routing, version gating, packing, and sending](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/orchestrator.py#L513-L685); local: [`src/prime_rl/orchestrator/orchestrator.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/orchestrator.py).
- [Group-completion counting, stale-sample removal, and sample conversion](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/train_sink.py#L131-L325); local: [`src/prime_rl/orchestrator/train_sink.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/train_sink.py).
- [The specific GRPO credit implementation in this snapshot](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/algo/grpo.py#L16-L45); local: [`src/prime_rl/orchestrator/algo/grpo.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/algo/grpo.py).
- [Local Verifiers dependency and TorchTitan pin](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/pyproject.toml#L20-L250); local: [`pyproject.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/pyproject.toml).

Also actually read `src/prime_rl/orchestrator/train_source.py:17–85`: it selects tasks by environment ratio, builds a curriculum for each environment, and saves RNG and individual curriculum states. The scheduler returns Episode, GroupCancellation, or DispatchFailure, and `Orchestrator.main_loop()` explicitly distinguishes these events. Actual Episodes are first written to the `all` artifact on arrival, then passed to the train/eval sink.

`TrainSink.add()` first performs rollout-local algorithm work, then collects Episodes by group. Group completion counts arrived Episodes + failure + cancellation count, avoiding waits for members that will never return. After group completion, it runs `finalize_group()` / curriculum admission, calls `trace_to_samples()`, sets temperature and loss routing, and puts samples into the pending batch by trace.

Before each batch is formed, `_drop_stale()` removes overly old data; it also checks again when a group first enters the queue. `finalize_train_batch()` waits until inference has applied at least the specified policy version, then packs, calls `sender.send()`, and advances the step. This observes the sending interface without claiming an audit of the full trainer-receive, gradient, and broadcast chain.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

- **Implementation fact:** When collecting a batch for step and training policy v(step−1), staleness of a sample from v(k) is interpreted as `(step−1)−k`. Early dispatcher cancellation only saves compute; sink removal of queued traces is the final boundary for training-data staleness. Frozen-source data has no live-policy span and does not age under this rule.
- **Implementation fact:** Stale cancellation also invalidates already returned Episodes in the same group and skips the curriculum: pipeline expiration is not an incorrect task answer. This preserves the distinction between environment-difficulty statistics and system events.
- **Implementation fact:** Top-p/top-k truncated sampling requires a sampling mask; if absent, `process_group():286–293` raises an error because the normalized probability space of rollout logprobs must match the trainer's. Identical token IDs do not establish probability consistency.
- **Implementation fact:** In this snapshot, `GRPOAlgorithm.score_group()` defaults to `reward - group mean` and does not divide by group standard deviation in that function; optional length shaping is also explicitly computed. This differs from verl's default GRPO standard-deviation normalization, so algorithm comparisons must check implementation and configuration.
- **Reading assessment:** Dropping overly old data, rejecting data with no training signal, and restricting sending versions reduce apparent rollout utilization but preserve training semantics. Useful metrics should include all/effective, stale drops, valid tokens, and wall time together.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Object | Relationship type | Evidence and boundaries in this reading |
|---|---|---|
| Verifiers | Direct dependency + Git submodule | `pyproject.toml:30,244` and actual `import verifiers.v1`; `.gitmodules` points to the official repository. |
| TorchTitan | Platform-specific direct dependency/component reuse | Linux dependency at `pyproject.toml:60`, git pin at 250; also located `trainer/utils.py:16`, which imports its `clip_grad_norm_`. This does not prove that the entire Prime trainer is built on TorchTitan's trainer. |
| Harbor | Optional taskset capability through Verifiers | `verifiers[harbor]` is explicitly declared; see the Verifiers note for specific task adaptation, which does not replace version verification. |
| verl / slime / Miles | Conceptual correspondence | All involve rollout logprobs, masks, and policy updates; this reading does not treat them as required dependencies of Prime RL. |

**Reproduction boundary:** `git ls-files --stage deps/verifiers` yields gitlink `828488fffe31aa3332b9d1bd4bd9ee320e375cf1`, while this learning repository's independent Verifiers clone is `27bbd216df0af719a43705866b2cf6139bcc95de`. These snapshots are not a tested combination. This shallow clone did not initialize submodules; running a repository example cannot assume complete dependencies.

<a id="动手实验"></a>

## Hands-on experiments

**Pending: samples that become stale after completion.** Construct two rollout groups with identical rewards/tokens; return the first immediately and delay the second, or enqueue it before delaying training. Control group size, curriculum, and sampling model, changing only version advancement and delay. Observe stale drops, pending-token counts, group-cancellation counts, effective cohort, and sending step. Expected: already queued samples can also expire; stale events do not enter the curriculum's task results. Actual result: **not executed**.

First test the state machine using synthetic Episodes and matching dependencies, without model inference. A complete closed-loop experiment requires a small model on Linux/CUDA; official basic examples commonly allocate two GPUs to trainer/inference, so this clone cannot be treated as an already prepared single-GPU training environment.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Trace `trace_to_samples()` to token loss routing and confirm subtrace and sampling-mask conversion.
- [ ] Trace sender → trainer → weight watcher to verify version atomicity and checkpoint-restoration boundaries.
- [ ] Before reproducing experiments, initialize the authors' pinned submodules instead of substituting the adjacent independent clone's HEAD.
