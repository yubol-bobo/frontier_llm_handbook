<a id="slime多轮-agent-的文本记录怎样成为可训练的-token-轨迹"></a>

# slime: How do multiturn agent text records become trainable token trajectories?

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/THUDM/slime @ `4c193f1f37509cca70f0e88807a9305b70f63f4e`  
Verification scope: Read the implementation, related configurations, and declarations in the pinned local snapshot; did not install upstream dependencies or start a model, sandbox, or training run. The control-flow assessments in this note come from static reading and do not represent a reproduction of the authors' performance or benchmarks.

<a id="核心问题"></a>

## Core question

Agents rewrite messages, compact context, and call subagents. Training cannot simply retokenize the final chat text: loss and rollout logprobs must correspond to the tokens actually sampled by the model at the time. How does this snapshot distinguish retaining something as context from allowing backpropagation through it?

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

- [Basic training loop and resource switching](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/train.py#L9-L94); local: [`train.py`](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/train.py).
- [Trajectory realignment, branching, shared-prefix deduplication, and export](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/agent/trajectory.py#L141-L501); local: [`slime/agent/trajectory.py`](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/agent/trajectory.py).
- [SGLang service import and process launch](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/sglang_utils/sglang_engine.py#L1-L64); local: [`slime/backends/sglang_utils/sglang_engine.py`](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/sglang_utils/sglang_engine.py).
- [Connections among Megatron, loss, and weight updater](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/actor.py#L8-L49); local: [`slime/backends/megatron_utils/actor.py`](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/backends/megatron_utils/actor.py).

The basic system chain is: `train.py` creates placement groups → creates a rollout manager with SGLang engines → creates actor/critic → initial weight synchronization → `generate` → `async_train` → save/release resources → synchronize weights again. This file waits for each rollout; other fully async entry points were not examined here, so framework marketing cannot establish an absence of synchronization points in this entry point.

The trajectory chain has its own clear boundaries: an adapter submits `TurnRecord(prompt_ids, output_ids, output_log_probs, finish_reason)` → `record_turn()` attaches records to a session tree by message → `get_trajectory()` traverses root-to-leaf chains → `_split_chain_into_builders()` checks token prefixes → `_SampleBuilder.to_sample()` exports `Sample`.

The message tree determines message correspondence, while the token builder determines exact token continuity; these are not a single kind of conversation merging. The initial prompt remains in the complete tokens, while `loss_mask` and `rollout_log_probs` export only the response region; the lengths of these two arrays must not be compared directly with the full token length.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

- **Implementation fact:** CLEAN appends only the new prompt suffix (mask=0) and new output (mask=1). REALIGN replaces the recent response region according to the new prompt with mask=0; FORK starts a new builder. Classification depends on the divergence position and `fork_threshold`. The actual condition compares the length of the new `output_ids`; this must not be independently changed to drift length based only on a “short drift” description.
- **Implementation fact:** `_split_chain_into_builders():469–476` deduplicates with `response_trained`: a shared generated prefix is trained on the first leaf and retained only as mask=0 context on later leaves. `get_trajectory():339–340` gives every exported Sample the **full outcome reward**. This snapshot does not simply use reward/K; “reward split” in older webpages or comments cannot replace the current implementation.
- **Implementation fact:** REALIGN discards some training signal to preserve trustworthy sampling provenance; FORK increases the number of training samples. Builders with no trainable tokens are filtered out. The session is deleted after export, so a second call cannot be treated as a side-effect-free read.
- **Reading assessment:** The actual training-data format is more than a `(prompt, answer, reward)` triple: it includes token provenance, logprobs, masks, grouping, branch/sample identity, and stop reasons. This boundary is where harness engineering intersects with RL correctness.

Also actually read the test organization and token vocabulary at `tests/test_agent/test_trajectory_manager_branching.py:1–100`. It provides an entry point for public-API-driven tests and human-readable dumps, but tests were not executed here; the existence of a test file does not establish that the current snapshot passes all tests.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Object | Relationship type | Evidence and boundaries in this reading |
|---|---|---|
| SGLang | Direct backend integration | `sglang_engine.py:9,48–53` imports and starts the actual service; it was not started here. |
| Megatron-LM | Direct dependency of the training backend | `actor.py:11` imports Core, and 47–49 connects train and weight updater. |
| Miles | Project lineage | Miles' own README states that it is forked from slime; Miles is not treated as a runtime dependency of slime here. See `../connections/rl.md` for consolidated evidence. |
| APEX recipe | Conceptual correspondence | Both preserve sampled tokens / loss masks / logprobs; APEX pursues strictly append-only token state, while this file also handles branching/realignment after message rewriting. There is no evidence of a direct dependency. |

<a id="动手实验"></a>

## Hands-on experiments

**Pending: a shared-prefix and token-drift matrix.** Use fixed small-integer tokens to construct four two-turn conversations: exact prefix continuation, drift within the recent response, drift in an early prompt, and two leaves sharing a response. Change only prefix/token relationships and fix reward at 1; observe Sample count, per-token masks, how many times each generated segment is trained, and reward per Sample. Expected: shared segments are not trained repeatedly, regions that lose sampling provenance are masked, and branching does not automatically divide reward equally by sample count. First adapt the public API of the existing branching tests; CPU execution suffices, but dependency resolution and the pytest environment remain to be prepared. Actual result in this study: **not executed**.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Trace how the adapter obtains SGLang tokens/logprobs and confirm sampling parameters and the probability space of logprobs.
- [ ] Trace `rollout_id` / `group_index` into advantage computation and the loss reducer to verify actual normalization of branched samples.
- [ ] Compare weight versions and data queues in the fully async entry point; this note does not claim to cover that path.
