<a id="verl一个-rollout-batch-如何变成正确分组的策略更新"></a>

# verl: How does a rollout batch become a correctly grouped policy update?

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/verl-project/verl @ `7cb65014d3a6c84f59458367df768999e4f36c67`  
Verification scope: Read the implementation, related configurations, and declarations in the pinned local snapshot; did not install upstream dependencies or start a model, sandbox, or training run. The control-flow assessments in this note come from static reading and do not represent a reproduction of the authors' performance or benchmarks.

<a id="核心问题"></a>

## Core question

“Supports GRPO” hides two kinds of work: keeping multiple answers to the same question correctly grouped, and connecting distributed rollout, logprob recomputation, training, and weight synchronization into a loop. This first reading follows one batch through `RayPPOTrainer.fit()` without expanding every backend at once.

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

- [Generation, grouping, advantage computation, and weight synchronization](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/ray_trainer.py#L1481-L1716); local: [`verl/trainer/ppo/ray_trainer.py`](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/ray_trainer.py).
- [Actual mathematical implementation of GRPO outcome advantage](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/core_algos.py#L267-L331); local: [`verl/trainer/ppo/core_algos.py`](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/core_algos.py).
- [Actual imports and interface boundaries of the Megatron engine](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/workers/engine/megatron/transformer_impl.py#L15-L84); local: [`verl/workers/engine/megatron/transformer_impl.py`](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/workers/engine/megatron/transformer_impl.py).
- [Optional dependency declarations for SGLang / vLLM / mcore](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/setup.py#L50-L77); local: [`setup.py`](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/setup.py).

1. Before training, `fit()` restores a checkpoint and synchronizes rollout weights (1405–1430). It assigns each original prompt a `uid`, then performs interleaved repeat according to `rollout.n` (1481–1491).
2. `async_rollout_manager.generate_sequences()` returns results; inference replicas are then put to sleep, and the results merge with a training batch repeated in the same way (1513–1544). Here, async characterizes the called rollout manager; the name alone does not make this `fit()` a fully barrier-free pipeline.
3. Optional token-count balancing reorders the batch; group membership depends on `uid`, not row position. Rewards, old-policy logprobs, reference logprobs, and critic values are added to `DataProto` according to configuration.
4. `compute_advantage()` executes on the driver; its GRPO branch passes `uid` to `core_algos.compute_grpo_outcome_advantage()`. The latter sums per-token rewards into response scores, computes group means/standard deviations by uid, and broadcasts the same response advantage back to valid response tokens.
5. After an optional critic update, the actor is updated; checkpointing and training-to-inference weight synchronization are explicit steps (1662–1716). Through its own implementation, the Megatron engine imports capabilities such as parallel state and pipeline forward/backward from Megatron Core.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

- **Implementation fact:** Within-group advantage is not computed by subtracting one overall batch mean. It aggregates by `uid`; GRPO's `norm_adv_by_std_in_grpo` determines whether to divide by within-group standard deviation. Singleton groups receive special treatment with mean=0 and std=1, so ordinary multisample-group intuition does not explain them.
- **Implementation fact:** The source near `_balance_batch` explicitly states that reordering does not affect uid-based advantage computation but may affect loss through minibatch composition. Data order and learning behavior are not entirely unrelated.
- **Implementation fact:** Recomputing old logprobs and directly using rollout logprobs are alternative branches. Decoupled mode can first apply importance-sampling / rejection correction; the bypass path compares different quantities in its metrics (1647–1660). The presence of this mechanism does not establish correct correction for arbitrary precision/inference-backend combinations.
- **Reading assessment:** The most valuable lesson in this repository is the boundary between control flow and the compute engine. Tracing where every batch field comes from explains system correctness more effectively than first memorizing algorithm names.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Object | Relationship type | Evidence and boundaries in this reading |
|---|---|---|
| Megatron-LM | Direct code dependency of an optional training backend | `transformer_impl.py:23–25` imports `megatron.core`; this does not mean every verl job uses Megatron. |
| SGLang | Optional rollout backend | `setup.py:57–61,73` declares the SGLang extra; this snapshot pins 0.5.8. The latest independent SGLang clone cannot simply be substituted with a claim of compatibility. |
| vLLM | Optional rollout backend, external project | `setup.py:55,72`; outside the scope of this study's 19 independent repository clones. |
| prime-rl / slime | Conceptual correspondence | All address the boundary from sampled data to policy updates; this reading found no need to draw them as direct dependencies of verl. |

<a id="动手实验"></a>

## Hands-on experiments

**Pending: group-identity and ordering experiment.** Inputs are three prompts, four manually specified response scores per question, and a response mask with padding. First keep uids fixed and shuffle rows, then deliberately use incorrect uids; compare advantages after restoring the original order. Control rewards, group size, and masks, changing only permutation and identifiers; observe per-response advantages, group means, and padding positions. Expected: with correct uids, reordering leaves the same response's advantage unchanged, while incorrect uids change its comparison group; this is a hypothesis awaiting verification. First run the actual function on CPU with matching PyTorch/NumPy versions; no model/GPU is needed. Actual result in this study: **not executed**.

A later GPU experiment should compare valid tokens per rank, step time, and minibatch loss with batch balancing enabled/disabled, without extrapolating CPU mathematics into system-throughput conclusions.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Read `rollout_corr_helper.py` and diagram clear definitions of rollout / old / current policy probabilities.
- [ ] Trace `_update_actor()` into the selected engine's `train_batch` to verify masks, temperature, and loss denominator.
- [ ] Build a separate environment for one selected recipe and follow its version constraints; only L1 is complete so far.
