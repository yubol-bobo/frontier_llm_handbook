<a id="open-instruct训练与-rollout-的概率差异如何进入-grpo-loss"></a>

# Open Instruct: How training–rollout probability differences enter GRPO loss

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/allenai/open-instruct @ `ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5`  
Verification scope: The local GRPO learner step, loss/correction, weight synchronization, and OLMo-core SFT entry point; did not run Ray, vLLM, DeepSpeed, Beaker, or model training.

Snapshot limitation: 4 upstream test-data files remain LFS pointers; their payloads were not fetched. See [this audit](https://github.com/yubol-bobo/frontier_llm_handbook/blob/main/notes/sessions/2026-09-08-training.md). Having the source does not mean that original test data and a runtime environment are ready.

<a id="核心问题"></a>

## Core question

When the sampling and training engines differ, or rollout versions even lag behind the learner, why is a single `exp(new_logprob-old_logprob)` insufficient? How do system scheduling, masking, and loss normalization jointly change training semantics?

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

1. In [grpo_fast.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_fast.py), [`PolicyTrainerRayProcess.step`, L586–765](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_fast.py#L586-L765): obtain dataloader batch → optional sequence-parallel split → ref / old logprobs → new logprobs → ratio / rho correction → policy + KL loss → masked normalization → backward / optimizer boundary.
2. In [grpo_utils.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_utils.py), [`compute_rho_correction` through `compute_grpo_loss`, L410–538](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_utils.py#L410-L538): specifically distinguishes train-old/infer-old correction from the new/old optimization ratio.
3. Same file, [`perform_weight_sync`, L745–782](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_utils.py#L745-L782): pause actor manager → wait for broadcast → optionally wait for inner engine RPC → wake up → resume actor in finally. The caller at `grpo_fast.py:1459–1517` triggers synchronization through an event/thread carrying the target step; this was also actually read.
4. [pyproject.toml](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/pyproject.toml), [L7–74](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/pyproject.toml#L7-L74), provides evidence of package dependencies and the OLMo-core commit pin. Also read L16–105 of [olmo_core_finetune.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/olmo_core_finetune.py): SFT directly imports OLMo-core and converts HF data into token IDs / labels mask / metadata shards.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

**The two ratios are different.** The optimization ratio in `step` is `exp(new_logprobs - old_logprobs)`; rho is `exp(old_train_logprobs - vllm_logprobs)`. The former measures the current policy update, while the latter addresses training/generation distribution differences. `use_vllm_logprobs` can take the old baseline directly from generator probabilities; otherwise it comes from the local model's cache / first-pass detach. With multiple minibatches, old probabilities are computed in advance to prevent the baseline moving with optimizer updates.

**Rho correction is a stabilization choice with bias tradeoffs.** The implementation first limits logprob differences to [-10,10], can aggregate by token or sequence, and can clamp or zero the weights of out-of-range tokens. When disabled by default, it returns all-1 weights but still retains a rho histogram. Enabling it changes which samples contribute gradients; it cannot be claimed to restore strict on-policy training at no cost. Asynchronous version lag, kernel numerical differences, and the actual sampling distribution must be distinguished separately.

**Different policy objectives place gradients differently.** The DAPO branch takes the larger loss of the unclipped / clipped surrogates; the CISPO branch detaches and clips the ratio, then multiplies by new logprob. These are more than two threshold names. rho_weights multiply the policy loss; KL is computed separately, and the caller finally combines `pg_loss + beta*kl`. This reading did not establish exact equivalence between these branches and every detail of a particular paper.

**Valid tokens determine distributed weighting.** The learner computes a token denominator for the accumulation group, and the response mask excludes nonresponse tokens. The code then multiplies by `world_size // sequence_parallel_size`; the comment explains that this counteracts DeepSpeed's averaging across ranks to match token normalization. Simply averaging rank losses changes training weights as sample lengths vary.

**Weight synchronization is a system protocol.** `perform_weight_sync` resumes the actor manager in finally to prevent exceptions from leaving rollout permanently stopped; `inflight_updates=True` skips waiting for inner RPCs. The function name alone does not establish identical update atomicity across all modes. The caller produces broadcast refs before entering this function, so this reading does not reduce the code to a synchronization barrier with absolutely no in-flight requests.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Relationship | Evidence and boundaries |
|---|---|
| OLMo-core: direct dependency with a pinned version | `pyproject.toml:21,74` pins `fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d`; the lab's independently cloned HEAD is `92870a33…`. Source reading can connect knowledge, but execution must follow the dependency pin rather than substitute the adjacent folder. |
| vLLM / DeepSpeed / Ray: direct dependencies | Explicitly listed in the manifest; the learner's model.backward/step and remote actors/engines divide training from generation. This study did not clone vLLM / DeepSpeed and cannot claim a completed low-level audit. |
| verl / slime / prime-rl: conceptual correspondence | Rollout versions, probability correction, valid-token normalization, and weight transfer are worth comparing across these projects; this does not imply this repo depends on those frameworks. |
| SGLang: conceptual correspondence | It is another rollout/inference engine choice in the learning tree. This GRPO path actually passes vLLM engines, so it cannot be described as an existing SGLang integration. |

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** Start with a numerical loss experiment that does not require a generative model.

- Inputs: manually constructed B×T new, old, and vLLM logprobs, positive/negative advantages, and response masks; include empty padding, short/long responses, and extreme differences.
- Controls: fix tensors and the old baseline; independently switch rho on/off, token/sequence selection, and DAPO/CISPO.
- Metrics: per-token weights, drop/clip fraction, loss, and gradients with respect to new_logprob; confirm that padding does not change the valid-token denominator.
- Expected: weights are all 1 with rho disabled; rho is 1 when old=vLLM (masks/thresholds still need checking); CISPO has no gradient through its ratio path, with gradients coming from the multiplied new logprob.
- Compute: CPU tensor-level tests in an isolated environment suffice. The current module's import chain includes research dependencies, so first identify the minimal dependencies or extract a custom test harness. Full rollout-latency experiments additionally require GPUs and are outside this execution scope.
- Actual result: none. No cluster was requested, image built, or model published.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Trace the dataloader / ActorManager: how advantages are normalized by prompt group, and how version lag is recorded and discarded.
- [ ] Follow `broadcast_to_vllm` to check concurrency constraints between optimizer steps and parameter transfer, especially inflight_updates.
- [ ] Validate a small model using the upstream pinned environment before discussing replacement of OLMo-core / inference-engine versions.
