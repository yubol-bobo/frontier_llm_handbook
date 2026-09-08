<a id="03从-base-model-到可靠-agent后训练rl-与评估闭环"></a>

# 03 | From a Base Model to a Reliable Agent: The Post-training, RL, and Evaluation Loop

[Back to the end-to-end process](00-end-to-end.md) · [Distributed execution](02-distributed-pretraining-operations.md) · [Marin 535B case study](04-marin-535b-live-case-study.md)

Updated: 2026-09-08. Following the pretraining chapters, this article discusses training and delivery as very large models acquire usable behavior. **Public facts** come from official reports and local pinned source snapshots; **engineering synthesis** is design guidance organized from that evidence, not the actual internal process of every lab. Complete data mixtures, rewards, training budgets, algorithm combinations, and release thresholds for closed frontier models are usually unknown; these gaps are not filled with inventions here. This article contains no local training or measured benchmark results.

<a id="1-接收-base-checkpoint然后按能力缺口选择分支"></a>

## 1. Receive the Base Checkpoint, Then Choose Branches According to Capability Gaps

The post-training team should receive more than weights: tokenizer, existing interaction protocol, model and parallelism configurations, training-data versions, base evaluations, long-context behavior, and numerical state. Diagnose knowledge coverage, instruction and tool protocols, search/reasoning strategies, and environment interaction separately, then use controlled experiments to choose continued pretraining, midtraining, demonstration learning, or RL. SFT/RL can also improve capabilities and strategies; they are not merely ways to express existing knowledge. A single task failure is also insufficient to establish that a capability is absent. Long-context extension may precede post-training or be interleaved with later stages.

OLMo 3 publishes a model flow covering base, midtraining, long context, and different post-training branches. It provides a complete path to study, not the only sequence for very large models. [OLMo 3 official description](https://allenai.org/blog/olmo3)

| Branch | Learning signal and main purpose | Key decisions |
|---|---|---|
| SFT / cold start | Supervised token loss on demonstration responses or successful trajectories; establish instructions, tool protocols, output structure, and initial exploration capabilities | Example quality, coverage, and generation provenance; which roles/tokens receive gradients |
| Preference optimization | Pairwise/ranked preferences or a reward model trained from preferences; shape usefulness, style, and behavioral constraints | Direct methods such as DPO or reward-model RL; annotator and judge biases |
| RLVR / agent RL | Answers, tests, file artifacts, task states, or combined scores verifiable after execution | Trustworthiness of the environment, whether task difficulty yields a signal, and affordability |
| Rejection sampling / distillation | Generate multiple candidates, verify and select them for SFT, or use teacher distributions | Preserve provenance and selection rules; avoid imitating only formatting or incorrect reasoning |

These branches may alternate, mix, or be omitted. They must not be described as a universal DPO-before-RL sequence. DeepSeek-R1-Zero demonstrates an RL path without prior SFT; R1 uses cold-start data and multiple training stages to address behavioral problems. This establishes that different choices exist, not that any base model can stably enter RL directly. [DeepSeek-R1 technical report](https://arxiv.org/abs/2501.12948)

In practice, first freeze a small set of development tasks and compare starting points: do they follow tool schemas, succeed occasionally, or frequently mix languages and break formats? If all attempts fail, improve the data, harness, or cold start first; if all succeed, raise the difficulty. More rounds of RL are not a substitute for diagnosis.

<a id="2-把-reward-和环境当成训练系统的组成部分"></a>

## 2. Treat Rewards and Environments as Parts of the Training System

An agent training sample is “initial task state → actions and observations → final state → scoring evidence.” Version task data, container images, tool schemas, dependencies and network conditions, time/resource budgets, verifiers, and reward aggregation rules. These conditions matter as much as model weights for reproduction.

First break capability goals into checkable events: whether code passes independent tests, a spreadsheet has correct formulas and values, a document contains required content and can be parsed, or an operation reaches the target state. Prefer objective checks; when a model judge is needed, fix the rubric, judge version, sample audits, and disagreement handling. Correct formatting can be an auxiliary signal but cannot replace task completion, or the model may optimize the scoring proxy.

Record task failures, environment faults, tool timeouts, model truncation, and exhausted budgets separately. Directly assigning negative rewards for environment faults may teach the model to avoid affected tasks; deleting all such cases creates selection bias instead. Verifiers' `open → step → close` lifecycle separates normal stopping from failure, manages runtime ownership, and preserves opportunities for artifacts and scoring during closure, providing a concrete boundary design to study. [Verifiers lifecycle implementation](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/rollout.py#L178-L540)

**Reward-hacking checks** should actively attempt to modify test/scoring files, read answers, forge logs or artifacts, exploit judge formatting preferences, and earn points with meaningless long output. Isolate hidden checks from the agent-writable environment; audit high-reward samples with a different verifier or human review. High reward is only the value measured by the training system; its correspondence to actual capability still needs validation.

<a id="3-harness-改变训练分布轨迹必须保留-token-来源"></a>

## 3. Harnesses Change the Training Distribution; Trajectories Must Preserve Token Provenance

The harness determines what the model sees each turn: system prompts, tool selection, error feedback, search results, context trimming, memory, subagent scheduling, retries, and stopping. Changing the harness changes the environment and observation distribution. Higher scores from the same model under a larger budget or stronger tools do not automatically imply greater capability in its parameters.

Align these conditions between training and deployment and retain experiments on their differences. A rigorous training trajectory should link at least:

```text
task_id / group_id / trajectory_id / branch_id
Actual input tokens, sampled output tokens, per-token rollout logprobs
Loss masks, sampling parameters, and necessary sampling masks
Policy/model version, tool and environment versions
Termination reason, reward components, scoring artifacts, time and resource consumption
```

Tool observations can serve as context but usually receive no action loss; initial prompts, template tokens, and padding also need correct handling. The response region may contain tool observations from multiple turns, so it cannot all be marked as model output. Retokenizing the final conversation may change the originally sampled tokens, misaligning logprobs and loss.

Compaction and subagent forks are especially likely to break this correspondence. slime checks prefixes, realigns or splits Samples, masks regions that lose reliable provenance, and avoids training a shared generated prefix repeatedly. This snapshot gives each Sample the full outcome reward instead of simply dividing it by the branch count. Actual normalization still requires following advantages and the loss reducer. [slime trajectory implementation](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/agent/trajectory.py#L141-L501)

<a id="4-一次-grpo-更新具体做什么"></a>

## 4. What Does One GRPO Update Actually Do?

The following is a **synchronous teaching sketch with outcome rewards**, not any repository's complete default recipe:

1. Draw prompts from the training-task distribution and sample G trajectories per prompt with a fixed behavior policy `μ`; execute tools and verifiers, retaining original tokens/logprobs and rewards.
2. Compute relative advantages by task/group identity, for example `A_i = (R_i − mean(R_group)) / (std(R_group)+ε)`; some implementations only subtract the mean. Fix normalization, length shaping, and the treatment of equal-reward groups.
3. The learner recomputes `log πθ` on the same token contexts; in this synchronous sketch, `r_it = exp(log πθ(a_it|h_it) − log μ(a_it|h_it))`. Preserve the old probability baseline rather than allowing it to drift with minibatch optimizer steps.
4. Apply the clipped surrogate only to valid action tokens, for example maximizing `min(r*A, clip(r, 1−ε_low, 1+ε_high)*A)`. Fix clipping bounds according to the recipe, with optional reference KL, entropy terms, or other regularizers. Specify normalization by token, trajectory, or group, because it changes the weights of long and short responses.
5. Perform backward, gradient synchronization/clipping, and the optimizer step. Record entropy, KL, clip/drop fractions, reward distributions, and valid target tokens. Save a checkpoint, then provide the complete new policy version to rollout.

Group identities are more reliable than batch-row positions throughout these steps. Negative advantages, padding, empty samples, and forks should all have small numerical tests. An increase in training-step reward is far from sufficient: independent holdout capability and gains at equal cost are also required.

<a id="5-从同步扩展到异步减少等待同时控制数据变旧"></a>

## 5. From Synchronous to Asynchronous: Reduce Waiting While Controlling Aging Data

Long-horizon agents vary greatly in execution time, and environments also consume substantial CPU, memory, network, and sandbox quotas. Synchronous execution waits for the slowest trajectory in a group/batch; asynchronous execution overlaps sampling, environment execution, scoring, data buffering, learning, and weight broadcasts. Scale these resources separately instead of only adding GPUs.

Completed trajectories continue aging in the queue. In-flight trajectories may span multiple weight updates as well; define whether they finish under a fixed policy or allow updates between segments, retaining recoverable version/probability provenance. Record the oldest version and staleness distribution, limit in-flight requests and queue length, and define admission, cancellation, dropping, and backpressure rules.

Prime RL explicitly separates early cancellation from cleanup after enqueueing: the former saves compute, while the stale sweep before dispatch constrains the data actually used for training. It also does not treat stale cancellation as an incorrect task answer when updating the curriculum. This is where algorithmic statistics meet scheduling semantics. [Prime RL queue/staleness implementation](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/train_sink.py#L131-L325)

Off-policy correction can use importance weighting, truncation, rejection sampling, and other methods, but finite trajectories, mixed versions, and clipping cannot be claimed to restore strict on-policy behavior without cost. Asynchronous comparisons must keep the training objective fixed and report valid samples/second, wall time, drop rate, and independent evaluation, rather than comparing rollout tokens/s alone.

<a id="6-策略变旧与数值不一致是两个问题"></a>

## 6. Policy Staleness and Numerical Inconsistency Are Different Problems

**Version differences** arise when the learner has updated weights while data came from an older policy. **Numerical/execution differences** can occur even with the same weight version: training and inference engines, low-precision quantization, kernels, MoE top-k routing, temperature, or truncated sampling distributions differ. Diagnose them separately instead of calling every logprob discrepancy staleness.

Open Instruct separates the update ratio `π_current / π_old_train` from the correction ratio `π_old_train / π_rollout`, with optional clamping or masking of the latter. In actual systems, that second ratio may contain both version and engine differences and must be interpreted according to the operating mode. Record both ratios and sampling configuration to locate the source. Truncation and filtering also change the training distribution. [Open Instruct correction/loss](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_utils.py#L410-L538)

MoE adds a discrete execution path: tiny score changes can select different experts. Miles replay saves routing indices and replays them after alignment by microbatch, CP/SP, and layer layout. It controls discrete selection rather than freezing all probabilities or gradients. Routing replay, precision choices, atomic weight updates, and sampling masks are separate checks; none alone proves consistency across the entire chain. [Miles replay implementation](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/utils/replay_base.py#L14-L241)

<a id="7-评估与发布门槛贯穿全程"></a>

## 7. Evaluation and Release Gates Apply Throughout

Before training, establish three sets: a development set for tuning, a less frequently accessed holdout, and final release evaluation. Splits by source/time/task family often test generalization better than randomly splitting questions. Beyond text matching, check code from the same repository, question variants, answer leakage, and whether generation teachers or tool retrieval accessed test material. Deduplication, semantic similarity search, and manual audits each have blind spots; one decontamination pass cannot prove complete absence of contamination.

Tülu 3 explicitly separates development from unseen evaluation and publishes its decontamination and evaluation process. Preserve this distinction: repeatedly changing rewards, harnesses, or models based on the holdout gradually turns it into a development set. [Tülu 3 evaluation design](https://allenai.org/blog/tulu-3-technical)

Retain the following gates at each stage handoff and candidate release. Thresholds depend on application risk and statistical uncertainty, rather than universal numbers invented here:

- **Capability regressions:** evaluate reasoning, code, knowledge, languages, instruction following, long context, and tool use separately, instead of one aggregate score alone.
- **Behavior and safety regressions:** completion of reasonable requests, over-refusal, dangerous behavior, unauthorized tool use, privacy leakage, prompt injection, and error recovery; the reward model itself also needs auditing.
- **System regressions:** timeouts, crashes, GPU memory, first-token/end-to-end latency, tokens and cost per task; compare identical tasks, tools, budgets, and sampling configurations.
- **Evidence completeness:** preserve model/tokenizer/harness/configuration/image/data identities, raw trajectories, final files, verifier logs, failed samples, and randomness settings. Estimate uncertainty with paired tasks and repeated sampling; report pass@k and higher-budget results separately.

High scores on valid samples alone hide environment failures, while average success alone hides regressions in a language or task. Release gates should route a candidate back to the concrete responsible layer: data, reward, harness, system, or numerics.

<a id="8-用四个源码接口把知识接起来"></a>

## 8. Connect the Concepts Through Four Source Interfaces

| Interface | Upstream artifact → downstream input | First reading location |
|---|---|---|
| Verifiers → Prime RL | Task/Harness/Runtime execution produces Episode/Trace → group credit, filtering, TrainingSample | `verifiers/v1/rollout.py` → `orchestrator/train_sink.py` |
| APEX/Harbor → SkyRL | Trial verifier reward + TITO tokens/masks/logprobs → GeneratorOutput → trainer | `tito_harbor_generator.py`; SkyRL is outside this library's dependency scope |
| slime agent → train backend | Sampled TurnRecord → branched Sample → rollout batch → actor update | `agent/trajectory.py` → `train.py` / Megatron actor |
| verl/Miles → compute and sampling backends | Batch with mask/group/logprob and parameter version → engine computation/weight update | verl `ray_trainer.py`; Miles `actor.py` / replay data |

These are software interfaces or comparison entry points, not four frameworks already compatible with one another. Prime RL pins a Verifiers submodule different from this library's independent HEAD; APEX pins Harbor 0.21.0, and SkyRL uses a release-checkout path on the authors' machine. **APEX explicitly states that its training data has not been open-sourced because of licensing issues.** Public evaluation traces cannot reconstruct the original training set. At present, one can study the recipe and substitute authorized, reproducible task data, but cannot claim reproduction of the original run. [APEX data statement](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/README.md#L47-L68)

<a id="9-部署是下一轮数据工程的起点"></a>

## 9. Deployment Starts the Next Round of Data Engineering

Before release, retest the actual serving configuration: quantization, chat templates, tool schemas, context budgets, routing, and rate limits can all change behavior. Use rollback-capable versions, staged traffic, and explicit stopping conditions, preserving the mapping between deployed models and offline candidates.

Send failures, cost anomalies, and user feedback collected with authorization into offline analysis, attributing causes before deciding to update data, harnesses, rewards, or models. User feedback is not inherently ground truth, and production logs should not automatically enter training without permission, privacy, and contamination checks. The complete deliverable is traceable weights, an operating protocol, evaluation evidence, and a rollback mechanism; the next improvement cycle begins with this evidence.
