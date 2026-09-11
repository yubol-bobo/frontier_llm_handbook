<a id="后训练与-agentm10m15"></a>

# Post-training and Agents: M10–M15

2026-09-08. For learners with basic programming skills and no frontier-lab experience, this sequence follows the foundation-model, data, pretraining, and serving modules, developing the ability to explain, implement, and audit a complete chain.

The recommended order is **M10 → M11 → M12 → M13 → M14 integrated completion checks → M15**. Begin using M14's splits, baselines, and failure categories after M03 and revisit them at every stage. The harness-only reading in M12 may begin after M01, but training integration requires M11. Running Pi first and then connecting asynchronous RL cannot replace probability fundamentals and a synchronous closed loop.

Read the three groups of core material in each section in order; optional reading is not required for completion. Source code uses pinned snapshots, with commits and line numbers in the notes. New external entry points were verified online but have not been cloned or installed. **All exercises await execution; workload figures consistently mean planned human effort for core reading plus CPU exercises, not measurements. GPU extensions require additional time.** CPU work prioritizes your own small programs/synthetic data; full upstream import chains are not guaranteed to support CPUs. GPU work requires checking memory, platform, and dependency locks, with download and training wall-clock time counted separately.

<a id="m10"></a>

<a id="m10sft偏好优化与-reward-model先理解学习信号"></a>

## M10 | SFT, Preference Optimization, and Reward Models: Understand the Learning Signal First

**Prerequisites and diagnostic for skipping material:** M02, M03, M04. Mark the SFT tokens in a multi-turn conversation, explain input/label shifting, and explain why “A beats B” does not prove A is correct. If you cannot do this independently, revisit the foundations; if you can, shorten the introductory reading but still submit the audit and objective-function experiments.

**Topics:** choose continued pretraining, midtraining, supervised fine-tuning (SFT), preference optimization, or reinforcement learning (RL) according to capability gaps, without a fixed requirement to use DPO before RL. Study chat templates, assistant-only masks, packing, and truncation; matching task conditions for chosen/rejected responses, annotation noise, and length preferences; differences among scalar reward models, rule-based verifiers, process scores, and monitoring metrics; reference policies and the limits of pairwise accuracy/reward margins.

**Core reading, in order:**

1. Read the first two sections of the [post-training handbook](../handbook/03-posttraining-agent-rl-evaluation.md), the data flow in the [original InstructGPT paper](https://arxiv.org/abs/2203.02155), and the derivation in the [original DPO paper](https://arxiv.org/abs/2305.18290). Draw the branches for a separate reward model and direct preference optimization; do not present historical recipes as current internal facts about closed models.
2. Read the [Open Instruct notes](../notes/repositories/open-instruct.md), trace the token/mask/metadata cache in [olmo_core_finetune.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/olmo_core_finetune.py#L71), then trace `_get_batch_logps` and `dpo_loss` in [dpo_utils.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/dpo_utils.py#L525), comparing sums with averages.
3. Read the single-output head and loss at lines 351–357 of [reward_modeling.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/reward_modeling.py#L351). Compare the source and purpose of DPO's detached implicit rewards.

**Optional reading:** public OLMo 3 and Tülu recipes, comparing stage order.

**Derivations and diagrams:** write `L_SFT=-Σmask·logpθ/Σmask`, noting the choice of token averaging. Derive pairwise loss from `P(chosen>rejected)=sigmoid(r_chosen-r_rejected)`, then substitute DPO policy/reference log-probability differences, marking beta and frozen quantities.

**Exercise A: Data to labels.** On CPU, create 12 short conversations covering multiple turns, tools, excessive length, and empty answers; output token labels and truncation counts. Swap the prompt in one preference pair and check whether you can detect it. Use integer fixtures if no tokenizer is available. The GPU extension only overfits these samples to inspect target probabilities; it does not count as a generalization result.

**Exercise B: Compare objective functions.** On CPU, compute three losses through hand calculation/finite differences/automatic differentiation. Fix probabilities and masks, vary padding, length, labels, and beta, and save the gradients. For the GPU extension, compare SFT with one preference branch on the same split, recording budget, length, evaluation, and differences across repeated runs. The preference branch need not win.

**Self-assessed completion and misconceptions:** why does adding a constant to every reward leave the pairwise loss above unchanged? Is changing sequence logprob aggregation from a sum to a mean merely numerical scaling? Why might lower SFT loss and higher preference accuracy still accompany declining factuality? “DPO has no explicit reward model” does not mean it has no assumptions about preference data.

**Deliverables:** `m10/data-audit.md`, a visual label table, three independent small loss implementations, and comparison records. **Workload:** 12–20 hours for the CPU core; another 8–16 hours of experiments and debugging for the GPU extension. **Next:** enter M11 once you can explain the source of every gradient, while fixing evaluation splits according to M14.

<a id="m11"></a>

<a id="m11rl-数学与同步最小闭环先让一次更新正确"></a>

## M11 | RL Mathematics and a Minimal Synchronous Loop: Make One Update Correct First

**Prerequisites and diagnostic for skipping material:** M10 and M09 serving concepts. Calculate the expected reward and gradient of a two-action policy by hand, and explain why a nondifferentiable verifier cannot be directly backpropagated through. If you fail, revisit probability/the chain rule; even experienced RL learners must explain the response mask and prompt group.

**Topics:** states, actions, policies, trajectories, returns; score-function gradients, baselines, advantages, critics/GAE; PPO's old policy, clipping, reference KL, and repeated updates; GRPO groups for the same prompt, outcome rewards, and within-group baselines. RLVR describes the reward source and does not guarantee stability. Make equal-reward groups, single-sample groups, length normalization, and truncation explicit.

**Core reading, in order:**

1. Read the surrogate objective in the [original PPO paper](https://arxiv.org/abs/1707.06347), then GRPO in the [original DeepSeekMath paper](https://arxiv.org/abs/2402.03300). First identify the sampling distribution for each expectation; the old policy in the formula is not the fixed reference.
2. Read the [verl notes](../notes/repositories/verl.md), comparing with `compute_grpo_outcome_advantage` in [core_algos.py](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/core_algos.py#L268): grouping by uid, summing per-token rewards, optional standard-deviation normalization, and handling special groups. Correct your formula notes against the actual function behavior.
3. Follow the batch through `RayPPOTrainer.fit` in [ray_trainer.py](https://github.com/verl-project/verl/blob/7cb65014d3a6c84f59458367df768999e4f36c67/verl/trainer/ppo/ray_trainer.py#L1405), then read [slime/train.py](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/train.py). Mark the barriers between generation, scoring, logprob computation, updates, saving, and synchronization; an async function name does not prove a fully asynchronous algorithm.

**Optional reading:** [Prime RL's GRPO implementation](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/algo/grpo.py#L16) subtracts the group mean in this snapshot, providing a comparison of normalization conventions. Leave its asynchronous scheduling for later.

**Derivations and diagrams:** derive `∇E[R]=E[(R-b)Σ∇logπ]`, stating the conditions on the baseline. Write `Ai=(Ri-mean(R))/(std(R)+ε)`, noting that standard-deviation normalization can be disabled. Let `rit=exp(logπθ-logπold)` and plot `min(rit·Ai,clip(rit,1-ε,1+ε)·Ai)` for positive and negative advantages. Negate it for the loss, aggregate using the mask/denominator, and optionally add KL. Diagram the four roles: behavior, old, reference, and current.

**Exercise A: Groups and masks.** On CPU, create three prompts with four responses each, including all-zero rewards and unequal lengths. Shuffle rows while preserving uid, then assign incorrect uids. Check advantages, padding, and denominators, adding K=1 and a comparison with verl. The GPU extension compares valid target tokens and updates under batch balancing; do not extrapolate CPU results into throughput claims.

**Exercise B: A synchronous loop.** On CPU, write a small-vocabulary policy and execute the following on a short-sequence task: freeze the sampling policy → generate K responses per prompt → verifier → advantage → update → synchronize. Fix seeds, save failed trajectories, KL, and valid sample counts, and verify the gradient direction. For the GPU extension, substitute a small language model while preserving the protocol; leave multi-turn tools, asynchrony, and MoE for later.

**Self-assessed completion and misconceptions:** why must batch reordering preserve uid? Does zero advantage mean the model has solved the problem? Why must old logprobs remain fixed across repeated updates instead of moving with current parameters? If longer generation raises reward, does that reflect capability, budget, or a scoring loophole? Clipping does not strictly guarantee that the policy has not drifted.

**Deliverables:** `m11/one-update.md`, two formula diagrams, a CPU synchronous trainer, and per-batch audit logs. **Workload:** 18–30 hours for the core; another 12–24 hours for the GPU extension. **Next:** enter M12 once you can trace one response to one weight update.

<a id="m12"></a>

<a id="m12任务harnessverifier-与-trajectory把互动变成可信数据"></a>

## M12 | Tasks, Harnesses, Verifiers, and Trajectories: Turn Interaction into Trustworthy Data

**Prerequisites and diagnostic for skipping material:** M01 for reading; M11 for training integration. Explain why tool results affect actions but usually do not enter policy loss directly. You may skim the concepts only if you can distinguish text logs, requests, actual tokens, and scoring artifacts.

**Topics:** Task objectives, Harness decisions, Runtime resources, and Verifier evidence; lifecycles, ownership, budgets, timeouts/retries, tool permissions, artifacts, and offline scoring. Trajectories record task/group/branch IDs, model versions, actual token IDs, logprobs, training masks, termination reasons, and sampling parameters/masks. Compaction, forks, subagents, and rewriting may break prefix continuity.

**Core reading, in order:**

1. Read the [original ReAct paper](https://arxiv.org/abs/2210.03629) to establish the interaction concept, then the [Verifiers notes](../notes/repositories/verifiers.md). Trace the lifecycle in [v1/rollout.py](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/rollout.py#L178) and `score` in [v1/task.py](https://github.com/PrimeIntellect-ai/verifiers/blob/27bbd216df0af719a43705866b2cf6139bcc95de/verifiers/v1/task.py#L198). This layer is not an optimizer.
2. Read the [slime notes](../notes/repositories/slime.md) and `record_turn`, `_split_chain_into_builders`, and `to_sample` in [trajectory.py](https://github.com/THUDM/slime/blob/4c193f1f37509cca70f0e88807a9305b70f63f4e/slime/agent/trajectory.py#L283). The shared prefix is trained only once, and every Sample receives the full reward; the earlier reward/K explanation is incorrect.
3. Compare the [APEX notes](../notes/repositories/apex-agents-skyrl-recipe.md) with [agents/tito.py](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe/blob/8e7702f03b7464a36ab800a624fd911de0968a87/apex_agents_skyrl_recipe/agents/tito.py#L148): appended tokens, tool mask=0, and stop-token/template boundaries. Training data is not public and SkyRL is an external dependency; reading glue code does not amount to reproduction.

**Optional reading:** request-reconstruction constraints in the [DeepSeek Harness notes](../notes/repositories/deepseek-harness.md) and context projection in the [Pi notes](../notes/repositories/pi.md). At this point you have a basis for judging training-data correctness when comparing harnesses.

**Draw:** a two-lane diagram: Task → Harness → tools → Verifier above, and request tokens → sampled tokens → mask/logprob → TrainingSample below. For one compaction event, mark which old tokens serve only as new context and which still have reliable sampling provenance. Explain why retokenizing the final chat text cannot recover all probability information.

**Exercise A: A task without a model.** A CPU script agent modifies temporary text, while the scorer reads only the artifact. Inject normal execution, timeout, missing artifacts, and scorer failure, recording separate states instead of assigning every case zero. Shut down the runtime and rescore, listing missing evidence. The GPU extension substitutes a small model for the script while preserving the task and budget.

**Exercise B: Trajectory boundaries.** CPU integer fixtures cover append operations, tools, rewritten answers, changes to early prefixes, and shared branches; output tokens/masks/logprobs and training counts. The model extension checks special tokens across two turns with a real tokenizer. The service must provide probabilities matching sampled tokens; ordinary text output is insufficient to validate RL.

**Self-assessed completion and misconceptions:** why can the response region contain mask=0? Who should release a borrowed runtime? How should an offline judge report missing files from the live environment? Replayable text does not imply replayable token probabilities; a framework's sandbox option does not mean an agent cannot interfere with its verifier.

**Deliverables:** `m12/task-contract.md`, a status-classification table, trajectory fixtures, a data schema, and artifacts that can be rescored. **Workload:** 16–28 hours for the core; another 10–20 hours for real-model/sandbox extensions. **Next:** connect one task back into the M11 synchronous loop before studying concurrency in M13.


**Claude Code advanced unit:** After the initial Pi reading, study the [complete guide](../handbook/05-claude-code-harness.md) and [pinned SDK / historical snapshot](../notes/repositories/claude-agent-sdk.md) in the order execution loop → context → permissions → recovery → evaluation. Run the [CPU state-machine experiment](../experiments/harness-state-machine/README.md), deliver traces for denied calls, exhausted budgets, and unknown execution outcomes, then identify the fields required by M13 training trajectories.

**SoL-Pi efficiency unit:** After the initial Pi events/tools reading, use the [guide](../handbook/06-sol-pi-efficient-harnesses.md) and [source note](../notes/repositories/sol-pi.md) in the order Action Fusion → ObservationPack → Evidence-Preserving Reducer → Online Context Compact, then run [Experiment 004](../experiments/004-sol-pi-contracts/README.md). Understand evidence retention and rejection of invalid replacements, calculate cache costs, and design paired quality/cost evaluation in M14.

<a id="m13"></a>

<a id="m13异步-agent-rl-与一致性吞吐增长不能掩盖训练语义变化"></a>

## M13 | Asynchronous Agent RL and Consistency: Throughput Gains Must Not Hide Changes in Training Semantics

**Prerequisites and diagnostic for skipping material:** M11, M12, M08, M09. Draw a timeline in which the trainer updates twice before a slow task returns, identifying behavior/training versions and cache expiration. If you classify everything as floating-point error, revisit the synchronous loop; communication experience does not remove the need to validate loss ratios.

**Topics:** stragglers, pipelines, asynchrony; backpressure, group admission, cancellation/failure, queue aging, broadcasting, and recovery. Separate three kinds of discrepancy: version lag; kernel/precision differences at the same version; and sampling-distribution changes from temperature and top-p/top-k. Then study discrete MoE routing, packing/CP/SP, and routing replay. Metrics include valid target tokens/second and time to a quality target.

**Core reading, in order:**

1. Read the [Prime RL notes](../notes/repositories/prime-rl.md), following group completion and `_drop_stale` in [train_sink.py](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/train_sink.py#L182) to version gating and batch dispatch in [orchestrator.py](https://github.com/PrimeIntellect-ai/prime-rl/blob/04a61d3b75c3c99f263b2c133e822f998909adf7/src/prime_rl/orchestrator/orchestrator.py#L513). The source handles Episodes, cancellation, and dispatch failures separately.
2. Revisit the [Open Instruct notes](../notes/repositories/open-instruct.md), tracing `compute_rho_correction`, `compute_grpo_loss`, and `perform_weight_sync` in [grpo_utils.py](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_utils.py#L410). Mark the `new/old` update ratio separately from the `old_train/infer` correction; clipping, rejection, and truncation introduce bias tradeoffs.
3. Read the [Miles notes](../notes/repositories/miles.md), tracing [replay_base.py](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/utils/replay_base.py#L14), [replay_data.py](https://github.com/radixark/miles/blob/3de96596f16b9e6d23ba550c4c47de3479c9f14c/miles/backends/training_utils/replay_data.py#L30), and the replay stages in the training actor. Replay restores discrete indices while current scores still participate in computation; it does not freeze old logits.

**Optional reading:** learn actor–learner separation from the [original IMPALA paper](https://arxiv.org/abs/1802.01561), without treating its V-trace as evidence of implementation in the frameworks above. Use the DeepEP/DeepGEMM notes to supplement communication and matrix computation.

**Derivations and diagrams:** identify the sources of the four policies, and diagram version progression and queue aging. Write the numerator/denominator and support assumptions for importance weights, explaining why clipping extreme weights does not guarantee unbiasedness.

**Exercise A: Event simulator.** On CPU, simulate delays, failures, cancellations, versions, and queues for 20 tasks. Hold rewards fixed while changing scheduling; compare completed work, staleness rate, valid groups, and task distribution, checking whether slow tasks are excluded. The GPU extension uses real delays and compares time and quality at a fixed valid-target-token budget.

**Exercise B: Separate inconsistencies.** Use CPU probability tables/top-k fixtures to perturb versions, logprobs, support, and packing separately, validating the detectors. For the GPU extension, first measure probability differences with identical weights/tokens/sampling configuration, then change precision and replay independently; report routing overlap, masks, and loss. Miles Replay uses CUDA/pinned memory; a CPU substitute does not count as executing that implementation.

**Self-assessed completion and misconceptions:** why can a sample eligible at enqueue time still be stale when dequeued? Does cancellation count as task failure? Is the same version sufficient to guarantee probability agreement? Can routing replay repair an incorrect tokenizer or stale weights? How do you avoid leaving an actor permanently paused after a weight-sync failure? High GPU utilization does not imply high learning efficiency.

**Deliverables:** `m13/version-contract.md`, an event simulator, a probability-discrepancy matrix, and a fault-recovery design. **Workload:** 20–35 hours for the CPU core; another 20–40 hours or more for multi-GPU/MoE extensions. **Next:** include admission, versions, and valid-sample statistics in the M14 report instead of isolating them in a separate throughput plot.

<a id="m14"></a>

<a id="m14独立评估安全发布与反馈先定义怎样相信结果"></a>

## M14 | Independent Evaluation, Safety, Release, and Feedback: Define How to Trust Results First

**Prerequisites and diagnostic for skipping material:** begin after M03, revisit at each stage, and integrate at the end. Given “a 3-point improvement,” can you ask about samples, budget, variance, templates, versions, and use during tuning? If not, do A first; experienced learners must still inspect loopholes shared by reward and completion checks.

**Topics:** separate training, development, frozen holdout, and release-regression sets; split by source/template/repository and check near duplicates. Fix the model, tokenizer, harness, permissions, budget, retries, judge, and image; define the statistical unit for pass@1, cost, and latency. Record intervals, failures, missingness, and bias from repeated checkpoint selection; establish separate thresholds for factuality, success rate, robustness, safety, and cost.

**Core reading, in order:**

1. Read the multi-metric approach in the [original HELM paper](https://arxiv.org/abs/2211.09110) and evaluation/delivery in the [post-training handbook](../handbook/03-posttraining-agent-rl-evaluation.md). Write a coverage table, identifying untested capabilities; curriculum thresholds are not closed-lab standards.
2. External bridging resources: the [official lm-evaluation-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md), for configuring model-evaluation tasks; [Inspect Tasks](https://inspect.aisi.org.uk/tasks.html) and [Scorers](https://inspect.aisi.org.uk/scorers.html), for tracing dataset, solver, and scorer boundaries. Neither is yet included in the local source scope. Read the interfaces first; do not treat current web pages as the versions pinned by an author's project.
3. Read the [Harbor notes](../notes/repositories/harbor.md), tracing shared/separate verifiers in [trial.py](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/trial/trial.py) and reward parsing in [verifier.py](https://github.com/harbor-framework/harbor/blob/9a2e3b135cc8fb1e41131020f83370cb2f12ae93/src/harbor/verifier/verifier.py#L166), comparing with runtime-only scoring in Verifiers. Returning a valid number and correctly measuring a task are two separate guarantees.

**Optional reading:** the [Marin 535B case study](../handbook/04-marin-535b-live-case-study.md), to learn how predictions, versions, and operational alerts are separated. For agent safety, first study permissions and scoring isolation in local temporary environments; real external dangerous operations are unnecessary.

**Diagrams and derivations:** draw reward → independent evaluation → candidate → staged rollout → monitoring → review → new data version. Calculate success-rate intervals; for multiple samples of the same task, resample by task cluster and explain how correlation affects intervals. Record model/judge random seeds and repeat counts.

**Exercise A: Judge an evaluation.** On CPU, construct two sets of predictions, adding length advantages, duplicates, missing results, and a candidate that improves only on development data. Calculate overall, stratified, and budget-matched results, then explain a decision to release, seek more evidence, or reject. The GPU extension substitutes real checkpoint outputs while preserving the frozen protocol; do not enter scores for models that have not been run.

**Exercise B: Scoring and safety regression.** CPU scripts attempt to modify a visible reward file, write to the wrong artifact path, and return invalid content, testing independent scoring. Use temporary harmless fixtures for unauthorized requests, prompt injection, and leakage of secret placeholders. The GPU extension substitutes a model and also measures false refusals on normal tasks.

**Self-assessed completion and misconceptions:** can scores be recomputed from raw artifacts? How do missing scores enter the denominator? What risks arise when the judge and training reward share a model and prompt? After repeated inspection, is the holdout still suitable for final evaluation? An improved average cannot automatically offset a serious safety regression or establish safety in every untested domain.

**Deliverables:** `m14/eval-protocol.md`, a frozen task list, results and intervals, failure clusters, a draft model card, rollback conditions, and approval points for data feedback. Production feedback must remove unnecessary sensitive information, undergo provenance review, and be versioned; it must not automatically flow back into training. **Workload:** 16–28 hours for the core; another 10–20 hours for the GPU extension. **Next:** freeze the capstone success criteria before starting M15.

<a id="m15"></a>

<a id="m15综合-capstone交付可复查成果选择与资源相符的范围"></a>

## M15 | Integrated Capstone: Deliver Reviewable Results at a Scope Your Resources Support

**Prerequisites and diagnostic for skipping material:** relevant modules from M01–M14. Explain data, objectives, training/inference, recovery, and evaluation; write one page stating the question, hypothesis, minimal system, budget, and falsification conditions. If you have only a leaderboard or demo goal, return to M14. The capstone requires actual deliverables and cannot be skipped through reading alone.

**Topics:** connect tasks, data, tokens/masks, updates, export, evaluation, and recovery; distinguish dependencies from inspiration and record changes. With limited resources, prioritize validating one mechanism and state assumptions for unexecuted layers. Finding that a method fails, identifying a reproduction gap, or exposing an incorrect evaluation can also produce an evidence-backed result.

**Core reading, in order:**

1. Revisit the [end-to-end handbook](../handbook/00-end-to-end.md) and [Marin practical case study](../handbook/04-marin-535b-live-case-study.md). List the steps required for a complete large run but omitted from this smaller project, explaining why their omission does not undermine the current research question.
2. Read the [RL connection notes](../notes/connections/rl.md) and [end-to-end connections](../notes/connections/end-to-end.md), extracting actual commits, version locks, and data restrictions from the selected project notes. APEX's unpublished training data, Prime RL's Verifiers gitlink, and Open Instruct's OLMo-core pin cannot be supplied automatically by neighboring directories.
3. Return to the original papers from M10/M11/M14 and one selected source chain, building a “claim → formula/function → configuration → experiment → artifact” index. Choose at most one main training framework, one task runtime, and one evaluation entry point; complete an explainable system before adding combinations.

**Optional reading:** use other frameworks in this library only for an explicit comparison question, such as token masks, queue admission, or replay protocols. Integrating many frameworks at once is not itself a result.

**Choose one of three capstones:**

| Track | Required core work | CPU alternative and GPU extension | Claims explicitly ruled out |
|---|---|---|---|
| A: Evidence audit without a GPU | Dependencies/versions/data flow for one recipe; trace three claims to implementation, perform two falsification checks, and create one boundary fixture | CPU source review and state experiments; GPU verification of the chosen boundary | Static reading is not training reproduction; unknown data cannot be fabricated |
| B: Small-model end-to-end pipeline | Randomly initialize a small decoder → tiny pretraining run → retain the base → SFT → preference optimization or synchronous RL → export/evaluate/recover | CPU character model; GPU expansion of model and tokenizer. If starting from existing weights, label it “end-to-end post-training” | Toy tasks do not demonstrate frontier capabilities; loading weights is not pretraining from scratch |
| C: Improve a systems/data contract | Choose staleness, masks, scoring isolation, checkpoints, or replay; construct the problem, make a local fix, and run regression checks | CPU fixtures; measure actual overhead only on GPU, matching valid-data and quality budgets | Simulated time is not real throughput; faster does not necessarily mean better training |

**Draw:** every track submits a complete system diagram and a diagram of the scope actually verified by this project. Use dashed lines for unexecuted components, labeling them as external services, unavailable data, or conceptual assumptions.

**Exercise A: Design review.** Freeze the baseline, variables, metrics, failure categories, and costs before implementation. CPU synthetic tasks must explain their correspondence to real systems; GPU extensions must estimate memory and stopping conditions. Ask a reader to restate how to overturn the conclusion; without a peer, review it yourself a day later.

**Exercise B: Rechecking and fault rehearsal.** Recompute results in a clean output directory using only the records; inject an interruption, stale samples, or invalid rewards. For the GPU extension, check the recovered step, random state, optimizer, and weight version. Set error tolerances across precision/parallel configurations instead of assuming bitwise identity in advance.

**Self-assessed completion questions:** what evidence could overturn the conclusion? Does it still hold with different data/hardware? Which components were only read, and which were actually executed?

**Rubric, 100 points:** evidence/versions 25; mathematical/token/state correctness 25; baseline and control design 20; failures and extrapolation boundaries 15; reviewable artifacts 15. The planned passing threshold is 75 with more than half the points in every category. Fabricated execution or unexplained critical denominators or leakage prevents completion. This assesses a project, not qualification as a frontier expert, and does not require a positive gain.

**Deliverables:** `m15/project.md`, implementation, configuration/environment locks, raw results, recomputation scripts, failure cases, a decision log, and a demonstration. **Workload:** core reading plus CPU track A: 24–45 hours; B: 50–100 hours; C: 35–70 hours. Human effort, wall-clock time, and cost for GPU extensions are additional. **Next:** continue reproduction or contribution along an exposed problem, progressing from explaining boundaries to taking responsibility for a subsystem while remaining open to external review.
