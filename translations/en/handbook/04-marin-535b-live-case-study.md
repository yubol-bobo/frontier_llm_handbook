<a id="04marin-535b一场仍在进行中的超大模型训练怎样作出真实决策"></a>

# 04｜Marin 535B: how real decisions are made during an ongoing very large model training run

[Back to the full process](00-end-to-end.md) · [Data and recipes](01-data-model-pretraining-design.md) · [Distributed execution](02-distributed-pretraining-operations.md) · [Post-training](03-posttraining-agent-rl-evaluation.md)

Review date: 2026-09-08. This chapter pins the local Marin snapshot to `5e2436d0f61462983003bd8b6eaef8235ecab78c`, tracing the model, data, launcher, training step, and recovery code in `experiments/grug/moe_hero_ep` and comparing them with official announcements and public issues. **Source facts** describe behavior at this snapshot; **team statements/plans** retain their publication dates; **engineering interpretation** explains the design implications. No upstream programs were run, training data accessed, jobs submitted, or training performance measured.

<a id="1-先对齐时间启动公告未来承诺和当前代码是三种证据"></a>

## 1. Align the timeline first: launch announcements, future commitments, and current code are three kinds of evidence

The September 2 announcement confirms that training had started: 535B total parameters, approximately 23B active parameters, a target of approximately 18T tokens, and an estimated pretraining completion date of December 1, with Paloma performance predictions registered in advance. It is not a completion report and does not establish final agent capabilities. [Official September 2 announcement](https://openathena.ai/blog/huang-foundation-marin-535b-training-run/)

David Hall's September 3 article describes the run at that time and reports that ragged all-to-all reduced dropping and improved throughput. This is the author's account of that period. In the pinned code, however, the production default has temporarily reverted to pooled-wave because ragged hangs after the watch step at 11 racks. **A performance optimization can have succeeded earlier and later been rolled back for stability.** A static checkout cannot replace live job status either. [September 3 launch note](https://openathena.ai/blog/marin-535b-launch-note/), [rollback record in the corresponding README](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/README.md#L19-L37)

The main thread of this case study is to constrain expectations for the large run with smaller experiments, make data and optimizer choices into calculable configurations, address communication, memory, and recovery on real hardware, and finally use independent checkpoint branches to establish the post-training pipeline early.

<a id="2-scaling-ladder预测损失也预测什么时候该介入"></a>

## 2. Scaling ladder: predict loss, and predict when to intervene

In issue #8435, created on August 18, the team explains that an earlier ladder exposed gradient norms increasing with the token horizon, prompting a logit z-loss correction; existing small-run trajectories also helped distinguish normal gradient changes from anomalies. The citation here refers to the editable issue body visible at review time, without attributing every paragraph to the issue's creation date. [Public decision record](https://github.com/marin-community/marin/issues/8435)

The first source-code chain is:

```text
build_ladder_run(size)
  → _ladder_model(size)
  → batch = 1024 × number of racks
  → steps ≈ 791 × active_parameters / (batch × 4096)
  → Generate optimizer configuration from token budget, width, and batch
  → ArtifactStep(run=run_grug, deps=data artifact + validation set)
```

The five widths are d768, d1024, d1536, d2048, and d6144, using 1, 2, 6, 11, and 11 racks, respectively. Tokens per rack are held fixed to try to retain the large run's routing pressure, rather than merely reducing parameters. The production configuration specifies 390,251 steps, each containing 11,264 sequences of length 4K. Static multiplication gives approximately **18.005T tokens**; this is a planned budget, not data already processed. Smaller configurations evaluate at approximately every 5% of progress; the large configuration evaluates every 3,000 steps. [Model and budget construction in the launcher](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/launch_scaling_ladder.py#L105-L222)

The learning rate is not simply copied from a small model. `MoeHeuristic` computes MuonH／Adam learning rates, epsilon, and beta2 from total tokens, width, and tokens per step; its comments explicitly describe an empirical fit from earlier sweeps. This is a transfer rule to validate, not a law that can be extrapolated to arbitrary architectures. [Optimizer budget mapping](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/heuristic.py#L24-L83)

Another detail affects reproduction: the README summarizes the ladder as using the same transport, but `_ladder_model` at this snapshot explicitly selects ragged for smaller configurations, while the hero configuration inherits the reverted pooled-wave default. Running the launcher today is therefore not automatically equivalent to reproducing that historical set of prediction experiments. **Engineering interpretation:** Every fitted curve should be tied to the commit, backend, data, and evaluation semantics that produced it; “the same script name” is insufficient to define the same experiment.

This also determines how predictions should be used: retain the entire validation-loss trajectory fitted from smaller configurations as a reference, together with experimental error and configuration differences. When the large run deviates, first locate whether the deviation immediately follows a data switch, restart, routing change, or throughput change, then decide whether to roll back model state or repair the system. A training-loss increase at one step alone is insufficient to judge the architectural choice a failure. Publishing an endpoint prediction prevents arbitrary explanations after the fact; retaining the intermediate trajectory helps decisions while compute is still being consumed.

<a id="3-数据配方候选库采样预算实际训练量必须分开"></a>

## 3. Data recipe: distinguish candidate inventory, sampling budget, and actual training volume

The second chain is `harrier_mix_2026_08_18_data_config → _two_phase_data_config → LmDataConfig`. It uses a versioned data artifact organized into 200 cells: 40 clusters × 5 quality tiers. The JSON explicitly stores available tokens in each cell and two-stage weights; code checks the tokenizer, store URI, whether both weight sets cover every cell and each sums to 1, and whether cumulative repeated exposure exceeds 8 epochs. Beyond parameter names, what matters are these conditions that reject invalid configurations. [Harrier recipe and validation](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.py#L36-L155)

Three scales coexist here: candidate inventory in the local JSON totals approximately **23.106T**; the mixture policy defines a target budget of **15T + 3.75T = 18.75T**; the production launcher's step budget is approximately **18.005T**. These are not interchangeable model specifications. Small experiments enable simulated epoching when analytical training FLOPs do not exceed `1e23`, using a limited experimental budget to simulate repeated exposure under a large budget; above that threshold, the original mixture is used. It is therefore incorrect to say the hero run also repeatedly simulates a small dataset.

The two-stage transition is derived from approximately 80% of total steps and aligned to mixture blocks; the `15T` constant alone does not mean the production run mechanically switches at its 15T-th token. Validation sets enter the same configuration with zero training weight, alongside checks for naming collisions. Zero weight prevents those components from being sampled, but cannot by itself prove that training text contains no near-duplicates of evaluation questions. [Stage boundaries and zero-weight validation sets](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe/launch_datakit_moe_mix.py#L295-L360)

**Engineering interpretation:** Changing training duration can change the learning-rate horizon, data-stage boundaries, and repeated exposure. Changing `num_steps` is a recipe change; recalculate these quantities before deciding whether the original scaling prediction still applies.

<a id="4-ep-的选择通信显存路由和数值状态一起设计"></a>

## 4. Choosing EP: design communication, memory, routing, and numerical state together

The production model has 48 layers, width 6144, 384 routed experts with 8 selected per token, plus 2 shared experts; expert width and latent width are both 3072. Each rack uses 64 GB200 GPUs to form an expert mesh: 16 four-GPU workers, with each GPU holding 6 routed experts; replication across 11 racks uses `replica_dcn`. NVL72 is the hardware system name; this recipe uses a 64-GPU EP mesh within it, so multiplying 72 by the rack count does not give the number of training processes.

Current pooled-wave places data for each destination into a fixed pool divided into 3 waves; expert assignments exceeding capacity are dropped, with both sender and receiver capacity factors set to 1.15. This gives communication and buffers bounded shapes, at the cost of changing the expert computation actually received by some tokens. Ragged attempts to adapt transmission to actual assignments, but increases complexity in the runtime and collective-communication paths. [Actual model and transport defaults](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/heuristic.py#L86-L130)

The third chain can be traced directly in the training step:

```text
Previous step's pending_qb_betas → update router bias
  → BF16 compute forward + CE／logit z-loss → gradients
  → Load optimizer state and FP32 master → optimizer.update
  → Produce BF16 device weights; return master／optimizer state to pinned host memory
  → Save router-balancing information for the next step
```

The default is `z_loss_weight=1e-4`; QB uses a global histogram to provide a stop-gradient routing bias for the next step. The total / sender / receiver drop-fraction metrics in `_drop_metrics` use batch × sequence × top-k × layers as their denominator; the additional `receiver_drop_fraction_of_received` first subtracts sender drops from that denominator. These drop fractions are therefore **fractions of expert assignments**, not “the number of complete tokens removed from the corpus.” [Training step, metric definitions, and state updates](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/train.py#L713-L882), [training defaults](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/hero_recipe.py#L31-L97)

Host offload is not free device memory: training steps still need to move state, so throughput and peak memory must be measured together. This recipe retains an FP32 host master for pooled-wave; switching to another weight layout also changes checkpoint structure. `template_for_candidate_layout` checks the manifest: a checkpoint with a master can restore the authoritative FP32 master as device parameters; a checkpoint without one is explicitly rejected when restoring into a mode that requires a master, because the code does not support synthesizing one on demand. **Engineering interpretation:** A rollback plan for changing kernels must also check whether checkpoints can be restored in the reverse direction. [Actual recovery-layout checks](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/train.py#L148-L190)

<a id="5-启动以后保存实验身份区分故障与正常重试"></a>

## 5. After launch: preserve experiment identity and distinguish failures from normal retries

Preproduction diagnostics also have explicit boundaries: the one-rack test retains production data, watch behavior, and process layout, preserving local load at 16 sequences per GPU. Synthetic data can isolate compute and memory behavior, but does not cover real storage reads. The README specifically notes that a one-rack trace excludes replication communication and global histogram reduction across eleven racks. Thus, “one rack runs” passes only a local gate and cannot prove full-cluster stability; the hang after watch in this case shows the independent value of the final layer of integration validation. [Coverage boundaries of diagnostics and profiling](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/README.md#L101-L155)

`trigger_hero.sh` records the full commit, dirty state, run ID, and coordinator job before submitting to Iris. The current script also explicitly derives a new run ID from the full checkpoint at step 58,014 to continue training with gate/router weight decay enabled. It retains its own output directory and is a traceable recipe branch; this is not the early-cooldown RL branch discussed later and does not prove that the latter is complete. [Launch and branching script](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/trigger_hero.sh#L14-L55)

For the production configuration, the launcher retains permanent checkpoints every 6,000 steps and configures hourly rolling temporary checkpoints. The latter are saved near the region, with only one retained; recovery searches multiple directories. If an explicitly specified parent checkpoint cannot be loaded, the run should fail instead of quietly restarting from scratch. The hourly interval trades multi-TB save costs against training that must be redone; it does not guarantee recovery from every failure within one hour and excludes downtime for recovery itself. A stuck step and a process making no progress for a long time also have different watchdog timeouts. [Checkpoint, watchdog, and evaluation configuration](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/experiments/grug/moe_hero_ep/launch_scaling_ladder.py#L245-L330)

Operations documentation separates signals for excessive gradients, consecutive skipped updates, token drops, router entropy/bias, MFU, evaluation degradation, and lost telemetry. For example, the current hero rules use thresholds such as drops above 7% and at least 3 skipped steps in 15 minutes; these are operating rules for a specific experiment, not industry-wide training standards. Particularly useful is the separation of `run_id` and `execution_uid`: retries retain the logical run ID, while a new execution gets a new UID. Aggregating only by run can add skipped steps from the previous execution into the current one, or compare evaluations repeated before and after recovery. Alerts also cannot all be interpreted as a worsening model: data switches, configuration changes, and recovery itself can shift the loss baseline. [Alert conditions and attempt isolation](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/docs/ops/hero-run-health-alerts.md#L38-L115)

Evaluation has another easily missed distinction: the current launcher reports dropless held-out evaluation and disables the capacity-limited current-model evaluation path; a configuration comment says the latter breaks the ragged training step at this model scale. Training may therefore still drop some expert assignments while reported loss comes from dropless evaluation. **Engineering interpretation:** Fixed evaluation semantics help compare model weights, but do not establish production-inference latency or routing behavior. When handing off downstream, test the target inference backend separately and preserve differences before and after export; language-modeling loss predictions also cannot be directly converted into code-execution success rates or multi-turn task completion rates.

<a id="6-长上下文与后训练在主跑结束前购买信息"></a>

## 6. Long context and post-training: buy information before the main run ends

The issue's plan starts at 4K to increase sequence diversity in each batch and improve expert balance; long context may significantly increase dropping. The team proposes taking a separate branch after approximately 10–20 days for 1–2 days of early cooldown, to try RL and context extension ahead of time; subsequent expansion to 8K, 65K, and 262K depends on experimental results. This is a conditional plan; this chapter has not verified that the branch has run or met its targets. [Context and early-cooldown plan](https://github.com/marin-community/marin/issues/8435)

**Engineering interpretation:** Context length changes more than attention compute and positional encodings; it also changes the composition of tasks in each batch, expert load, and communication pressure. An independent branch lets the post-training team test model export, inference consistency, rollouts, and reward environments early; even poor results can feed problems back to the still-training main run, instead of waiting months to discover interface or capability gaps. The September 3 announcement confirms that the team was already working on Grug MoE support in vLLM, checkpoint export, and inference consistency, but this does not mean a complete 535B post-training recipe was public or successful. [Official account of inference handoff work](https://openathena.ai/blog/marin-535b-launch-note/)

<a id="7-这个案例目前能教什么哪些结果仍然没有答案"></a>

## 7. What this case can teach now, and which outcomes remain unanswered

What can be learned directly is an auditable chain of decisions: bind predictions to experiment identity; let budgets drive data and optimizers; evaluate routing quality alongside throughput; make communication optimizations recoverable; allow controlled branches within one logical training effort; and make evaluation and alerts respect each execution's boundaries. These are closer to actual training work than a final benchmark table.

As of this chapter's review, these materials do not support announcing completion of 18T tokens, achievement of the preregistered loss, attainment of the target long context, or frontier agent results; live token counts, total costs, final safety gates, and the release date have not been verified either. Current source defaults, configurations actually loaded by a particular job, historical prediction experiments, and the final model must be connected through their own provenance and result artifacts. When reading the next update, continue tracing that evidence chain instead of replacing old facts with a new announcement.
