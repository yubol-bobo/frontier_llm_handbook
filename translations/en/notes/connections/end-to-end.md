<a id="把训练研发连接起来的五个产物契约"></a>

# Five artifact contracts connecting training research and development

Date: 2026-09-08. Nature: engineering synthesis supported by pinned source and primary reports; these repositories were not run together. Entry point: [the full training process](../../handbook/00-end-to-end.md).

The [first lesson](../../lessons/01-one-token-to-update.md) introduces the same connection in the course: valid-token loss (M02) → data/masks (M03) → distributed reduction (M05) → RL trajectory objectives (M11/M13). [Experiment 002](../../experiments/002-token-weighted-loss/README.md) verified a CPU mathematical counterexample involving group weighting; it did not add evidence of software integration or GPU execution. See [ROADMAP](../../ROADMAP.md) and the relationship overview for learning prerequisites and software relationships, respectively.

Repository relationships can be drawn by imports or by what each layer must deliver to the next. The table below takes the latter approach: a shared problem or transferable contract is not an existing adapter.

| Handoff | Minimum identity / semantics to retain | Why it affects correctness | Evidence entry points |
|---|---|---|---|
| Raw data → training batch | Provenance, processing / tokenizer version, shard, sample position, packing / masks, actual mixture | Changing document boundaries or loss positions changes the objective; dataset names alone do not enable reproduction | [Data design](../../handbook/01-data-model-pretraining-design.md), [Marin](../repositories/marin.md), [OLMo-core](../repositories/olmo-core.md) |
| Research recipe → distributed update | Architecture, initialization, optimizer, token horizon, parallel groups, valid-token denominator | Independent sample counts must not be multiplied again by CP / TP / EP; local means cannot be arbitrarily averaged again | [Distributed training](../../handbook/02-distributed-pretraining-operations.md), [TorchTitan](../repositories/torchtitan.md), [Megatron](../repositories/megatron-lm.md) |
| update → checkpoint → export | Parameter and optimizer ownership, master / random / data states, complete-commit conditions, export layout | Loading inference weights does not imply training can resume; resumability does not imply bitwise agreement across topologies | [Recovery contract](../../handbook/02-distributed-pretraining-operations.md), [Marin real-world case](../../handbook/04-marin-535b-live-case-study.md) |
| harness / environment → learner | Observation and action tokens, logprobs, masks, branches, environment / verifier / policy versions | Reconstructing text can lose the original sampled tokens; reward dimensions and their aggregation change the learning objective | [Detailed RL chapter](../../handbook/03-posttraining-agent-rl-evaluation.md), [RL connections](rl.md), [Reward experiment](../../experiments/001-harbor-reward-contract/README.md) |
| learner → rollout / deployment | Complete weight version, tokenizer, template, routing / precision / sampling, cache-version policy | Mixed old/new versions, probability inconsistency, or reuse of stale KV can change behavior; each requires separate diagnosis | [Miles](../repositories/miles.md), [SGLang](../repositories/sglang.md), [GPU connections](infra.md) |

Together these contracts make experiments traceable: data and run identity span pretraining, while task and behavior-policy identity span RL. Marin's artifact DAG and Pi's session logs both handle identity and history, but record different objects; this note does not therefore claim that they are integrated.

<a id="一个贯穿多层的取舍长上下文--moe"></a>

## A multilayer tradeoff: long context × MoE

Long context first changes sample length and document organization, then changes the number of independent sequences in a batch, token distributions received by experts, GPU memory, and communication. In post-training, it also changes rollout duration, KV cache, queue staleness, and failure-recovery cost. Expanding a context window therefore requires joint completion checks across data, training, communication, inference, and environments, beyond changing positional-encoding parameters.

Source / recipe evidence: [SmolLM stage configurations](../repositories/smollm.md), [Marin 535B decision chain](../../handbook/04-marin-535b-live-case-study.md), [MoE communication](infra.md), [Asynchronous RL](rl.md). These are complementary concrete examples; numerical thresholds from one implementation cannot be applied to all projects.

<a id="验证顺序"></a>

## Verification order

First verify identity, shapes, masks, and numerical behavior at one handoff; then verify equivalent semantics across processes / devices; finally add asynchrony, failures, and performance pressure. Runnable combinations depend on the versions actually pinned by each recipe, rather than the HEADs of this lab's independent clones; see the [version connection table](../../REPO_RELATIONSHIPS.md#版本连接表). The present addition is an evidence structure, with no new GPU or complete RL experiment results.
