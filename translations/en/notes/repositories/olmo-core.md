<a id="olmo-core一个-optimizer-step-如何跨-microbatch-与分布式同步保持语义"></a>

# OLMo-core: Preserving optimizer-step semantics across microbatches and distributed synchronization

Date: 2026-09-08  
Stage: L1 initial source reading  
Source: https://github.com/allenai/OLMo-core @ `92870a33c3fee060d57c3faec52b0b369ece85a6`  
Verification scope: The local official Olmo 3 recipe and relevant methods in Trainer and TransformerTrainModule; did not instantiate models or execute numerical or GPU tests.

<a id="核心问题"></a>

## Core question

Why does a training framework split into experiment configuration → lifecycle Trainer → model training module? How does this separation coordinate token normalization, gradient accumulation, communication, and evaluation?

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

1. In the [Olmo 3 7B pretraining recipe](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py), [`build_config`, L40–111](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py#L40-L111): assembles the tokenizer, data mix, model, train module, and TrainerConfig, then passes the builder to `main` at the end of the file. This reading did not further audit distributed-launch details in `main`.
2. [trainer.py](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/trainer.py), [`_fit_epoch`, L1466–1537](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/trainer.py#L1466-L1537): reshuffle → pre_epoch → per-batch bookkeeping / pre_step → train_batch → pre_optim_step → optim_step → zero_grads → post hooks → termination check.
3. [transformer/train_module.py](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/train_module/transformer/train_module.py), [`train_batch`, L345–441](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/train_module/transformer/train_module.py#L345-L441): generates labels, counts tokens contributing to loss, splits microbatches, runs forward/backward for each, and immediately all-reduces loss for SkipStepOptimizer when needed.
4. Same file, [`optim_step` and `_train_microbatch_context`, L514–584](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/train_module/transformer/train_module.py#L514-L584): clip gradient → scheduler → optimizer; the final microbatch triggers different synchronization behavior in HSDP / DDP.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

**Token counts must be distinguished from sample counts.** The official configuration expresses `rank_microbatch_size=2*8192` in tokens. The implementation uses `rank_microbatch_size // seq_len` to obtain samples per microbatch and explicitly raises an error if the token budget is smaller than one sequence. Microbatch loss reduction is sum, sharing the valid-token denominator of the entire rank batch. Simply adding per-microbatch mean losses changes weighting when the last microbatch is smaller or masking proportions differ.

**Gradient accumulation also changes communication timing.** `_train_microbatch_context` sets `set_requires_all_reduce(True)` only for the final microbatch under HSDP; FSDP also uses `set_is_last_backward` to manage pending reduction / prefetch, while nonfinal DDP microbatches enter `no_sync()`. This confirms delayed all-reduce in HSDP. It does not establish that earlier microbatches perform no communication, because reduce-scatter, parameter fetching, and other operations may still occur.

**Trainer should not need every layer's training details.** `_fit_epoch` manages step/token/FLOP counters, callbacks, and stopping conditions; the train module manages loss, gradients, and the model. The same lifecycle can therefore support different training modules. The tradeoff is that debugging a behavior requires tracing configuration, callbacks, and modules, rather than reading only `Trainer.fit()`.

**An actual recipe differs from library capabilities.** This 7B recipe selects HSDP, BF16 parameters / FP32 reduction, no embedding weight decay, z-loss=1e-5, synchronous checkpoints, and `Float8Config(enabled=False)`. Thus, library support for Float8 does not mean this released model used Float8. The recipe also records the transition from the originally planned 5T tokens to a longer schedule, providing a clue to training history.

**A statistical tradeoff acknowledged by the source.** After `instance_mask` masks instances, the implementation still includes their tokens in the loss denominator. A comment explains that this lowers loss in exchange for equal rank contributions and avoiding expensive distributed processing. This is a known code-level tradeoff, not a finding from tests performed here. When comparing runs, observe masked labels / instances alongside CE curves.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Relationship | Evidence and boundaries |
|---|---|
| Open Instruct: actual direct dependency | Its `pyproject.toml:21,74` declares and pins `ai2-olmo-core`, and its SFT code imports this library; see the [Open Instruct note](open-instruct.md) for pinned evidence. However, its pinned commit is `fa6c5014…`, different from this directory's HEAD; direct interoperability between the two HEADs has not been verified. |
| TorchTitan: conceptual correspondence | Both separate training loops from parallelism-application logic, with implementations that manage synchronization by microbatch; neither directly depends on the other. |
| SmolLM: conceptual correspondence in training recipes | Both explicitly handle embedding weight decay and effective token budgets. This identifies mechanisms worth comparing, without inferring algorithmic origin from similar configurations. |

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** Use a tiny transformer to check whether microbatch splitting preserves gradients for a fixed batch.

- Inputs: fixed token IDs and two batches with some label=-100, with dropout disabled; keep the global/rank batch unchanged.
- Controls: initial model parameters, optimizer, precision, data, and effective loss denominator; change only rank_microbatch_size.
- Metrics: full gradients or comparable gradient summaries before the first step, parameter differences after the update, CE, and peak GPU memory; add communication traces for multi-GPU runs.
- Expected: mathematically, gradients should approximate those of the same batch; floating-point accumulation order can change the last bits, so bitwise identity is not promised in advance. Cases containing instance_mask should follow the source's special denominator rule.
- Compute: first validate token normalization with a tiny model on one GPU; verifying HSDP's delayed all-reduce requires an appropriate multi-GPU topology. Actual GPU memory was not measured.
- Actual result: none; neither `pytest` nor training was run because this deliverable is source notes, not an upstream code modification.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Trace how `SkipStepOptimizer` turns latest_loss / latest_grad_norm into skip decisions, and whether scheduler / token counters continue advancing when a step is skipped.
- [ ] Read the build order in `script_utils.ExperimentConfig` and check paired constraints on data cursors and optimizer state during checkpoint restoration.
- [ ] Compare identity and recovery semantics in the official legacy Numpy data-mix API and current composable data API.
