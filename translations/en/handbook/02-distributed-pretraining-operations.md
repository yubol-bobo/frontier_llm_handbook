<a id="分布式预训练从集群验收到可恢复的优化器更新"></a>

# Distributed Pretraining: From Cluster Validation to Recoverable Optimizer Updates

[Back to the end-to-end process](00-end-to-end.md) · [Data and recipes](01-data-model-pretraining-design.md) · [Post-training and agent RL](03-posttraining-agent-rl-evaluation.md)

This chapter focuses on operating base-model training from random initialization: how to keep many GPUs performing correct updates to the same objective. Its evidence is the TorchTitan, Megatron-LM, DeepEP, and DeepGEMM source versions pinned in this repository, together with public NVIDIA documentation. **“Source fact” applies only to the cited version; “engineering synthesis” is a recommended validation process; “unverified” means this repository has not executed the corresponding GPU or cluster experiment.** The material below cannot reconstruct an unpublished very large model's complete recipe, and a library's capabilities do not establish that a particular model training run actually used them.

<a id="1-先验收集群再讨论训练吞吐"></a>

## 1. Validate the Cluster Before Discussing Training Throughput

**Engineering synthesis.** Before launch, freeze a run manifest: model and data versions, tokenizer, container and dependency versions, GPU architecture and memory, driver/CUDA/NCCL, NIC firmware, GPU–NIC–NUMA affinity, node and switched-network topology, storage paths, and account quotas. Identical software versions do not establish healthy physical links; several source HEADs compiling separately do not establish compatibility in combination.

Expand validation gradually: single-GPU memory and compute correctness → intra-node GPU interconnects → cross-node communication → the full workload reading data and writing checkpoints concurrently. Communication tests must check errors and sweep latency/bandwidth over the message sizes used in training. Cover TP/FSDP collectives, PP point-to-point communication, and MoE token dispatch separately. NCCL Tests can provide basic validation; custom paths such as DeepEP need dedicated tests. Its `busbw` is a metric converted through a collective-traffic model, not a direct measure of effective training bandwidth across every topology. [NCCL Tests metric definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

Storage validation should measure at least three things: sustained random/sequential shard reads, burst writes when every rank saves state, and the time to reconstruct training from those states. Check local-cache misses, shared-directory metadata pressure, background-upload backlog, and disk space. One large-file sequential-read benchmark cannot prove that the dataloader or distributed checkpointing will not stall training. Agree passing criteria before the pilot run rather than lowering them after one successful startup.

<a id="2-并行是数据与状态的布局不是一串可随意相乘的参数"></a>

## 2. Parallelism Is a Layout of Data and State, Not Arbitrarily Multipliable Parameters

| Dimension | What it partitions | Main costs and interactions |
|---|---|---|
| DP | Different samples whose gradients jointly update one logical model | Gradient synchronization; sharded DP also communicates parameter/optimizer states |
| TP | Tensor computation within the same layer | Frequent collectives; tied to sequence parallelism and operator layouts |
| PP | Different layers | Activations and their gradients cross stages; pipeline bubbles and stage imbalance |
| CP | The context of the same sample | Reduces the local long-sequence burden; attention exchanges information across shards |
| EP | Different experts | Moves tokens according to routing; expert loads and return combination determine tail latency |
| FSDP/distributed optimizer | Storage of parameters, gradients, and optimizer states | Trades communication for GPU memory; does not independently add samples |

**Source fact.** TorchTitan's current dense mesh satisfies `N = D_replicate × D_shard × CP × TP × PP`, and the DP degree for independent data batches is `D = D_replicate × D_shard`. The sparse view over the same GPUs is `[PP, D_replicate, EFSDP, EP]`; EP must divide `D_shard × CP × TP`, so EP cannot be multiplied into the first formula again. This equation belongs to this implementation and is not a universal formula for every framework, expert TP configuration, or heterogeneous deployment. [TorchTitan mesh definition](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/distributed/parallel_dims.py#L106-L240)

**Engineering synthesis.** For fixed-length batches, begin reconciliation with `B_sequence = D × b × m`: `b` is the number of sequences per microbatch per DP replica; `m` is the total number of microbatches actually accumulated in one optimizer update. If a framework exposes both PP microbatch count and gradient-accumulation group count, first check whether their product is `m` here. TP, PP, and CP cooperate on the same data and do not add independent samples.

For a purely hypothetical 256-GPU configuration with `TP=4, PP=4, CP=2, D=8`, if `b=1,m=16,S=8192`, each update contains 128 sequences and at most 1,048,576 prediction positions. Padding, document boundaries, and loss masks reduce valid target tokens; after sequence packing, sequence count also differs from document count. This example checks arithmetic only and recommends no hardware recipe. Layout decisions should place the most frequent communication within suitable interconnect domains while measuring PP bubbles, cross-node MoE traffic, and peak memory, instead of optimizing only one communication dimension.

<a id="3-一次训练更新究竟流过哪些状态"></a>

## 3. Which States Does a Training Update Actually Pass Through?

**Engineering synthesis.** The real step 0 also establishes process groups, allocates model shards, and randomly initializes parameters. Replicas of the same logical parameters must agree, while different parameter shards must not become duplicate matrices through incorrect seed use. Check initialization distributions, tied weights, and parameter finiteness; create the optimizer on the final parameter layout, and record initial random streams and data position. After changing initialization, old optimizer momentum cannot be reused; equivalence between distributed and single-GPU initialization needs separate validation.

**Source facts and engineering synthesis.** Logically this is one chain; physically many stages overlap:

`Fix this step's data → forward/loss → backward/accumulation/communication → complete gradient synchronization → numerical checks and clipping → optimizer → scheduler → recoverable progress`

1. **Determine the data and denominator.** Read token IDs, position/attention information, and label masks, constructing next-token targets for the autoregressive task. Record consumed-data positions for this step and clear the previous gradients. Do not simply average rank-level mean losses: differing valid-target-token counts change sample weights. TorchTitan first counts valid target tokens across all accumulation groups and PP microbatches, sums them over the batch mesh, and supplies the same denominator to forward/backward.
2. **Forward.** All-gather parameter shards as needed; pass through embeddings, attention, and MLP/MoE to produce logits and cross-entropy. TP/CP exchange information within layers, and PP sends activations according to its schedule. With activation checkpointing, only selected intermediates are retained and other activations are recomputed during backward. The memory savings have a corresponding compute cost.
3. **Backward and accumulation.** Gradients propagate backward from the loss; PP sends activation gradients backward, while TP/CP perform their required reductions. DP gradient buckets can reduce-scatter/all-reduce as they become ready, so the timeline usually has no completely isolated synchronization-only-at-the-end phase. Deferring synchronization during accumulation also has implementation constraints: TorchTitan explicitly accounts for CUDA graph capture when switching HSDP replicate all-reduce for the final group.
4. **Finalize gradients.** Wait for outstanding communication and handle special synchronization for shared embeddings, non-TP parameters, or expert replicas. Compute the global gradient norm over unique logical parameters, avoiding double-counted replicas or omitted shards. With loss scaling, check overflow and unscale before clipping in the correct units.
5. **Update and commit progress.** Check that loss and gradients are finite; once checks pass, execute the optimizer update and advance the learning rate by the established rules. TorchTitan checks global finiteness before update and waits for checkpoint staging to finish so the next update cannot overwrite state before it has been copied. Termination/replay after failures and expected overflow skips are different policies: separately record consumed tokens, successful updates, and skip counts, and advance/save the data cursor, scheduler, and counters under an explicit contract without conflating their meanings.

Follow this sequence directly in [TorchTitan `train_step`](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/trainer.py#L860-L1003). Megatron's [gradient-finalization logic](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/distributed/finalize_model_grads.py#L560-L713) shows another implementation of `finish_grad_sync`, special-parameter synchronization, and valid-target-token scaling. **These are not one training loop that can be assembled by combining their steps**: first locate loss normalization and gradient scaling in the current backend to avoid dividing twice by the DP degree or token count.

With AdamW, for example, the update uses current gradients to update first and second moments, applies bias correction to construct the parameter step, and performs decoupled weight decay as configured. A distributed optimizer can have each rank hold and update only its own state shards, then exchange parameters needed for subsequent computation. State ownership and communication-completion order are central here; other optimizers have different states and update rules, and not every frontier model can be assumed to use AdamW.

Megatron provides a concrete counting boundary: this path advances the scheduler only after a successful update, while the outer loop still advances iteration and consumed-sample counts. Consumed samples therefore cannot be equated with samples that all contributed to parameter updates. [Megatron scheduler / skip branch](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/training/training.py#L3309-L3323), [iteration and sample counts](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/training/training.py#L4887-L4920)

<a id="4-moe计算稀疏后通信与路由进入核心路径"></a>

## 4. MoE: Sparse Compute Brings Communication and Routing onto the Critical Path

**Source fact.** Megatron's MoE forward organizes routing, dispatch, expert computation, and combination into separable stages. Trace each token's selected expert IDs, routing weights, permutation indices, tokens received per expert, and metadata that returns expert outputs to original token positions. [Megatron MoE forward](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/moe_layer.py#L637-L742)

**Engineering synthesis.** A typical path is router → top-k/capacity handling → pack → dispatch → expert grouped GEMM and activation → combine → reconstruct outputs using routing weights; the exact weighting location is implementation-defined. Backward must also traverse permutation and communication. Expert weights across EP are different parameters; gradients from arbitrary EP ranks cannot be directly averaged. Identify the DP replica group for the same expert instead.

Insufficient capacity, dropping/padding, load-balancing losses, and router precision affect both throughput and optimization. In addition to average token counts, record the busiest experts, empty experts, drop rate, and cross-node destination distribution. Total parameters determine part of the storage cost, while experts activated per token affect computation; neither total nor active parameter count alone predicts whole-step time.

DeepEP provides efficient dispatch/combine along this path. Its SM-resource estimation function explicitly assumes balanced routing and excludes certain group-limited gates. This shows that communication optimization depends on the actual distribution; theoretical bandwidth or SM guidance cannot be copied directly to another routing scheme. [DeepEP bandwidth-model boundaries](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L729-L834) Communication kernels and expert GEMMs compete for SMs, HBM, and links. After adding overlap, recheck whole-step tail latency and GPU memory instead of only observing a shorter communication kernel.

<a id="5-低精度必须经过数值与收敛验收"></a>

## 5. Low Precision Requires Numerical and Convergence Validation

**Source fact.** DeepGEMM's FP8/FP4 interfaces check matrix layouts, output types, scale formats, and architecture before selecting an implementation; casting a tensor to low precision is far from sufficient. [DeepGEMM interface contract](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/gemm.hpp#L73-L123) Transformer Engine's public documentation further distinguishes low-precision formats, scale granularity, amax history, and transposed-quantization requirements; together they form a numerical recipe. [Transformer Engine low-precision documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

**Engineering synthesis.** Expand validation in order: compare output and gradient errors for representative shapes against a high-precision reference; compare short loss/gradient curves on identical data in a small model; then assess validation quality and stability at representative scale, sequence length, and token budget. Check outliers, overflow/underflow, scale updates, and extreme expert loads. Parameters, activations, gradients, reductions, and optimizer states may use different precisions. Define thresholds for each object in advance, instead of requiring only the absence of NaNs.

If persistent states such as scale/amax affect the next step, include them in the recovery contract. Whether sensitive operators need higher precision is an experimental question. A kernel benchmark, normal loss for several steps, and full pretraining reaching target quality are three evidence levels. This chapter has not run those experiments and does not claim that FP4 has replaced every numerical path in the training of a very large model.

<a id="6-测量整步的有效产出"></a>

## 6. Measure Useful Output Across the Whole Step

**Engineering synthesis.** First separate cold start, JIT compilation/autotune, graph capture, and steady-state operation. Then record step-time distributions by rank to find the slowest rank's critical path. Observe input waits, CPU scheduling, H2D, operators, collectives, PP bubbles, recomputation, waiting to save state, and write backlogs. Averaging only one rank hides the tail before each step's barrier.

Report at least valid training target tokens/s, total processed tokens/s, step-time quantiles, peak GPU memory, and long-term valid target tokens/wall-clock time including failure recomputation and checkpoint overhead. For MFU, state the model-FLOP estimate, precision used for hardware peak, and counting conventions for MoE/recomputation; changing the denominator must not manufacture an improvement. Change only a few factors per optimization and retain before/after traces, configurations, shapes, and error results, distinguishing an eliminated bottleneck from a relocated one.

SGLang primarily serves inference and RL rollouts. Its request scheduling, KV cache, and prefill/decode optimizations are instructive, but it does not perform base-model optimizer updates here. Its serving tokens/s also cannot be directly compared with pretraining tokens/s. See the [SGLang source-learning notes](../notes/repositories/sglang.md).

<a id="7-checkpoint-是一致的训练状态而不只是权重文件"></a>

## 7. A Checkpoint Is Consistent Training State, Not Just a Weight File

**Engineering synthesis.** State for continued training should specify at least model parameters; optimizer momentum/variance and necessary master parameters; scheduler and successful-update count; consumed tokens and data-sampling/mixture stage; dataloader cursor and shuffle/packing state; random streams; persistent low-precision state; and configuration, source, tokenizer/data manifests. Record the mapping from ranks to logical shards, and publish a recoverable marker only after verifying completion of all necessary shards. Exact fields vary by algorithm and framework and must be confirmed through recovery experiments.

Asynchronous saving has two completion points: copying GPU state into stable CPU buffers, and finishing background writes to persistent storage. The first allows updates to continue; the second determines what remains recoverable after losing a node. They must not share one save-complete dashboard metric. TorchTitan implements separate waits for staging and saving; its current loading path also retains a TODO for saving rank-local RNG, so this snapshot cannot be claimed to guarantee exact resumption. [TorchTitan recovery and asynchronous state](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/components/checkpointer/dcp.py#L580-L650)

Megatron explicitly collects Python, NumPy, Torch, CUDA, and TP RNG tracker states, illustrating why setting the same seed cannot replace restoring random streams. [Megatron RNG state](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/training/checkpointing.py#L451-L516) Saving RNG state alone still does not guarantee bitwise determinism for every kernel.

Recovery validation should compare uninterrupted execution with save–terminate–resume: subsequent sample IDs, masks, learning rates, optimizer states, loss/gradients, and parameter differences. Test the same topology first, then resharding under a changed parallel layout. Loadable weights, resumable optimization, statistical reproducibility, and bitwise identity are four different promises; changing world size changes reduction order and data assignment. Model-export files must not be assumed to be complete training checkpoints.

<a id="8-投产关卡与故障处理顺序"></a>

## 8. Production Gates and the Order of Failure Handling

**Engineering synthesis; all checks below await execution and are not experimental results from this repository.**

| Gate | Required evidence |
|---|---|
| G0: Freeze environment/data | Version and topology manifests, data checksums, communication correctness, and read/write validation |
| G1: Minimal numerical baseline | Comparisons for loss masks, normalization, gradients, and low-precision errors |
| G2: Small-scale multi-GPU | Error against the reference update after parallelization; MoE dispatch/combine and gradient comparisons |
| G3: Target-scale pilot | Steady-state traces, slow ranks, concurrent storage load, save and recovery rehearsals |
| G4: Representative training experiment | Validation curves at a fixed budget, numerical stability, and throughput/cost records |
| G5: Sustained training | Quality monitoring, anomaly thresholds, recoverable state, and audits of data/configuration changes |

On failure, first preserve evidence of the failed step, first error-reporting rank, configuration, data position, and last complete checkpoint, then classify the issue:

- **Slow rank/communication timeout:** align rank timelines and distinguish blocked input, GPU throttling/errors, skewed routing, NIC links, and inconsistent collective order. The timed-out rank may not be the root cause; merely increasing the timeout can conceal the problem.
- **NaN/Inf or sudden loss changes:** prevent invalid gradients from entering the update. Handle expected loss-scaling overflow using the predefined skip/scale-adjustment policy and count it separately. For unexpected or persistent anomalies, start from healthy state and use the same data to narrow the failure to the first anomalous layer, checking scales, normalization, gradient norms, and data anomalies. Temporarily increase precision if needed to locate the problem, then compare the repaired trajectory. Explicitly distinguish rules for skipping updates, advancing data, and advancing the scheduler.
- **Node exit or silent data corruption:** isolate suspected nodes and rebuild consistent process groups; the default process should recover from complete valid state. Restarting one rank alone requires a dedicated fault-tolerance protocol and cannot be assumed to work in ordinary synchronous training. Replay/redundancy detection also has limited coverage; one passing check does not establish that every operator is safe.

After recovery, run the agreed health-check window first, checking sample sequence, gradients, learning rate, routing, and throughput before permitting long training. Record the reason, scope, rollback point, and comparison evidence for every change. Operational validation passes only when correct updates, model quality, and long-term useful output all meet requirements.
