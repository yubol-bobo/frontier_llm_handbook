<a id="marin训练实验的身份依赖与执行位置如何分离"></a>

# Marin: Separating training-experiment identity, dependencies, and execution location

Date: 2026-09-08  
Stage: L2 focused mechanism tracing (artifact DAG and 535B recipe handoff)

Source: https://github.com/marin-community/marin @ `5e2436d0f61462983003bd8b6eaef8235ecab78c`  
Verification scope: Local ArtifactStep / StepContext, fingerprinting, training builder, mixture, single-step runner, and 535B launcher / data / transport / checkpoint configuration; static reading and budget calculations, without starting JAX, Fray, Iris, TPU/GPU execution, or cloud-data access.

<a id="新增535b-真实运行的配方交接"></a>

## Addition: recipe handoff for the actual 535B run

See the [Marin 535B case study](../../handbook/04-marin-535b-live-case-study.md) for the subsequent complete record. This reading traced four concrete paths through `experiments/grug/moe_hero_ep`: scaling launcher → token horizon / optimizer; Harrier data → two-stage mixture; router / expert transport → state update; checkpoint manifest → restoration of master / device layouts.

The core conclusion is that “the same project” does not mean “the same experiment”: communication defaults in the current pin and the earlier run announcement differ in time, and backends for smaller scales and hero must also be checked separately. The candidate-data inventory, mixture target, and launcher token budget are three different quantities; falling back to a communication implementation may be constrained by the checkpoint's master-state format. The case study retains pinned-SHA evidence and dates for the authors' plans. It does not present ongoing training as a completed result or substitute current source for live job configuration.

<a id="核心问题"></a>

## Core question

Open-source model research is more than a training loop. How can the dependencies from data → recipe → checkpoint be traced, clusters changed without redefining the experiment, and caches prevented from silently hiding changes?

<a id="源码路径与执行链路"></a>

## Source paths and execution chain

1. In [execution/lazy.py](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/lazy.py), [`StepContext` / `ArtifactStep`, L66–245](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/lazy.py#L66-L245): a handle declares name, version, artifact type, run, build_config, and deps. Declaration does not read large datasets or launch jobs.
2. [`_lower` / `run`, L302–426](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/lazy.py#L302-L426) in the same file: recursively lowers to a StepSpec DAG and records fingerprint / provenance; delegates uniformly to StepRunner. Only during execution does it perform `build_config(real_context) → run(config) → ArtifactRecord`.
3. [experiment/train.py](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/experiment/train.py), [L27–242](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/experiment/train.py#L27-L242): `train_lm` assembles Levanter TrainLmConfig / TrainerConfig and generates dependencies from tokenized dataset handles. Only `_train_job → remote(run_levanter_train_lm, resources=…)` crosses into an actual training job.
4. [execution/fingerprint.py](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/fingerprint.py), [L90–170](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/fingerprint.py#L90-L170): normalizes configuration into sorted JSON, covering dataclass, Enum, set, dtype / arrays, then takes the first eight MD5 characters as the recipe fingerprint.

Also actually read `mixture` in [experiment/data.py](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/experiment/data.py):265–328 and `run_step` in [execution/step_runner.py](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/step_runner.py):430–474 to check how data validation, caching, and failure states connect.

<a id="已确认的机制与取舍"></a>

## Confirmed mechanisms and tradeoffs

**An artifact address is not a content hash.** The normal path is explicitly `{prefix}/{name}/{version}`. The fingerprint records how configuration is constructed for drift checking; it is not a content hash of model weights or the complete source. `_lower` explicitly does not inspect the implementation of the `handle.run` function; source provenance is stored separately in ArtifactRecord. Changing an implementation while keeping its version requires manual version discipline. A fingerprint does not establish fully automatic end-to-end reproducibility.

**The same configuration has two resolution contexts.** During fingerprinting, the output directory/region/runtime args use placeholders and dependency paths use `name@version`; at runtime they become real paths, regions, and resources. Thus, `resources=ctx.runtime_arg('train_resources')` does not enter experiment identity, while literals such as model, data versions, and mixture weights do. This mechanism does not automatically determine which parameters are scientifically important: the builder author is responsible for dividing literal from runtime values. Do not assume that every tracker, mesh, or operational parameter is automatically excluded.

**Dependency declarations are an execution contract.** `ctx.artifact_path` / `ctx.resolved` accept only handles declared in deps; otherwise they raise an error. `resolved` loads only an existing artifact record and does not silently execute its producer. `train_lm` builds all_deps in one pass from datasets, validation, and init_from so configuration and DAG use the same handles. `mixture` checks that component names do not conflict and that all tokenizers match at runtime; validation components have zero weight.

**Pinned versions cannot depend on mutable dev versions.** `_lower` explicitly rejects this relationship to avoid rebuilding child data while leaving the parent training cache untouched. `expected_fingerprint` is an optional hard pin; drift checking is advisory by default. The comments and branches in `run_step` show that mutable artifacts are rebuilt, while pinned versions can hit the cache. Execution uses locks and heartbeat/status for concurrency management, writing different success and failure states. Lock contention was not measured in this study.

**Epochs are defined at the token level.** `train_lm` requires exactly one of steps / epochs, and epochs allows only a single training data source. At runtime it uses `ceil(epochs * num_train_tokens / (seq_len*batch_size))`. This avoids overtraining packed SFT when steps are calculated from raw document counts, but depends on correct tokenized-cache counts. “One epoch” has no single natural meaning for mixed datasets, so steps must be supplied directly.

**Another kind of scarce knowledge in research platforms.** The core here is how cache identity, lineage, data counting, resource configuration, and failure recovery preserve the meaning of an experiment, rather than a new attention kernel. An 8-character hash is a short fingerprint, not a cryptographic integrity guarantee. Strict serialization can reject unstable objects, while the default best-effort fallback may still require auditing.

<a id="与其他项目的连接"></a>

## Connections to other projects

| Relationship | Evidence and boundaries |
|---|---|
| Levanter: direct training dependency within the monorepo | `experiment/train.py:30–37` imports Levanter model / optimizer / trainer; L196–225 actually constructs training configuration, and L98 calls the training wrapper. Corresponding source is in `sources/marin/lib/levanter`. This reading did not cover its inner JAX optimizer step. |
| Fray / Iris: scheduling-layer connection | `_train_job` calls the Marin remote wrapper; the single-step runner has `_run_iris_job` / RemoteCallable branches. This reading stops at the entry points and does not claim every training path is forced through the same scheduling backend. |
| OLMo-core / SmolLM: conceptual correspondence | The shared learning question is reproducibility of data, stages, and checkpoints. Their recipes can serve as comparisons, but the paths read here contain no evidence that Marin calls these two frameworks. |
| Agent RL harness: conceptual correspondence | Environment and reward-function versions also need identity / lineage. ArtifactStep provides a design to learn from, but this transferable idea must not be described as an existing integration. |

<a id="动手实验"></a>

## Hands-on experiments

**Status: pending.** Start by checking artifact identity; training a large model is unnecessary.

- Inputs: two small synthetic-data artifact handles and one consumer in a personal experiment directory, all using local temporary paths.
- Controls: change only storage prefix, runtime resource, literal hyperparameter, or dep version in turn; keep other configuration and name/version unchanged.
- Metrics: fingerprint payload/hash, lowered dependency graph, expected_fingerprint errors, and whether any data is read/written.
- Expected: prefix/runtime changes leave the fingerprint unchanged; literal/dep-version changes should alter it. A pinned parent version referencing a dev child is rejected; no training job should be submitted without calling run.
- Compute: a CPU suffices for this configuration experiment, but the current monorepo's Python/JAX dependencies still need separate isolation. No environment was installed and no upstream entry point was called.
- Actual result: none; expectations here are derived from source, without fabricated cache hits or training results.

<a id="下一步与疑问"></a>

## Next steps and questions

- [ ] Read drift comparison, record migration, and recovery from lock exceptions in artifact.py / step_status.py.
- [ ] Select a current experiments driver and trace a complete, unexecuted DAG through its data builder, train_lm, and eval.
- [ ] Build a CPU-only identity experiment in an isolated environment, explicitly avoiding cluster defaults in remote() / run().
