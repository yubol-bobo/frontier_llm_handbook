# Marin：训练实验的身份、依赖与执行位置如何分离

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/marin-community/marin @ `5e2436d0f61462983003bd8b6eaef8235ecab78c`  
验证范围：本地 ArtifactStep / StepContext、fingerprint、训练 builder、mixture 与单 step runner；未启动 JAX、Fray、Iris、TPU/GPU 或云数据访问。

## 核心问题

开源模型研究不只是训练循环。怎样让“数据 → 配方 → checkpoint”的依赖能被追踪，换集群不必重定义实验，并防止缓存把变更悄悄隐藏起来？

## 源码路径与执行链路

1. [execution/lazy.py](../../sources/marin/lib/marin/src/marin/execution/lazy.py) 的 [`StepContext` / `ArtifactStep`，L66–245](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/lazy.py#L66-L245)：一个 handle 声明 name、version、artifact type、run、build_config 与 deps。声明时不读取大数据或启动作业。
2. 同文件 [`_lower` / `run`，L302–426](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/lazy.py#L302-L426)：递归转为 StepSpec DAG，记录 fingerprint / provenance；统一交给 StepRunner；执行时才 `build_config(real_context) → run(config) → ArtifactRecord`。
3. [experiment/train.py](../../sources/marin/lib/marin/src/marin/experiment/train.py) 的 [L27–242](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/experiment/train.py#L27-L242)：`train_lm` 组装 Levanter TrainLmConfig / TrainerConfig，以 tokenized dataset handles 生成依赖；`_train_job → remote(run_levanter_train_lm, resources=…)` 才跨入实际训练作业。
4. [execution/fingerprint.py](../../sources/marin/lib/marin/src/marin/execution/fingerprint.py) 的 [L90–170](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/fingerprint.py#L90-L170)：配置规范化为排序 JSON，覆盖 dataclass、Enum、set、dtype / arrays，再取 MD5 前八位作为 recipe fingerprint。

另实际读取 [experiment/data.py](../../sources/marin/lib/marin/src/marin/experiment/data.py):265–328 的 `mixture` 和 [execution/step_runner.py](../../sources/marin/lib/marin/src/marin/execution/step_runner.py):430–474 的 `run_step`，用于检查数据校验、缓存与失败状态的接续。

## 已确认的机制与取舍

**artifact 地址不等于内容 hash。** 正常路径明确是 `{prefix}/{name}/{version}`。fingerprint 记录“配置怎样构造”，用于漂移检查；它不是模型权重或完整源码的内容 hash。`_lower` 明确不检查 `handle.run` 的函数实现；源码 provenance 则另存入 ArtifactRecord。改实现但沿用版本需要人工版本纪律，不能因为有 fingerprint 就宣称端到端完全自动可复现。

**同一配置有两种解析上下文。** fingerprint 时，输出目录/region/runtime args 用占位符，依赖路径用 `name@version`；run 时换成真实路径、region 和资源。因此 `resources=ctx.runtime_arg('train_resources')` 不进入实验身份，而模型、数据版本、混合权重等 literal 会进入。这个机制不是自动判断“哪些参数科学上重要”：literal / runtime 的划分由 builder 作者负责；不能假设所有 tracker、mesh 或 operational 参数都会被自动排除。

**依赖声明是执行契约。** `ctx.artifact_path` / `ctx.resolved` 只接受 deps 中声明的 handle；否则报错。`resolved` 只加载已有 artifact record，并不偷偷执行生产者。`train_lm` 从 datasets、validation、init_from 一次性构建 all_deps，让配置与 DAG 使用同一组 handle。`mixture` 检查 component names 不冲突、运行时所有 tokenizer 一致；validation 分量权重为零。

**固定版本不能依赖 mutable dev。** `_lower` 明确拒绝这种关系，避免子数据重建而父训练缓存不动。`expected_fingerprint` 是可选 hard pin；默认漂移检查是 advisory。`run_step` 的注释和分支说明 mutable artifact 会重建，固定版本可命中缓存；执行期间用锁、heartbeat/status 管理并发，成功和失败写不同状态。没有在本次实测锁竞争。

**epoch 定义在 token 层。** `train_lm` 要求 steps / epochs 二选一，epochs 只允许单训练数据源；运行时用 `ceil(epochs * num_train_tokens / (seq_len*batch_size))`。这避免 packed SFT 按原始文档条数算步数时过训，但依赖 tokenized cache 的计数正确。对混合数据集“一个 epoch”没有单一自然含义，因此要求直接提供 steps。

**研究平台的另一类稀缺知识。** 这里的核心不是一个新 attention kernel，而是 cache identity、lineage、数据计数、资源配置和失败恢复如何保持同一个实验含义。8 位 hash 是短指纹而非密码学完整性承诺；strict serialization 能拒绝不稳定对象，默认 best-effort fallback 仍可能需要审计。

## 与其他项目的连接

| 关系 | 证据与边界 |
|---|---|
| Levanter：monorepo 内直接训练依赖 | `experiment/train.py:30–37` 导入 Levanter model / optimizer / trainer；L196–225 实际构造训练配置，L98 调用训练 wrapper。对应源码在 `sources/marin/lib/levanter`。本次未读其内层 JAX optimizer step。 |
| Fray / Iris：调度层连接 | `_train_job` 调用 Marin remote wrapper；单 step runner 有 `_run_iris_job` / RemoteCallable 分支。本次止于入口，不声称每条训练路径都强制走同一调度后端。 |
| OLMo-core / SmolLM：概念对应 | 相同学习问题是数据、阶段、checkpoint 的可复现性；它们的配方可用于对照，但没有在本次路径发现 Marin 调用这两个框架的证据。 |
| agent RL harness：概念对应 | 环境版本、奖励器版本也需要 identity / lineage；可借鉴 ArtifactStep 设计，不能把这种可借鉴关系写成现成集成。 |

## 动手实验

**状态：待执行。** 先检查 artifact 身份，无需训练大模型。

- 输入：自己实验目录的两个小型人工数据 artifact handles 与一个 consumer；全部使用本地临时路径。
- 控制变量：依次只改 storage prefix、runtime resource、literal hyperparameter、dep version；保持其他配置和 name/version 不变。
- 指标：fingerprint payload/hash、lower 后依赖图、expected_fingerprint 错误、是否读取/写入任何数据。
- 预期：prefix/runtime 变化不改指纹，literal/dep version 变化应改指纹；固定父版本引用 dev 子版本被拒绝；没有调用 run 时不应提交训练作业。
- 算力：CPU 配置实验足够，但当前 monorepo 的 Python/JAX 依赖仍需单独隔离。没有安装环境，也未调用任何上游入口。
- 实际结果：无；本次依据源码推导预期，不填写伪造 cache 命中或训练结果。

## 下一步与疑问

- [ ] 读取 artifact.py / step_status.py 的 drift 比较、记录迁移与锁异常恢复路径。
- [ ] 选一个当前 experiments driver，沿数据 builder、train_lm、eval 追踪一个完整但不执行的 DAG。
- [ ] 在隔离环境建立 CPU-only identity 实验，明确避免 remote() / run() 的集群默认值。
