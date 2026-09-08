# OLMo-core：一个 optimizer step 如何跨 microbatch 与分布式同步保持语义

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/allenai/OLMo-core @ `92870a33c3fee060d57c3faec52b0b369ece85a6`  
验证范围：本地正式 Olmo 3 配方、Trainer 和 TransformerTrainModule 的相关方法；未实例化模型、未执行数值或 GPU 测试。

## 核心问题

为何训练框架要拆成“实验配置 → 生命周期 Trainer → 模型训练模块”？这种拆分如何把 token 归一化、梯度累积、通信与评估协调起来？

## 源码路径与执行链路

1. [Olmo 3 7B pretrain 配方](../../sources/olmo-core/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py) 的 [`build_config`，L40–111](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py#L40-L111)：组装 tokenizer、data mix、模型、train module、TrainerConfig，文件末尾将构造器传给 `main`。本次未继续审计 `main` 的分布式启动细节。
2. [trainer.py](../../sources/olmo-core/src/olmo_core/train/trainer.py) 的 [`_fit_epoch`，L1466–1537](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/trainer.py#L1466-L1537)：reshuffle → pre_epoch → 每批 bookkeeping / pre_step → train_batch → pre_optim_step → optim_step → zero_grads → post hooks → 结束判断。
3. [transformer/train_module.py](../../sources/olmo-core/src/olmo_core/train/train_module/transformer/train_module.py) 的 [`train_batch`，L345–441](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/train_module/transformer/train_module.py#L345-L441)：生成 labels，统计参与 loss 的 tokens，拆 microbatch，逐批 forward/backward，必要时立即 all-reduce loss 供 SkipStepOptimizer 使用。
4. 同文件 [`optim_step` 与 `_train_microbatch_context`，L514–584](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/train_module/transformer/train_module.py#L514-L584)：clip gradient → scheduler → optimizer；HSDP / DDP 的最终 microbatch 触发不同的同步行为。

## 已确认的机制与取舍

**token 数与样本数必须区分。** 官方配置 `rank_microbatch_size=2*8192` 按 token 数表达。实现用 `rank_microbatch_size // seq_len` 得到每个 microbatch 的样本数；若 token budget 小于单条序列，明确报错。microbatch 的 loss reduction 为 sum，并共享整个 rank batch 的有效 token 分母。不能把每个 microbatch 的平均 loss 再简单相加，否则最后一个较小 microbatch 或不同 masking 比例会改变权重。

**梯度累积也改变通信时序。** `_train_microbatch_context` 对 HSDP 仅最后一批设置 `set_requires_all_reduce(True)`；FSDP 同时用 `set_is_last_backward` 管理未完成 reduction / prefetch；DDP 的非最后一批进入 `no_sync()`。这里可确认 HSDP 的 all-reduce 被延迟；不能进一步概括成“前面完全不通信”，因为 reduce-scatter、参数获取等仍可能发生。

**Trainer 不应知道每一层的训练细节。** `_fit_epoch` 管理 step/token/FLOP 计数、callback、停止条件；train module 管理 loss、梯度与模型。这样同一生命周期可承载不同训练模块。代价是调试一条行为时需要跨配置、callback 和模块追踪，不能只看 `Trainer.fit()`。

**真实配方与库能力不相同。** 该 7B 配方选择 HSDP、BF16 参数 / FP32 reduction、embedding 不做 weight decay、z-loss=1e-5、同步 checkpoint，且 `Float8Config(enabled=False)`。因此“库有 Float8”不是“这个发布模型用了 Float8”。配方记录从原计划 5T tokens 转为更长 schedule 的切分点，也是理解训练历史的线索。

**源码自己承认的统计取舍。** `instance_mask` 屏蔽实例后，实现仍将对应 tokens 加入 loss 分母，并注释说明这会使 loss 偏低，换取各 rank 公平贡献且避免昂贵分布式处理。这是代码层面的已知取舍，不是本次测试发现。比较 runs 时应同时观察 masked labels / instances，而不是只比较 CE 曲线。

## 与其他项目的连接

| 关系 | 证据与边界 |
|---|---|
| Open Instruct：真实直接依赖 | 其 `pyproject.toml:21,74` 声明并 pin `ai2-olmo-core`，SFT 代码导入本库；固定证据见 [Open Instruct 笔记](open-instruct.md)。但所 pin commit 为 `fa6c5014…`，不同于本目录 HEAD，未验证两份 HEAD 可直接组合。 |
| TorchTitan：概念对应 | 都将训练循环与并行应用逻辑分开，均有按 microbatch 管理同步的实现；不是彼此直接依赖。 |
| SmolLM：训练配方概念对应 | 两者显式处理 embedding weight decay 与有效 token budget；只能说明值得比较的机制，不能从相似配置推断算法来源。 |

## 动手实验

**状态：待执行。** 用微型 transformer 检查固定 batch 下 microbatch 切分是否保留梯度。

- 输入：固定 token IDs、部分 label=-100 的两种 batch，关闭 dropout；global/rank batch 不变。
- 控制变量：模型初始参数、optimizer、精度、数据、有效 loss 分母；只改变 rank_microbatch_size。
- 指标：首次 step 前完整梯度或可比梯度摘要、更新后参数差异、CE、峰值显存；多卡时加 communication trace。
- 预期：数学上应接近同一 batch 梯度；浮点累加顺序可能改变末位，不事先许诺 bitwise identical。含 instance_mask 的 case 应与源码的特殊分母规则对齐。
- 算力：单 GPU 微型模型先验 token 归一化；验证 HSDP 延迟 all-reduce 需适当多卡拓扑。未测实际显存。
- 实际结果：无；没有运行 `pytest` 或训练，因为本次交付是源码笔记，不是上游代码修改。

## 下一步与疑问

- [ ] 追踪 `SkipStepOptimizer` 怎样将 latest_loss / latest_grad_norm 转成跳步决策，以及跳步时 scheduler / token counter 是否继续推进。
- [ ] 读 `script_utils.ExperimentConfig` 的 build 次序，检查 checkpoint 恢复时数据游标和 optimizer 状态的配套约束。
- [ ] 对比官方旧 Numpy data mix API 与当前 composable data API 的身份和恢复语义。
