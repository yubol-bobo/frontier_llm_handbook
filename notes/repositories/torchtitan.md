# TorchTitan：并行变换的次序与全局 token 归一化

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/pytorch/torchtitan @ `4e20e76b235ec415bd81912ce09eed46fa39f717`  
验证范围：本地 Llama3 sharding / parallelize、训练入口与 Trainer 的 token accumulation、forward/backward、finite check；未安装 nightly PyTorch 或运行 GPU/收敛测试。

## 核心问题

DP、TP、PP、CP、activation checkpointing、compile 并不是可以任意重排的开关。怎样从实现理解它们的组合顺序，并确认不同 batch/parallelism 配置优化的仍是同一个目标？

## 源码路径与执行链路

入口 [torchtitan/train.py](../../sources/torchtitan/torchtitan/train.py):17–68 解析配置，`config.build()` 创建 trainer，再 `trainer.train()`；seed checkpoint 是单设备专门分支。当前核心循环在 `trainer.py`，不是旧版教程可能展示的长 `train.py`。

1. [models/llama3/sharding.py](../../sources/torchtitan/torchtitan/models/llama3/sharding.py) 的 [L25–74](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/models/llama3/sharding.py#L25-L74)：填写 decoder/layer 的 sharding config。`model.py:69–83` 的 `update_from_config` 调入这个函数；声明与实际 mesh 的启用分离。
2. [models/llama3/parallelize.py](../../sources/torchtitan/torchtitan/models/llama3/parallelize.py) 的 [`parallelize_llama`，L23–87](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/models/llama3/parallelize.py#L23-L87)：`model.parallelize → AC → per-block compile → FSDP/mixed precision`。PP 另由 trainer/model_spec 的 pipelining 路径处理。
3. [trainer.py](../../sources/torchtitan/torchtitan/trainer.py) 的 [`forward_backward_step` / `_forward_backward_body`，L749–830](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/trainer.py#L749-L830)：preprocess_inputs 处理模型/并行布局，model → loss_fn(global_valid_tokens) → backward。PP 有 list-of-microbatches 的独立分支。
4. 同文件 [`train_step`，L860–1003](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/trainer.py#L860-L1003)：先收集一个 optimizer step 的 microbatch groups 和有效 tokens，分布式汇总分母，再执行 forward/backward、梯度裁剪、非有限值检查及 optimizer update。

## 已确认的机制与取舍

**布局声明与激活变换。** Llama3 sharding config 总是填写，实际 `Module.parallelize()` 按启用的 mesh axes 决定哪些声明生效。Sequence Parallel 可以独立配置：开启时 attention/FFN 周围 activations 分片，output projection 对应 reduce-scatter；关闭时保持 replicate 对应 all-reduce。代码调用共享 decoder_sharding helpers，本次没有审计其 DTensor/SPMD lowering 的每个 collective。

**变换顺序是实现约束。** `parallelize_llama` 先布局，再包 activation checkpointing，再按 TransformerBlock compile，最后 FSDP。源码建议 model 在 meta device 上构建，避免预先把完整权重塞入某张卡。FSDP shard degree=1 仍安装 MixedPrecisionPolicy；不能把“不分片”理解成这个 wrapper 没任何语义。

**训练与生成复用有边界。** `skip_dp` 为 inference 跳过 FSDP，注释说明其 forward hooks 与 vLLM 的 inference_mode 不兼容；AC / compile 需由配置关闭。这只是代码中存在的复用入口，本次未验证实际 vLLM engine 集成。

**batch budget 与真正分母是两层。** `trainer.py:455–474` 从 tokens/microbatch/DP-rank、PP microbatch 数及 DP degree 推出 accumulation steps，并检查整除。`train_step` 则先读取所有 groups 的 `num_valid_tokens`，在设备上跨 batch mesh 求和。前者决定排程与资源；后者排除 padding/masking，决定梯度的权重。非 PP 的返回 loss 明确为 local sum / global_valid_tokens，而不是各 rank 各自求均值。

**通信优化受到 CUDA graph capture 约束。** HSDP 在最后 accumulation group 才 all-reduce，但仅在 accumulation=1 或禁用 CUDA graphs 的条件下切换。源码指出 graph 在首组捕获并复用于后续组，因此不能把普通 Python 时序优化无条件套在捕获路径上。

**异步并不等于可以忽视一致性。** 在 optimizer step 前，将 loss finite 状态按相关 loss / PP mesh 归约，并与梯度范数有限性合并，用设备侧 `_assert_async` 停止无效更新；还 `maybe_wait_for_staging()`，再修改参数。这显示数值检查、checkpoint staging 与 optimizer 存在顺序约束。没有实测故障注入或断点恢复。

## 与其他项目的连接

| 关系 | 证据与边界 |
|---|---|
| OLMo-core：概念对应 | 对照两者有效 token 分母、HSDP 最后 microbatch 同步和 Trainer/module 分层。当前实现不同，不是互为依赖。 |
| Open Instruct：概念对应 | RL 中的 response_mask 和此处 num_valid_tokens 都影响分布式 loss 权重；RL 还多出 policy ratio、rho、版本滞后，不能直接照搬预训练目标。 |
| Megatron-LM：概念对应 | 两者解决模型分片/通信/重计算，TorchTitan 可作为清晰的组合入口，Megatron 的 MoE dispatcher 可作更深层案例；没有直接依赖证据。 |
| vLLM：代码支持的复用边界 | `parallelize.py:58–62` 存在明确 inference skip_dp 分支和约束；仅此不证明完整服务已运行。 |

## 动手实验

**状态：待执行。** 首先复现“切换 AC，不改变计算目标”。

- 输入：固定 seed、微型 Llama3 config、固定短 token batch；先相同设备和 parallelism。
- 控制变量：模型初始化、数据顺序、dtype、kernel、optimizer；只切换 activation checkpointing，先关闭 CUDA graphs，稳定后单独研究 graph 路径。
- 指标：每步完整精度 loss/grad_norm、更新后参数、有效 tokens、峰值显存、warmup 后 step time。
- 预期：AC 用重算换显存；在相同确定性条件下按项目要求比较数值一致性，而不是只看日志四舍五入后的 loss。性能测量至少跨过 warmup；本次不声称一定更快。
- 算力：微型模型先单 GPU；TP/HSDP、CP 和通信 trace 需要多 GPU。具体显存/时间尚未测量。
- 实际结果：无；未执行上游 tests 或 training。

## 下一步与疑问

- [ ] 读 `models/common/decoder_sharding.py` 与 `protocols/module.py`，追到布局声明如何变为每个 collective。
- [ ] 读 loss implementation 的 reduction / gradient scale，核对 global_valid_tokens 与 DP average 的准确关系。
- [ ] 读 distributed checkpoint staging：为什么 optimizer 必须等待某些 staging 操作，哪些 I/O 可以继续异步。
