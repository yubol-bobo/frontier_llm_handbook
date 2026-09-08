# Open Instruct：训练与 rollout 的概率差异如何进入 GRPO loss

日期：2026-09-08  
阶段：L1 源码初读  
源码：https://github.com/allenai/open-instruct @ `ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5`  
验证范围：本地 GRPO learner step、loss/correction、权重同步与 OLMo-core SFT 入口；未运行 Ray、vLLM、DeepSpeed、Beaker 或模型训练。

快照限制：4 个上游 test-data 文件仍为 LFS pointers，payload 未拉取，详见 [本次审计](../sessions/2026-09-08-training.md)。源码存在不等于原始测试数据和运行环境已经就绪。

## 核心问题

当采样引擎与训练引擎不同，甚至 rollout 版本落后于 learner 时，为什么不能只写一个 `exp(new_logprob-old_logprob)` 就结束？系统调度、masking 和 loss 归一化怎样共同改变训练语义？

## 源码路径与执行链路

1. [grpo_fast.py](../../sources/open-instruct/open_instruct/grpo_fast.py) 中 [`PolicyTrainerRayProcess.step`，L586–765](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_fast.py#L586-L765)：取 dataloader batch → 可选 sequence parallel 切分 → ref / old logprobs → new logprobs → ratio / rho correction → policy + KL loss → masked normalization → backward / optimizer boundary。
2. [grpo_utils.py](../../sources/open-instruct/open_instruct/grpo_utils.py) 的 [`compute_rho_correction` 到 `compute_grpo_loss`，L410–538](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_utils.py#L410-L538)：具体区别 train-old/infer-old 的修正与 new/old 的优化比率。
3. 同文件 [`perform_weight_sync`，L745–782](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/open_instruct/grpo_utils.py#L745-L782)：暂停 actor manager → 等待 broadcast → 按配置等待内层 engine RPC → wake up → finally 恢复 actor。调用方 `grpo_fast.py:1459–1517` 用携带目标 step 的 event/thread 触发同步，本次也实际阅读。
4. [pyproject.toml](../../sources/open-instruct/pyproject.toml) 的 [L7–74](https://github.com/allenai/open-instruct/blob/ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5/pyproject.toml#L7-L74) 是包依赖与 OLMo-core commit pin 的证据。另读 [olmo_core_finetune.py](../../sources/open-instruct/open_instruct/olmo_core_finetune.py) 的 L16–105：SFT 直接导入 OLMo-core，并将 HF 数据转换为 token IDs / labels mask / metadata 分片。

## 已确认的机制与取舍

**两个 ratio 不是一回事。** `step` 的优化比率为 `exp(new_logprobs - old_logprobs)`；rho 为 `exp(old_train_logprobs - vllm_logprobs)`。前者衡量本轮策略更新，后者处理训练/生成分布差异。`use_vllm_logprobs` 可使 old baseline 直接取生成器概率；否则来自本地模型的缓存 / 首轮 detach。多个 minibatch 时预先算旧概率，防止 baseline 随 optimizer 更新移动。

**rho correction 是有偏差取舍的稳定化选择。** 实现先将 logprob 差限制在 [-10,10]，可按 token 或序列聚合，可 clamp，也可将越界 token 的权重置零；默认关闭时返回全 1 权重，但仍留 rho histogram。开启后会改变哪些样本贡献梯度；不能声称它“无代价恢复严格 on-policy”。异步版本滞后、kernel 数值差异及实际 sampling 分布必须另行区分。

**不同 policy objective 的梯度位置不同。** DAPO 分支取 unclipped / clipped surrogate 的较大 loss；CISPO 分支对 ratio detach 后裁剪，再乘 new logprob。它们不只是两个阈值名字。rho_weights 乘入 policy loss；KL 另算，最终在 caller 组合 `pg_loss + beta*kl`。本次未证明这些分支与某篇论文每个细节完全等价。

**有效 tokens 决定分布式权重。** learner 计算 accumulation group 的 token 分母，response mask 排除非回答 token。代码随后乘 `world_size // sequence_parallel_size`，注释说明这是抵消 DeepSpeed 按 ranks 平均，以匹配 token 归一化。样本长度变化时简单地“每 rank loss 平均”会改变训练权重。

**权重同步是系统协议。** `perform_weight_sync` 在 finally 恢复 actor manager，避免异常把 rollout 永久留在停止状态；`inflight_updates=True` 则跳过内层 RPC 等待。只从函数名不能推断每种模式都提供同样的更新原子性。调用方先产生 broadcast refs 再进入该函数，因此本次不把代码简化成“绝无在途请求的同步屏障”。

## 与其他项目的连接

| 关系 | 证据与边界 |
|---|---|
| OLMo-core：直接依赖且版本固定 | `pyproject.toml:21,74` pin `fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d`；本实验室独立 clone 的 HEAD 为 `92870a33…`。读代码可连接知识，运行时必须遵循依赖 pin，不能用相邻文件夹替代。 |
| vLLM / DeepSpeed / Ray：直接依赖 | manifest 明确列出；learner 的 model.backward/step 与远程 actors/engines 构成训练与生成分工。本次没有 clone vLLM / DeepSpeed，不能称完成底层审计。 |
| verl / slime / prime-rl：概念对应 | 都值得比较 rollout 版本、概率校正、有效 token 归一化与权重传输；并不说明本 repo 依赖这些框架。 |
| SGLang：概念对应 | 在学习树中属于另一种 rollout/inference 引擎选择；本次这条 GRPO 路径实际传递的是 vLLM 引擎，不能写成已接入 SGLang。 |

## 动手实验

**状态：待执行。** 先做不需生成模型的 loss 数值实验。

- 输入：手工构造 B×T 的 new、old、vLLM logprobs、正负 advantages、response mask；包含空 padding、短长回答与极端差值。
- 控制变量：固定 tensor 和 old baseline；分别切换 rho 开关、token/sequence 选择、DAPO/CISPO。
- 指标：逐 token 权重、drop/clip fraction、loss、对 new_logprob 的梯度；确认 padding 不改变有效分母。
- 预期：rho 关闭的权重全 1；old=vLLM 时 rho 为 1（仍需检查 mask/阈值）；CISPO 的 ratio 路径无梯度，梯度来自乘入的 new logprob。
- 算力：隔离环境的 CPU 张量级测试即可；当前模块导入链含研究依赖，需先明确最小依赖或提取自有测试 harness。完整 rollout 延迟实验另需 GPU，不在本次执行范围。
- 实际结果：无。没有申请集群、构建镜像或发布模型。

## 下一步与疑问

- [ ] 追踪 dataloader / ActorManager：advantages 如何按 prompt group 归一化、版本滞后怎样被记录和丢弃。
- [ ] 沿 `broadcast_to_vllm` 检查 optimizer step 与参数传输的并发约束，特别是 inflight_updates。
- [ ] 用上游已 pin 环境验证一个小模型，再谈替换 OLMo-core / inference engine 版本。
