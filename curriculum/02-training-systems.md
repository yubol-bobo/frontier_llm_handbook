# 训练系统课程：M05—M09

先完成 M05，再并行学习 M06/M07；M08 综合前三者。M09 可在 M05 后开始，并须在 agent RL 系统课程前完成。

**全部练习均为待执行设计，不代表个人结业。** 无 GPU 可完成控制流、数值、状态机和 trace 路线；CPU 模拟不提供真实 GPU 性能证据。工时是核心阅读加纸上/CPU 练习的估计，不含安装、排队和完整训练；GPU 扩展另计，均非保证。

源码固定到记录的快照；新增官方入口于 2026-09-08 核查。每模块按三组核心阅读推进，再做练习和验收。课程设计是工程归纳，源码事实、作者 benchmark 与自己的测量须分开标记。

<a id="m05"></a>

## M05 — 分布式与并行基础

**先修：M01、M02、M04。可跳过诊断：**画双卡梯度平均，解释不等长 microbatch 的全局目标，列出 DP/TP/PP/CP 切分对象及通信代价。均正确可略读原理组；源码与验收仍须完成。

**知识细目。** rank、进程组、物理拓扑与逻辑 mesh；sum/average、all-reduce、reduce-scatter、all-gather、点对点；参数/梯度/优化器状态的所有者；DDP、FSDP/ZeRO 与 HSDP；列/行张量切分及非 TP 参数；PP 调度、microbatch、空泡、重计算；CP 与 TP 配套 sequence parallel 的区别。最后把显存账本、有效 batch 和梯度归一化放到同一张图上。

**核心阅读，按顺序：**

1. 读 [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 中 data/tensor/pipeline/context parallelism 的图解，再对照 [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) 的输入输出。目标是能在小张量上写出每个 rank 最终得到什么；此轮不追 benchmark 表格。
2. 读 [TorchTitan 笔记](../notes/repositories/torchtitan.md) 和 [parallel_dims.py:106–240](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/distributed/parallel_dims.py#L106-L240)，找 world size、batch/sparse mesh。再读 `models/llama3/parallelize.py:23` 的 `parallelize_llama`：它调用声明式 `model.parallelize`，然后处理 AC/compile/FSDP。输出 mesh 职责与变换顺序。
3. 对照 [Megatron 梯度收尾:560–713](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/distributed/finalize_model_grads.py#L560-L713) 与 [运行章 §3](../handbook/02-distributed-pretraining-operations.md)。标出等待梯度通信、特殊参数归约、有效 token 缩放的位置；解释为何不能直接把两套循环拼起来。

**选读：**Playbook 的 overlap/recomputation；Megatron pipeline schedules。掌握基本更新后再追 interleaved PP、CUDA graph 与特殊重算 mesh。

**画/推导：**8 rank 的 `DP=2,TP=2,PP=2` 参数/样本图；`B_sequence=D×b×m`，m 为一次更新的全部 microbatch。分项计算状态、激活及临时 buffer 字节，解释重算/分片改变哪项，保留 dtype 与峰值假设。

**练习 A：CPU 梯度对账。** 用小线性模型比较整批 token 平均、有效 token 加权分组，以及错误的“局部平均再平均”。覆盖等长、不等长、全 mask；对照 loss、梯度、一次更新，预设容差和零有效 token 策略。可用 NumPy 手写梯度或 CPU autograd。**GPU 升级：**对照通过后用兼容双卡验证真实 DP 缩放；列表求和不算通信测试。

**练习 B：纸上布局账本。** 为同一模型拟定两套合法布局，画 2 个 PP stage、4 个 microbatch 的时序，标同步、阶段不均衡和通信逻辑字节。**GPU 升级：**固定全局 batch 并通过数值对照后采集两种布局的 trace；不把同时改变 batch 的效果归因给并行方法。

**验收：**CP 为何不增加样本？FSDP 为何可能更慢？全局 norm 如何排除副本？EP 为何不能随意乘进设备数？回答须指向源码；“卡多必须扩大 batch”“all-reduce 总在 backward 最后”是典型误解。

**提交产物：**`M05-layout.md`、状态/通信账本、数值对账脚本与结果（未运行则写设计）、一张流水线图。**工时：**16–26 小时，GPU 扩展另计 4–8 小时。**下一模块：**M06 与 M07；也可开始 M09。

<a id="m06"></a>

## M06 — MoE 架构、路由与分布式执行

**先修：M05。可跳过诊断：**为 6 token、4 专家、top-2 写出正反向映射，解释专家副本、容量与 padding。通过可略读路由定义，仍需读 handle 和异步所有权。

**知识细目。** 总参数/激活参数/共享专家；router logits、top-k、路由概率及其梯度；capacity、dropping、padding；辅助均衡目标与 router z-loss 的不同作用；expert DP/TP/EP 组；token permutation、前缀和、grouped GEMM；dispatch/combine 的反向配对；stream/event、buffer 与 handle 生命周期；均匀与倾斜负载、节点内/跨节点路由。不同配方的均衡策略不可统称为同一种“防拥塞”。

**核心阅读，按顺序：**

1. 从 [Megatron 笔记](../notes/repositories/megatron-lm.md) 进入 [router.py:750–841](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/router.py#L750-L841)。记录 logits 到 routing map/probs 的 shape、容量分支及辅助损失接入梯度的位置；区分训练目标与日志指标。
2. 沿 [moe_layer.py:637–742](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/moe_layer.py#L637-L742) 读 route、dispatch、expert compute、combine。借笔记追 `token_dispatcher.py` 的 manager 选择；只画已证实的 adapter 边，不能因为有 grouped GEMM 就断言接了 DeepGEMM。
3. 读 [DeepEP 笔记](../notes/repositories/deepep.md) 与 [dispatch:855–1033](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L855-L1033)，随后读同文件 `combine:1046–1107`。说明 handle 保存什么、cached dispatch 为何限制新路由，以及 consumer 必须等待哪个 event。

**选读：**DeepEP `EventOverlap`、SM 估计；[DeepGEMM 笔记](../notes/repositories/deepgemm.md) 的分离基准。Megatron 对接路径与 DeepEP V2 并非自动兼容，需另做版本测试。

**画/推导：**token→专家→原顺序及反向梯度图，标 shape、索引、权重、所有者。区分逻辑复制、跨 rank/节点字节与 padding，解释最忙专家的影响。

**练习 A：CPU 最小 MoE。** 固定 top-k/权重，以四个线性变换为专家，对照逐 token 与 pack/group/restore 实现。覆盖空专家、重复目的 rank、倾斜与容量裁剪，检查输出和固定路由梯度；不验证 top-k 离散选择的可导性。**GPU 升级：**对照通过且版本兼容后测试多卡 dispatcher，先正确性后延迟。

**练习 B：CPU handle 与负载模拟。** 故意用 route A 的 handle 合并 route B，构造可检测错误；生成均匀、集中、节点受限路由并统计专家/通信账本。**GPU 升级：**掌握 event/所有权后扫 SM budget，比较通信与整层耗时。模拟器不预测 RDMA 吞吐或最佳 SM 数。

**验收：**哪些路由变化改变目标？哪些专家梯度可平均？何时 handle 失效？为何最短 dispatch 未必整层最快？用反例纠正“active 参数决定全部成本”“无 drop 即均衡”“async 即充分重叠”。

**提交产物：**`M06-token-journey.md`、可运行参考实现、正反向对账、三类路由表、真实依赖与概念关系图。**工时：**18–30 小时，GPU 扩展另计 6–12 小时。**下一模块：**补齐 M07 后进入 M08；M09 可并行。

<a id="m07"></a>

## M07 — GPU 内核、低精度与 Profiling

**先修：M05；可与 M06 并行。可跳过诊断：**解释相同 FLOPs 不同速度，手算分块 softmax，给出 dtype 正确而 scale/stride 错误的反例。通过可略读内存层级，保留数值与测量练习。

**知识细目。** HBM、片上存储、寄存器、warp/线程块；算术强度、访存合并、tiling、融合、occupancy 与资源竞争；attention 中间矩阵与 online softmax；低精度格式、scale 粒度、amax、累加和主状态精度；shape/stride/对齐、架构分派；JIT/autotune/graph 冷启动与稳态；CPU launch、GPU kernel、通信重叠的时间线。MFU 与包含额外硬件工作的 HFU 口径要分清，重计算不能偷偷计入“有用模型 FLOPs”。

**核心阅读，按顺序：**

1. 读 [FlashAttention-2 作者论文](https://tridao.me/publications/flash2/flash2.pdf) §2.3 与 §3：只追分块读写、online softmax、重算和工作分配，暂不背具体硬件加速数字。它解释的是精确 attention 的 IO 优化；“精确”不承诺不同浮点归约顺序逐位相同。
2. 按顺序读 Triton 官方 [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 与 [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。逐项标明程序实例、tile、mask 和地址计算；先懂为什么需要边界 mask，再看 autotune 搜索什么。
3. 读 [DeepGEMM 笔记](../notes/repositories/deepgemm.md) 与 [gemm.hpp:73–123](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/gemm.hpp#L73-L123)，再追笔记 G2/G3 的配置生成和编译缓存。为一次调用写出 shape/dtype/layout/scale/架构契约，不要求首次就读完整 CUDA 内核。

**选读：**[TE 低精度原理](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)、[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 的 schedule/export trace；FlashAttention 新架构实现。不同 GPU 代际的 kernel 不能直接互换。

**画/推导：**Q/K/V tile 的内存流向；用共同最大值合并 softmax 指数和及输出；含 host、kernel、event 的 trace 关键路径，不能把重叠区间全部相加。

**练习 A：CPU 数值参考。** 对照整行与 online softmax，覆盖普通、大幅值、mask 与全 mask 输入；简化量化/反量化比较异常值对 per-tensor/per-block scale 的影响，注明并非完整硬件 FP8/FP4。**GPU 升级：**参考及梯度对照明确后测试支持的 Triton/TE 路径，记录误差、异常值、shape 与耗时。

**练习 B：测量审计。** 给 CPU 流程加区间标签并解析 trace；在假设 GPU 时间线上识别错误同步、重复计时与 JIT 污染。**GPU 升级：**固定模型/shape/版本，分别测冷/热调用，采用同步或 device events，报告采样数、分位数及 profiler 开销。一次只改 tile 或一个融合边界，数值通过后再比较；CPU trace 不提供 GPU 带宽。

**验收：**为什么少写矩阵可能比少乘法更值钱？大 tile 为何可能更慢？kernel 正确为何不保证收敛？纠正“支持 FP4 即全流程 FP4”“峰值即训练速度”“profiler 各行相加即墙钟”。

**提交产物：**`M07-kernel-contract.md`、数值参考/误差表、一张内存流图和带口径的 trace 审计；实测另附设备与预热记录。**工时：**20–34 小时，GPU 扩展另计 6–12 小时。**下一模块：**M08；有兴趣的读者可另做单内核 capstone，不能据此跳过训练状态恢复。

<a id="m08"></a>

## M08 — 训练运行、存储、Checkpoint 与容错

**先修：M05，应用 M06/M07。可跳过诊断：**列出权重之外五类状态，解释 staging/持久写入差别，设计检测恢复后数据顺序错误的测试。通过可略读清单，保留故障练习。

**知识细目。** 集群健康与 GPU/NIC/NUMA 拓扑；数据分片、shuffle、packing、prefetch 和 reader 状态；共享存储吞吐、元数据压力与尾延迟；checkpoint shard、manifest、提交点、异步 staging；optimizer/scheduler/RNG/低精度状态；同拓扑恢复与 reshard；overflow skip、失败重放、慢 rank、通信超时、静默损坏；checkpoint 间隔对保存开销和重算损失的取舍。

**核心阅读，按顺序：**

1. 读 [运行章 §7–8](../handbook/02-distributed-pretraining-operations.md)，对照 [TorchTitan dcp.py:580–650](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/components/checkpointer/dcp.py#L580-L650)。追 load、staging future、saving future；注意当前 rank-local RNG TODO，不能把库名当成精确恢复保证。
2. 读 [Megatron RNG 状态:451–516](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/training/checkpointing.py#L451-L516)，再在同文件找 `maybe_save_dataloader_state`。结合 `training.py` 的 update 成功/跳过分支，区分已消费样本、尝试迭代与成功更新，说明为什么它们可以按确定规则分开推进。
3. 读 [3FS Design Notes](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md) 的 Design and implementation、File system interfaces 与 FUSE 限制。画 cluster manager、metadata、storage、client 四类角色及读写路径；把 CRAQ 存储一致性与“所有训练状态属于同一步”的应用一致性分开。这是外部原理阅读，不要求部署 3FS。

**选读：**[NCCL Tests 指标](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)、TorchTitan checkpoint 文档、3FS native client。作者 benchmark 不承诺本机吞吐。

**画/推导：**update、staging、写盘、发布时序及各边界可恢复状态；参数、优化器、RNG、低精度状态的所有者；dataloader 预取与提交游标的区别。

**练习 A：CPU 中断恢复。** 随机采样的小训练器连续跑 K 步，与第 j 步保存后新进程续跑对照；逐项漏存 optimizer、scheduler、样本顺序、RNG/计数，确认能检测差异。只用实验临时目录；重设 seed 不等于恢复随机流。**GPU 升级：**CPU 对照通过后先测小任务同拓扑恢复，再测 reshard；逐位一致与允许误差分别验收。

**练习 B：纸上/CPU 故障矩阵。** 小文件模拟 shard、校验与最终 manifest；注入缺失、截断、未发布、游标超前和慢写，验证拒绝不完整状态。为 overflow skip 与 fail-fast/replay 分别写计数规则。**GPU 升级：**有测试集群及回滚点后并发测读数/保存和指定进程退出；本地模拟不证明 3FS/RDMA 的真实容错性能。

**验收：**权重齐全为何不能保证续训？强一致存储为何不保证训练状态一致？timeout 为何可能掩盖错误？分开可加载/统计复现/逐位相同；区分 overflow skip 与失败重放、consumed tokens 与成功更新。

**提交产物：**`M08-recovery-contract.md`、状态清单、故障矩阵、连续/恢复对照记录、运行与回滚手册。**工时：**18–30 小时，真实集群扩展另计 8–16 小时。**下一模块：**补齐 M09；这些运行规范在 M13 异步 agent RL 和 M15 综合项目中继续使用。

<a id="m09"></a>

## M09 — 推理 Serving 与 Rollout 成本

**先修：M01、M02、M05；agent RL 前必须完成。可跳过诊断：**解释 prefill/decode、TTFT/ITL/吞吐，估算 MHA/GQA KV，解释权重变更后失效。通过可略读术语，保留调度与版本边界。

**知识细目。** 自回归生成、prefill 与逐 token decode；continuous batching、chunked prefill、抢占/排队；KV page、radix prefix、活动引用与驱逐；模型/adapter/权重版本的缓存命名空间；TP/EP 的推理布局；TTFT、ITL、TPOT、尾延迟与请求成功率；冷/热 cache、到达率与并发；rollout 中长轨迹、工具等待、取消和策略陈旧。SGLang 在此承担生成系统，不承担基础模型 optimizer update。

**核心阅读，按顺序：**

1. 读 [SGLang 笔记](../notes/repositories/sglang.md) 和固定版 [Bench Serving 指标:230–255](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/docs/docs/developer_guide/bench_serving.mdx#L230-L255)。给各指标写时间起止点、token 分母和失败样本处理；只输出一个 tokens/s 不能描述服务质量。
2. 沿 [scheduler.py:1893–2022](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/managers/scheduler.py#L1893-L2022) 对比普通与 overlap 循环，再根据笔记追 prefill 合并。画“本批 GPU 执行、上一批结果处理、下一批准入”的依赖，而不是把 overlap 理解成无限并发。
3. 读 [radix_cache.py:400–680](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/radix_cache.py#L400-L680)，追 match、split、lock ref 和 eviction。记录 key、KV 索引与活动请求的关系；由 token 前缀命中到实际省下多少工作，中间仍需测量。

**选读：**SGLang adapters、speculative decoding、权重更新，3FS KV cache。掌握调度后再读 prefill/decode 分离；实现存在不代表默认启用。

**画/推导：**未压缩且 K/V 同维度的 MHA/GQA：`KV bytes≈2×层数×缓存token数×KV头数×head_dim×元素字节数`，另计分片、page 浪费与元数据，不套到 MLA。画共享前缀 radix 树，并拆出轨迹中的模型、排队和工具时间。

**练习 A：CPU 缓存/调度模型。** 用整数 token 前缀树测试共享、取消、释放、驱逐和版本切换，检查活动 KV 不被回收、跨版本不误命中；假设服务时间比较两种准入顺序。**GPU 升级：**不变量通过后用能放入设备的小模型测 cache 开关和请求顺序；模拟命中率不等于真实提速。

**练习 B：纸上成本实验。** 设计“共享/随机前缀 × 冷/热缓存 × 负载”矩阵，固定模型、分词、输出预算与采样，区分到达率和并发限制。CPU 可解析有来源的 trace，自造数据需标明。**GPU 升级：**数值基线通过后记录成功/失败、token、TTFT/ITL、墙钟与排队；不以丢弃慢请求当加速，成功轨迹成本还包含工具和无效采样。

**验收：**prefill 提速为何可能拖慢 decode？同文本为何可能不同缓存键？serving benchmark 为何不证明 rollout 的 policy/logprob 正确？纠正“命中越高越好”“token 便宜即任务便宜”“HTTP API 即 rollout”。

**提交产物：**`M09-serving-budget.md`、KV 账本、调度/引用图、实验矩阵、带口径的成本表；待执行项明确留空。**工时：**16–26 小时，GPU 扩展另计 4–10 小时。**下一模块：**进入 [M10—M14 后训练与 agent 课程](03-posttraining-and-agents.md)，尤其先完成 M10/M11，再把本模块用于 M12 harness 与 M13 异步 agent RL；最终在 M15 串起端到端证据。
