# 分布式预训练：从集群验收到可恢复的优化器更新

[返回全流程](00-end-to-end.md) · [数据与配方](01-data-model-pretraining-design.md) · [后训练与 agent RL](03-posttraining-agent-rl-evaluation.md)

本章关注基础模型从随机初始化开始的训练运行面：怎样让大量 GPU 对同一个目标函数持续完成正确更新。它以本仓库锁定的 TorchTitan、Megatron-LM、DeepEP、DeepGEMM 源码为证据，结合 NVIDIA 的公开文档。**“源码事实”只代表所引版本；“工程归纳”是建议采用的验证流程；“未验证”表示本仓库没有执行对应 GPU 或集群实验。**以下不能还原未公开的超大模型完整配方，也不把库提供某项能力等同于某次模型训练实际使用了它。

## 1. 先验收集群，再讨论训练吞吐

**工程归纳。**启动前应冻结一份运行清单：模型与数据版本、tokenizer、容器及依赖版本、GPU 架构与显存、驱动/CUDA/NCCL、网卡固件、GPU—NIC—NUMA 亲和性、节点及交换网络拓扑、存储路径、账号配额。软件版本相同不代表物理链路健康；多份源码 HEAD 能分别编译也不代表组合兼容。

验收应逐级扩大：单 GPU 内存与计算正确性 → 节点内 GPU 互联 → 跨节点通信 → 同时读数据和写 checkpoint 的完整负载。通信测试既要查错误，也要按训练实际消息大小扫延迟和带宽；分别覆盖 TP/FSDP 的集合通信、PP 的点对点，以及 MoE 的 token 分发。NCCL Tests 可作为基础验收，DeepEP 等自定义路径还需专项测试。其 `busbw` 是经集合通信流量模型换算的指标，不能直接当作所有拓扑的训练有效带宽。[NCCL Tests 指标定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

存储验收至少测三件事：随机/顺序读取数据分片的持续能力、所有 rank 同时保存状态的突发写入、从这些状态重建训练的时间。检查本地缓存未命中、共享目录元数据压力、后台上传积压和磁盘空间。只测一个大文件顺序读，不能证明 dataloader 或分布式 checkpoint 不会拖住训练。通过标准应在试运行前约定，而非看到一次成功启动后临时降低要求。

## 2. 并行是数据与状态的布局，不是一串可随意相乘的参数

| 维度 | 切分对象 | 主要代价与交互 |
|---|---|---|
| DP | 不同样本，梯度共同更新一份逻辑模型 | 梯度同步；分片 DP 还涉及参数/优化器状态通信 |
| TP | 同一层的张量计算 | 高频集合通信；与序列并行、算子布局绑定 |
| PP | 不同层 | 激活及其梯度跨阶段传递；流水线空泡和阶段不均衡 |
| CP | 同一样本的上下文 | 减少本地长序列负担；attention 需要跨分片交换信息 |
| EP | 不同专家 | 根据路由移动 token；专家负载及返回合并决定尾延迟 |
| FSDP/分布式优化器 | 参数、梯度、优化器状态的存储 | 用通信换显存；不是独立增加样本的维度 |

**源码事实。**TorchTitan 当前 dense mesh 满足 `N = D_replicate × D_shard × CP × TP × PP`，独立数据批次的 DP 度为 `D = D_replicate × D_shard`。同一组 GPU 的 sparse view 则是 `[PP, D_replicate, EFSDP, EP]`；EP 必须整除 `D_shard × CP × TP`，因此不能在前式后再乘 EP。这个等式属于该实现，不能冒充所有框架、专家 TP 或异构部署的统一公式。[TorchTitan mesh 定义](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/distributed/parallel_dims.py#L106-L240)

**工程归纳。**固定长度批次可先用 `B_sequence = D × b × m` 对账：`b` 是每个 DP 副本每个 microbatch 的序列数；`m` 是一次 optimizer update 实际累计的全部 microbatch 数。框架若同时暴露 PP microbatch 数和梯度累积组数，应先核对二者乘积是否就是这里的 `m`。TP、PP、CP 对同一批数据协作，不重复增加样本。

例如纯假设的 256 GPU、`TP=4, PP=4, CP=2, D=8`，若 `b=1,m=16,S=8192`，则每次更新有 128 条序列、最多 1,048,576 个预测位置。padding、文档边界及 loss mask 会使有效目标 token 更少；sequence packing 后序列数也不等于文档数。此例只校验算术，不推荐某套硬件配方。布局决策应把最频繁通信放在适合的互联域内，同时测 PP 空泡、MoE 跨节点流量和显存峰值，不能只优化单一通信维度。

## 3. 一次训练更新究竟流过哪些状态

**工程归纳。**真正的第 0 步还包括建立进程组、分配模型分片和随机初始化：逻辑上相同的参数副本要一致，不同参数分片不能因为 seed 使用错误而成为重复矩阵。核对初始化分布、共享权重及参数有限性，在最终参数布局上创建优化器，记录初始随机流和数据起点。改变初始化后不能继续沿用旧优化器动量；分布式初始化与单卡初始化是否等价应单独验证。

**源码事实与工程归纳。**逻辑上是一条链，物理上很多阶段重叠：

`确定本步数据 → 前向/损失 → 反向/累积/通信 → 完成梯度同步 → 数值检查与裁剪 → optimizer → scheduler → 可恢复进度`

1. **确定数据与分母。**读取 token IDs、位置/attention 信息及 label mask，按自回归任务构造下一 token 目标。记录本步消费的数据位置，清空上一步梯度。不能把各 rank 的平均 loss 再简单平均：有效 token 数不同会改变样本权重。TorchTitan 先收集全部累积组与 PP microbatch 的有效 token 数，在 batch mesh 汇总，再把同一个分母交给前向/反向。
2. **前向。**必要时 all-gather 参数分片；经过 embedding、attention、MLP/MoE，产生 logits 与交叉熵。TP/CP 在层内交换信息，PP 按调度传激活。开启 activation checkpointing 时，只保留选定中间结果，其他激活在反向重算。节省的显存有相应计算成本。
3. **反向与累积。**梯度从损失向前传播；PP 反向传激活梯度，TP/CP 完成各自要求的归约。DP 梯度 bucket 可以边生成边 reduce-scatter/all-reduce，因此时间线上通常不存在一个完全孤立的“最后才同步”阶段。延迟累积期间的同步还受实现限制：TorchTitan 对 HSDP replicate all-reduce 的最后一组切换，就显式考虑了 CUDA graph 捕获条件。
4. **收尾梯度。**等待未完成通信，并处理共享 embedding、非 TP 参数或专家副本所需的特殊同步。全局梯度范数必须按唯一逻辑参数计算，避免重复计算副本或遗漏分片。使用 loss scaling 时，检查溢出并反缩放后，才能按正确单位裁剪。
5. **更新与提交进度。**检查 loss 和梯度是否有限；通过检查后执行优化器更新，再按既定规则推进学习率。TorchTitan 在 update 前检查全局有限性，并等待 checkpoint staging 完成，防止尚未复制完的状态被下一次更新覆盖。故障后终止 / 重放与预期的 overflow skip 是不同策略：分别记录已消费 token、成功更新和跳过次数，按明确契约推进并保存数据游标、scheduler 与计数，不能混用它们的含义。

上述顺序可直接沿 [TorchTitan `train_step`](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/trainer.py#L860-L1003) 阅读。Megatron 的 [梯度收尾逻辑](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/distributed/finalize_model_grads.py#L560-L713) 则展示 `finish_grad_sync`、特殊参数同步及按有效 token 数缩放的另一种实现。**二者不是可以逐项拼接的同一训练循环**：应先确认当前 backend 的损失归一化和梯度缩放位置，避免重复除以 DP 度或 token 数。

以 AdamW 为例，update 会用当前梯度更新一阶/二阶矩，经偏差修正构造参数步长，并按配置执行解耦权重衰减。分布式优化器可让不同 rank 只持有并更新自己的状态分片，再为后续计算交换所需参数。这里的状态所有权和通信完成顺序是关键；其他优化器有不同状态与更新规则，不能假设所有前沿模型都使用 AdamW。

一个具体计数边界是 Megatron：此路径只在成功更新时推进 scheduler，但外层循环仍推进 iteration 与已消费样本数。因此，已消费样本不能直接当作所有样本都贡献了参数更新。[Megatron scheduler / skip 分支](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/training/training.py#L3309-L3323)、[iteration 与样本计数](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/training/training.py#L4887-L4920)

## 4. MoE：计算稀疏后，通信与路由进入核心路径

**源码事实。**Megatron 的 MoE forward 把路由、分发、专家计算、合并组织成可分离阶段。读者应追踪的对象是：每个 token 选择的专家 ID、路由权重、重排索引、每个专家接收的 token 数，以及把专家输出送回原 token 位置的元数据。[Megatron MoE 前向](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/core/transformer/moe/moe_layer.py#L637-L742)

**工程归纳。**典型路径为 router → top-k/容量处理 → pack → dispatch → 专家 grouped GEMM 与激活 → combine → 按路由权重还原输出；具体加权位置由实现约定。反向也必须穿过重排和通信。EP 的专家权重是不同参数，不能把任意 EP rank 的专家梯度直接平均；应找出同一专家的 DP 副本组。

容量不足、drop/padding、负载平衡损失和 router 精度既影响吞吐，也改变优化过程。除了平均 token 数，要记录最忙专家、空专家、丢弃比例及跨节点目的地分布。总参数量决定部分存储成本，每 token 激活的专家影响计算量；仅用总参数量或 active 参数量都不足以预测整步耗时。

DeepEP 负责这条路径中的高效分发/合并。它的 SM 资源估计函数明确假设均衡路由，并排除特定 group-limited gate：这说明通信优化受实际分布限制，不能直接把理论带宽或 SM 建议搬到另一套路由上。[DeepEP 带宽模型边界](https://github.com/deepseek-ai/DeepEP/blob/01dc3aaac82068020353dce2c302e38153c0bfaa/deep_ep/buffers/elastic.py#L729-L834) 通信 kernel 与专家 GEMM 争用 SM、HBM 和链路；增加重叠后，应重新检查整步尾延迟和显存，而非只看通信 kernel 变短。

## 5. 低精度必须经过数值与收敛验收

**源码事实。**DeepGEMM 的 FP8/FP4 接口同时检查矩阵布局、输出类型、scale 格式与架构，再选择实现；“把 tensor cast 成低精度”远远不够。[DeepGEMM 接口契约](https://github.com/deepseek-ai/DeepGEMM/blob/559d79fb6994a58b8a15b4b93bf13ccc16edf247/csrc/apis/gemm.hpp#L73-L123) Transformer Engine 的公开说明进一步区分不同低精度格式、scale 粒度、amax 历史及转置量化要求；这些共同构成数值配方。[Transformer Engine 低精度说明](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

**工程归纳。**验收依次扩大：先用高精度参考检查代表性形状的输出与梯度误差，再比较小模型相同数据上的短程 loss/梯度曲线，最后在代表性规模、序列长度和 token 预算上看验证集质量与稳定性。检查 outlier、溢出/下溢、scale 更新和专家负载极值；参数、激活、梯度、归约、优化器状态可以采用不同精度。阈值必须按对象预先定义，不能只要求“没有 NaN”。

scale/amax 等持久状态若影响下一步，就要纳入恢复契约。敏感算子保留较高精度是否必要应实验决定。kernel benchmark、若干步 loss 正常、完整预训练达到目标质量，是三级不同证据。本章没有执行这些实验，也没有据此宣称 FP4 已替代某个超大模型训练的全部数值路径。

## 6. 测量整步的有效产出

**工程归纳。**先分开冷启动、JIT 编译/autotune、图捕获与稳定运行；再按 rank 记录 step time 分布，找最慢 rank 的关键路径。观测输入等待、CPU 调度、H2D、算子、集合通信、PP 空泡、重算、保存状态的等待和写入积压。只对单 rank 求平均会掩盖每步 barrier 前的拖尾。

至少同时报告：有效训练 token/s、总处理 token/s、每步耗时分位数、显存峰值，以及包含失败重算和 checkpoint 开销的长期有效 token/墙钟时间。计算 MFU 时写出模型 FLOP 估计、硬件峰值所用精度及 MoE/重算的计数约定；不能通过更换分母制造“提升”。优化一次只改变少量因素，保存前后 trace、配置、形状和误差结果，区分瓶颈被消除还是转移。

SGLang 主要位于推理和 RL rollout：其请求调度、KV cache 和 prefill/decode 优化值得借鉴，但不承担这里的基础模型 optimizer update。它的 serving tokens/s 也不能与预训练 tokens/s 直接比较。参见 [SGLang 源码学习笔记](../notes/repositories/sglang.md)。

## 7. Checkpoint 是一致的训练状态，而不只是权重文件

**工程归纳。**一份用于继续训练的状态至少要说明：模型参数；优化器动量/方差及必要的 master 参数；scheduler 与成功更新数；已消费 token 和数据采样/混合阶段；dataloader 游标、shuffle/packing 状态；各随机数流；低精度持久状态；配置、源码和 tokenizer/data manifest。记录 rank 到逻辑分片的映射，并验证所有必要 shard 完成后才发布“可恢复”标志。具体字段随算法和框架变化，应通过恢复实验确认。

异步保存有两个完成点：GPU 状态复制到稳定 CPU 缓冲完成，以及后台写入持久存储完成。前者允许更新继续，后者才决定节点丢失后还能恢复什么。它们不能共用一个“保存完成”的仪表盘指标。TorchTitan 对 staging 与 saving 的等待有独立实现；其当前加载路径同时保留保存 rank-local RNG 的 TODO，因此不能声称该快照已经保证精确续训。[TorchTitan 恢复与异步状态](https://github.com/pytorch/torchtitan/blob/4e20e76b235ec415bd81912ce09eed46fa39f717/torchtitan/components/checkpointer/dcp.py#L580-L650)

Megatron 明确收集 Python、NumPy、Torch、CUDA 以及 TP RNG tracker 状态，说明“设相同 seed”不能代替恢复随机数流。[Megatron RNG 状态](https://github.com/NVIDIA/Megatron-LM/blob/c9b53d0a87cb926f47115259593ecfeb351ca29f/megatron/training/checkpointing.py#L451-L516) 但保存 RNG 本身也不保证所有 kernel 逐位确定。

恢复验收应对比“连续运行”与“保存—终止—恢复”：后续样本 ID、mask、学习率、优化器状态、loss/梯度及参数差异。先做同拓扑测试，再做改变并行布局的 reshard 测试。可加载权重、可继续优化、统计上可复现、逐位一致是四种不同承诺；改变世界大小会改变归约顺序和数据分配。模型导出文件不能默认作为完整训练 checkpoint。

## 8. 投产关卡与故障处理顺序

**工程归纳；以下均为待执行验收，不是本仓库实验结果。**

| 关卡 | 必须留下的证据 |
|---|---|
| G0：环境/数据冻结 | 版本与拓扑清单、数据校验、通信正确性及读写验收 |
| G1：最小数值基线 | loss mask、归一化、梯度和低精度误差对照 |
| G2：小规模多卡 | 并行后与参考更新的误差；MoE 分发/合并与梯度对照 |
| G3：目标规模试运行 | 稳态 trace、慢 rank、存储并发、保存及恢复演练 |
| G4：代表性训练试验 | 固定预算下验证集曲线、数值稳定性与吞吐/成本记录 |
| G5：持续训练 | 质量监控、异常阈值、可恢复状态、数据和配置变更审计 |

发生故障时先冻结故障步、首个报错 rank、配置、数据位置和最后完整 checkpoint 的证据，再分类处理：

- **慢 rank/通信超时：**对齐各 rank 时间线，区分输入阻塞、GPU 降频/错误、路由倾斜、NIC 链路和集合通信顺序不一致。超时 rank 未必是根因；只调大 timeout 会隐藏问题。
- **NaN/Inf 或 loss 突变：**阻止错误梯度进入 update。对预期 loss-scaling overflow，按事先定义的 skip / scale 调整策略处理，并独立统计；对未预期或持续异常，从健康状态用相同数据缩小到首个异常层，检查 scale、归一化、梯度范数和数据异常。必要时暂时提高精度以定位，再比较修复后的轨迹。跳过更新、推进数据和推进 scheduler 的规则应显式区分。
- **节点退出或静默数据损坏：**隔离疑似节点并重建一致进程组；默认流程应由完整有效状态恢复。单独重启某个 rank 需要专门的容错协议，不能假设普通同步训练天然支持。重放/冗余检测也有覆盖范围，不能以一次检查通过证明所有算子安全。

恢复后先跑约定的健康窗口，核对样本序列、梯度、学习率、路由和吞吐，再放行长程训练。每次变更留存原因、影响范围、回滚点和对照证据；只有正确更新、模型质量与长期有效产出同时满足要求，才算通过运行验收。
