# 资源地图：在哪个阶段读，为什么读，读到哪里停

核查日期：2026-09-08。课程顺序见 [ROADMAP](ROADMAP.md)。这里按学习用途组织资源，不按热度或 stars 排名。第一轮每模块只选核心材料，其余按问题选读。

**状态说明：**“固定源码”表示已克隆、登记 SHA 并有局部笔记，不表示完整运行过；“外部入口”表示本轮核实了官方页面及学习用途，没有纳入 20 个源码快照，也没有安装或完成其作业。论文的结论限于论文实验条件；模型报告、推理 demo 与完整训练配方分别标识。

<a id="1-主干19-个固定源码项目"></a>

## 1. 主干：20 个固定源码项目

| 项目 / 固定源码笔记 | 第一次进入 | 最值得带着读的问题 | 第一轮停止条件 |
|---|---|---|---|
| [SmolLM](notes/repositories/smollm.md) | M03–M04 | 数据、阶段、长度与预算如何一起改变？ | 解释两份配置的差异，算清 token budget |
| [OLMo-core](notes/repositories/olmo-core.md) | M01–M04 | 一条样本怎样变成 labels、loss、update 和 checkpoint？ | 把正式 recipe 跟进一条真实训练路径 |
| [Marin](notes/repositories/marin.md) | M03–M04，M08 | 如何保存研究决策、数据 lineage、预测和运行身份？ | 跟清一个 artifact DAG，再读 535B 案例 |
| [TorchTitan](notes/repositories/torchtitan.md) | M05 | PyTorch 训练中的 mesh、归一化与状态怎样组合？ | 画 rank 布局、定位梯度与状态交接 |
| [Megatron-LM / Core](notes/repositories/megatron-lm.md) | M05–M06 | TP/PP/CP/EP 与分布式优化器的真正边界？ | 跟清一个后端，不通读全部分支 |
| [DeepEP](notes/repositories/deepep.md) | M06–M07 | 专家分配怎样跨设备送出、计算、返回？ | 解释 shape、stream、元数据与兼容接口 |
| [DeepGEMM](notes/repositories/deepgemm.md) | M07 | layout、量化 scale、JIT 与 grouped GEMM 如何配合？ | 选一个 kernel 契约，列数值和性能验收 |
| [SGLang](notes/repositories/sglang.md) | M09 | prefill/decode、KV cache 与调度怎样改变 rollout 成本？ | 跟一条请求和一次 cache 生命周期 |
| [Open Instruct](notes/repositories/open-instruct.md) | M10–M11 | SFT、偏好、reward 与 RL 的目标怎样实际实现？ | 手算一个 mask / probability ratio 例子并找到消费位置 |
| [verl](notes/repositories/verl.md) | M11，M13 | 算法 driver 与计算后端如何分工？ | 固定一种算法/后端追一次 update |
| [Harbor Cookbook](notes/repositories/harbor-cookbook.md) | M12 | 什么样的现实任务有可检验反馈？ | 一个任务与奖励向量的契约 |
| [Harbor](notes/repositories/harbor.md) | M12，M14 | 环境、agent、verifier 的生命周期怎样可靠结束？ | 正常、超时和失败分支都能解释 |
| [Verifiers](notes/repositories/verifiers.md) | M12 | task / harness / runtime / rollout 谁拥有状态？ | 把一次 rollout 跟到评分与 artifact |
| [Pi](notes/repositories/pi.md) | M12 | 事件、工具、上下文和 durable session 如何衔接？ | 跟一次受控工具调用与取消/恢复问题 |
| [Claude Code / Agent SDK](notes/repositories/claude-agent-sdk.md) | M12，在 Pi 之后 | SDK 控制协议、历史 harness 快照与当前行为怎样区分？ | 追权限回调，解释 compaction 与恢复，并通过 CPU 状态机实验 |
| [DeepSeek Harness](notes/repositories/deepseek-harness.md) | M12 | 日志如何投影成模型请求？ | 指出上下文与日志必须满足的不变量 |
| [Prime RL](notes/repositories/prime-rl.md) | M11 概览，M13 深入 | 任务生成、分组、过滤、队列和训练如何连接？ | 一条样本的 reward/token/version 到 learner |
| [slime](notes/repositories/slime.md) | M13 | 多轮、分叉、compaction 怎样变成训练样本？ | 共享前缀和 loss mask 的反例 |
| [Miles](notes/repositories/miles.md) | M13 | token、expert 路由、精度和策略版本怎样保持对应？ | 对一种不一致给出定位和验收方法 |
| [APEX / SkyRL recipe](notes/repositories/apex-agents-skyrl-recipe.md) | M13 综合 | 长程知识工作任务如何接入训练？ | 识别 TITO/Harbor/SkyRL 交接及未公开数据缺口 |

实际的软件关系及 pin 不同问题见 [REPO_RELATIONSHIPS](REPO_RELATIONSHIPS.md)。例如下游使用 Megatron Core，不代表运行 Megatron 顶层脚本；独立 DeepEP HEAD 也不保证兼容所读 adapter。

## 2. 补足基础、数据与系统的外部入口

以下不是必须一次读完的新清单；它们各自填补一个明确先修或证据缺口。

| 一级资源 | 模块 | 优先读什么 / 为什么需要 | 状态与边界 |
|---|---|---|---|
| [Stanford CS336 2025](https://cs336.stanford.edu/spring2025/) | M00–M05，M10 | basics、systems、scaling、data、alignment 的精选讲义与作业 | 外部入口；课程自身有数学/深度学习先修。本路线补桥接，不要求直接做完整课程 |
| [D2L 线代](https://d2l.ai/chapter_preliminaries/linear-algebra.html)、[微积分](https://d2l.ai/chapter_preliminaries/calculus.html)、[概率](https://d2l.ai/chapter_preliminaries/probability.html) | M00 按需桥接 | 只补诊断中缺失的矩阵乘法、导数、条件概率 | 作者教材入口；按 M00 出口题决定何时停止补课 |
| [PyTorch Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | M01–M02 | 计算图、梯度累积与停止记录梯度 | 外部入口；在线版本会更新，实验另记环境 |
| [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) | M04 | 训练设计、消融与数据/模型决策 | 作者工程总结；与 SmolLM 固定配置对照 |
| [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | M05–M08 | 并行、重计算、通信与显存的图解 | 外部入口；其中性能结果不是自己的机器实测 |
| [nanotron](https://github.com/huggingface/nanotron) | M04–M05 选读 | SmolLM 配方背后的训练执行器 | 外部源码入口，尚未 clone；运行须对齐 recipe 版本 |
| [datatrove](https://github.com/huggingface/datatrove) | M03，F04 | 从 `examples/fineweb.py`、MinHash 示例看数据处理阶段与产物 | 外部源码入口；读 pipeline 不等于获得完整语料 |
| [DCLM](https://github.com/mlfoundations/dclm) | M03–M04，F04 | 源选择→处理→tokenize/shuffle→train→eval，以及 reference JSON 记录 | 外部源码入口；真实规模涉及数据与计算条件 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | M07 | 结合作者论文理解 IO、tiling、重计算与 attention | 外部源码入口；接口与支持硬件按具体版本确认 |
| [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html) | M07 | 先 vector add，再 fused softmax / matmul | 外部教程；只在支持的 GPU 环境做性能练习 |
| [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) / [NCCL Tests](https://github.com/NVIDIA/nccl-tests) | M05，M08 | collective 输出语义，消息大小和带宽口径 | 外部入口；不覆盖所有自定义 EP 通信 |
| [DeepSeek 3FS](https://github.com/deepseek-ai/3FS) | M08，F05 | Design Notes：训练数据和状态的存储数据流 | 外部源码/设计入口；不要求先搭存储集群 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | M03，M14 | task 定义、prompt、计分与可复现评估 | 外部源码入口；与 agent 的运行 harness 职责不同 |
| [Inspect](https://inspect.aisi.org.uk/) | M12–M14 | Dataset / Solver / Scorer、日志、sandbox、预算与错误处理 | 外部官方文档；实际 eval 另需模型或受控 mock |

**课程版本陷阱：**CS336 年度页面与作业仓库 `main` 不一定是同年版本。当前 scaling 作业入口存在跨年度变化和 API 访问条件，因此模块里优先给固定讲义及本地可完成的替代题；不能因为公开了网页就假定所有教学服务可访问。

## 3. 论文与作者记录怎样搭配代码读

| 资源 | 放在哪里 | 读完要回答 |
|---|---|---|
| [Chinchilla](https://arxiv.org/abs/2203.15556) | M04 | 计算最优假设是什么，哪些变量/成本没有被统一覆盖？ |
| [Delphi](https://openathena.ai/blog/delphi/) | M04 | 拟合和验证规模怎样隔离，首次失败后修正了什么？ |
| [Marin 535B 公开 tracker](https://github.com/marin-community/marin/issues/8435) | M04，M08，M15 | 什么是预先预测、条件性计划、代码现状与实际结果？ |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | M10 | 示范、偏好和 reward model 分别提供什么信号？ |
| [DPO](https://arxiv.org/abs/2305.18290) | M10 | 目标中的 reference、logprob 与偏好标签各起什么作用？ |
| [PPO](https://arxiv.org/abs/1707.06347) / [DeepSeekMath](https://arxiv.org/abs/2402.03300) | M11 | policy ratio、优势、clip 与 GRPO 分组如何推导？ |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | M11 | 不同起点和阶段解决什么行为/探索问题，而非记训练顺序？ |
| [DeepSeek-V4](https://arxiv.org/html/2606.19348v1) | F01–F02 | 长上下文架构、数值、领域训练与能力整合怎样共同设计？ |
| [LoRA](https://arxiv.org/abs/2106.09685) / [QLoRA](https://arxiv.org/abs/2305.14314) | M10 后选读 | 哪些参数更新，冻结/量化状态怎样影响资源与表达能力？ |
| [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060) | F01 选读 | 序列混合与状态更新有哪些不同代价和归纳偏置？ |

论文阅读限制为当前模块的问题；理论推导先在小张量上复核，再追代码的实际 reducer、mask 和通信位置。较早的原始论文用于建立概念，不暗示它仍是今天每个实验室的默认最优配方。

## 4. 多模态扩展的两个不同证据入口

- [Molmo2 官方训练仓库](https://github.com/allenai/molmo2)：有阶段训练、数据处理与评估入口，适合 F03 追踪。其公开 pipeline 从已训练的 LLM 与视觉编码器开始；“多模态预训练”不能误写成全部权重随机初始化。具体训练/数据内容继续按所选版本审计。
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) 与 [作者报告](https://arxiv.org/abs/2509.17765)：用于理解文本/图像/音视频接口与 Thinker–Talker 等架构决策。发布权重、推理示例和技术报告，不自动意味着完整训练原始数据和全部运行配方都公开。

这两项是外部入口，未纳入本仓库固定 clone。不是让读者立刻启动多模态大模型，而是补上主干文本课程之外的能力地图。

## 5. 为什么不继续无限加 repo

一个资源只有在能回答“补哪一个问题、先修是什么、读哪一段、产出什么证据”时才进入核心路线。新资源先登记用途与公开程度；完成固定版本审读后再成为核心源码。这样既保留前沿更新入口，也避免资源数量增长快于理解深度。

新增规则见 [贡献指南](CONTRIBUTING.md)；未覆盖的领域见 [覆盖地图](COVERAGE.md)。

## 6. Claude Code：先读机制，再检查历史实现

学习顺序是 [M12 进阶讲解](handbook/05-claude-code-harness.md) → [官方 SDK 与历史快照审读](notes/repositories/claude-agent-sdk.md) → [CPU 状态机实验](experiments/harness-state-machine/README.md)。历史镜像中的局部源码观察不等于当前产品行为；官方 SDK 是可阅读的接口与 transport 实现，完整 harness 运行依赖另行发布的 CLI。
