# 前沿专题：核心课程之后怎样继续深入

[返回路线](../ROADMAP.md) · [资源地图](../RESOURCE_ATLAS.md) · [覆盖状态](../COVERAGE.md)

版本：2026-09-08。F01–F06 是有先修、问题和产物的专题研读设计；外部来源已核查入口，尚未全部固定源码或完成实验。它们用于继续深入，不能把此页当作已经完成的六门高级教材。每次只选一个专题，每组先用 6–12 小时做证据阅读与方案（规划估计，实验另计）。

<a id="f01"></a>

## F01 长上下文、新 attention 与混合架构

**先修：**M01/M04/M05，讨论稀疏模型时补 M06；性能结论需要 M07/M09。

按顺序读 [长上下文设计](../handbook/01-data-model-pretraining-design.md)、[DeepSeek-V4 架构与系统报告](https://arxiv.org/html/2606.19348v1)、选读 [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060)。前者给出配方和工程问题，后两者提供不同序列计算设计；不要把它们组合成未经验证的万能架构。

研读问题：预填充和逐 token 解码分别保存什么状态？稀疏选择、KV 压缩或递归状态在哪些信息上作取舍？位置编码和训练长度改变后，远处信息真的被利用了吗？训练、推理与 checkpoint 格式需要什么支持？

**练习：**给定相同序列长度、batch 和 dtype，画两种方案的状态/数据流，逐项计算理论存储并写出省略项；再设计含短任务回归、跨段推理和位置变化的评估。无需 GPU 可完成账本与评估设计，不能报告实测加速。

**交付与验收：**一份架构对照，至少一个不能由复杂度公式回答的问题，以及来源中未公开的部分。能区分算法近似、实现优化和能力证据；之后才选一种固定实现进行小实验。

<a id="f02"></a>

## F02 领域 specialist、蒸馏与 test-time compute

**先修：**M09–M11/M14。先能手算 KL、概率比率并解释采样预算。

先读 [后训练分支](../handbook/03-posttraining-agent-rl-evaluation.md)，再读 [DeepSeek-V4 §5](https://arxiv.org/html/2606.19348v1#S5) 的 specialist 与 on-policy distillation。重点是学生自己采样的轨迹、教师分布、能力整合和执行成本，而不是仅记教师数量。

**练习 A：**在三词元分布上分别计算两个方向的 KL，交换 teacher/student 后解释差异；画示范 SFT、离线蒸馏、on-policy 蒸馏的数据来源。**练习 B：**为一次推理比较预算匹配的单条长生成、多个候选和 verifier 选择，预先定义成功率、成本、失败计数和置信区间。实际模型采样需要单独资源；纸上方案不是能力提升证据。

**验收：**教师更强是否就保证学生更强？增加推理 token、改善 verifier 与改变权重怎样分别对照？量化训练与 rollout 的数值路径怎样影响概率？交付一个可否证的实验设计和 teacher/actor/learner 状态图，再进入小规模整合实验。

<a id="f03"></a>

## F03 图像、视频、音频：扩展文本主干

**先修：**M01–M04/M09/M10，长视频增加 M05/M08。先补 patch、时间采样、视觉/音频编码器等表示概念。

阅读顺序：先看 [Molmo2 官方 README 的训练阶段](https://github.com/allenai/molmo2#training-and-evaluations)，沿 `launch_scripts/pretrain.py`、`launch_scripts/sft.py` 和数据预处理入口提出源码问题；再用 [Qwen3-Omni 作者报告](https://arxiv.org/abs/2509.17765) 和 [官方仓库](https://github.com/QwenLM/Qwen3-Omni) 比较流式音视频输入输出。以上是外部入口，目前未 clone 或运行。

必须明确起点：Molmo2 的公开多模态训练流程以已训练 LLM 和视觉编码器开始。复现这个阶段不等于随机初始化全部权重。Qwen 的推理、模型说明与报告也不能自动补成完整原始训练语料。

**练习：**用自己制作的少量图像/视频说明帧采样、分辨率、时间信息与 token budget；画 preprocessor→encoder→connector→LM→输出的 shape/状态图。把视频切分到不同数据集时，检查同一视频和相邻片段是否泄漏。先做数据与接口审计，不需要下载大模型；若升级训练，先明确哪些组件冻结、哪些更新。

**验收与产物：**一份数据卡、一张多模态样本图、阶段初始化/冻结表，以及空间/时间定位与普通问答分别怎样评分的协议。能区分“输入接得上”“训练能收敛”“模型理解时序”三种证据，再追一条完整源码链。

<a id="f04"></a>

## F04 数据质量、合成数据与数据混合研究

**先修：**M03/M04/M14；在线生成还需 M09，轨迹数据需 M12。

先读 [datatrove 数据处理示例](https://github.com/huggingface/datatrove)，再读 [DCLM 工作流](https://github.com/mlfoundations/dclm#workflow-overview-and-exp-data)，最后回看 [Marin 数据预算案例](../handbook/04-marin-535b-live-case-study.md)。顺序是先理解数据产物，再理解受控比较，最后理解阶段消费预算。

**练习：**构造有完全重复、模板近重复、罕见有效样本和评测变体的小语料。比较两种过滤规则的保留率、误删与漏删；为合成样本增加 teacher/prompt/verifier 标识，设计不依赖老师熟悉题目的保留集。没有教师访问条件时先用手写样本验证流程，不能称为真实合成数据收益。

**验收：**质量分提高但覆盖下降怎么办？增加训练 token 与增加信息量怎样分开？数据比较应固定哪些模型、计算和评估条件？交付处理流程、样本级 provenance、分领域分布与消融预注册，然后做小模型验证。

<a id="f05"></a>

## F05 训练基础设施：存储、编译、调度和运维

**先修：**M05/M07/M08；研究 rollout 调度时补 M09/M13。

顺序读 [Marin 实际运行](../handbook/04-marin-535b-live-case-study.md)、[3FS Design Notes](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md)，再按瓶颈选 [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 或 [NCCL Tests](https://github.com/NVIDIA/nccl-tests)。把训练 step、数据读取、保存和评估放在同一条时间线上。

**练习：**从公开记录制作一个假设事故的因果图，区分观察到的事实、候选解释和下一步取证；估算 checkpoint 的状态总量和读写下界，给缓存失效、节点中断和编译重启各设计一个检查。CPU 模拟只展示调度/状态语义，不能推导真实存储和网络性能。

**验收与产物：**拓扑与数据流、故障 runbook、恢复成功条件和未覆盖风险。能解释为什么单 rack 正常不证明全拓扑稳定、为什么异步 staging 完成不等于持久化完成，再进入受控多机实验。

<a id="f06"></a>

## F06 长期 agent、记忆与持续学习：先明确改变了什么

**先修：**M12–M14，涉及权重更新时必须完成 M11/M13。

从 [Pi 状态与上下文](../notes/repositories/pi.md)、[Verifiers 任务生命周期](../notes/repositories/verifiers.md)、[Prime RL 样本和版本管理](../notes/repositories/prime-rl.md) 进入；这些提供局部机制，不构成完整的终身学习配方。当前领域的长期泛化与稳定在线权重学习仍有大量开放问题。

**练习：**给同一个跨回合任务设计四个条件：固定上下文、外部持久记忆、检索已有资料、实际参数更新。规定存储内容、可见信息、重置条件、时间切分和评估预算；先用受控记录验证信息流，再决定是否需要模型实验。不能因下次任务做得更好就认定模型发生了权重学习。

**验收与产物：**状态归属表、长期评估协议、遗忘/污染/反馈偏差的反例。能区别任务状态积累、上下文内适应、数据检索和 optimizer update，再评审一项公开工作到底解决了哪部分问题。
