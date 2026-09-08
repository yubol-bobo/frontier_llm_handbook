# M00–M04｜从 Python 到能审查预训练配方

设计与链接核查：2026-09-08。这是面向自学者的**课程设计**，不是个人学习完成记录，也不是某家实验室的内部培训材料。本文的十项练习及 GPU 升级均尚未执行；其他独立 lesson 若有运行证据，以其各自记录为准。阅读过源码、跑通小例子、完成性能验证与复现公开模型，是四种不同成果。

默认会 Python 函数、类、文件和常见容器，不要求研究背景。顺序为 **M00 → M01 → M02 → M03 → M04 → M05**。CPU 足以完成本篇验收；部分练习需另行准备 CPU 版 PyTorch，本文不安装环境。工时估计包含核心阅读和两项 CPU 练习，全套原作业及 GPU 复现另计；均为规划，玩具结果不能推断前沿质量。

Stanford CS336 2025 原课程要求深度学习、PyTorch、线代、微积分与概率。本课程增加桥接，**不声称 Python 熟练者可以直接完成整套原作业**。[2025 官方归档](https://cs336.stanford.edu/spring2025/) 是课程入口；其中 GPU 报价已是历史资料。先完成核心阅读和练习，再按不足选读，无需预先读完所有论文。

保留自己的解释、计算、源码位置与未解决问题。练习可放个人目录，以模块 ID 命名；所列产物尚未生成。可跳过重复学习，不能跳过验收证据。

<a id="m00"></a>
## M00｜入门诊断与复现素养

### 先修与可跳过诊断

只需 Python 基础。先尝试解释一个 `B×T×D` 数组的三个轴，手算两个长度三的向量内积，解释概率和为何为一，并指出函数输出随输入微小变化的含义。不会某项就先补该项，不必先完整修读一门数学课。能够记录环境版本、用断言检查形状、定位一个函数的调用者，并区分“计划结果”和“实测结果”的学习者，可直接做本模块验收；进入 M01 前仍须通过下面的数学出口检查。

数学桥接依次补向量与加权求和、条件概率、均值与方差、局部变化率；M01 再把链式法则用于模型计算图。暂时不理解证明时，先标记假设和输入输出，不只背符号。

**只补诊断失败的部分。**以下是按需启用的桥接，不要求通读 D2L，也不替代后面的三项核心阅读。先尝试出口题，能独立手算并解释者跳过对应项；卡住时只读指定概念，复做题目后停止。

| 诊断中卡住的地方 | 作者教材入口与本次阅读范围 | 读到哪里可以停 |
|---|---|---|
| 不会内积、矩阵乘法或 shape 推理 | [D2L 线性代数](https://d2l.ai/chapter_preliminaries/linear-algebra.html)：标量、向量、矩阵、内积、矩阵乘法 | 能把输出的一项写成输入的加权和，解释内维为何相同；暂不学特征分解 |
| 不清楚导数、偏导和参数变化 | [D2L 微积分](https://d2l.ai/chapter_preliminaries/calculus.html)：导数、梯度、链式法则的小例子 | 能沿两步函数传递变化率，区分函数值与导数；本模块不要求积分或矩阵微积分证明 |
| 混淆联合概率与条件概率 | [D2L 概率与统计](https://d2l.ai/chapter_preliminaries/probability.html)：样本空间、联合与条件概率、期望和方差 | 能说明条件改变了分母，解释均值为何不能表达全部波动；暂不学复杂分布推断 |

可先用纸笔和 Python 列表计算；教材里的张量示例用于对照，不必为这段桥接安装整套配套环境。把第一次错在哪里、改用了什么分母或维度写进先修缺口清单，比反复观看同一解释更有用。

### 学习目标细分

- 分清模型权重、训练配置、代码快照、数据版本、日志、评估协议，知道缺少哪项会限制复现。
- 理解训练集用于更新，开发集用于选择，保留集用于较少接触的泛化检查；随机种子不能消除数据泄漏。
- 把“能运行”“数值正确”“与原文设置一致”“复现作者成绩”分开报告，并能为一个结论指出证据位置。
- 能解释 tensor 的 shape、dtype、device；区分参数、激活、梯度和优化器状态，暂时不要求计算大模型显存。

### 核心阅读：按顺序，最多三项

1. [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)：先读 Tensors，再看模型、优化与保存的目录结构。遇到图像示例只追张量和训练循环，不把图像任务当成 LLM 配方。
2. [一枚 token 到一次更新](../lessons/01-one-token-to-update.md)：先跟完一个具体样例，再把术语映射到 [全流程手册第 8 节](../handbook/00-end-to-end.md)。暂不要求理解分布式细节。
3. [OLMo-core 源码笔记](../notes/repositories/olmo-core.md)：只读快照、入口与证据边界；到源码确认 `get_labels` 是函数，训练报告中的 token 数是运行元数据，两者回答不同问题。

**选读：**CS336 归档页的先修与课程结构。目标是判断自己缺哪块基础，不以“看完所有课”作为开始练习的条件。

### 应画或推导的对象

画一张 `原文 → token IDs → logits → loss → gradients → 参数更新` 图，每条边标数据类型和所有者。再画一张实验记录关系图，将代码 SHA、配置、数据校验值和结果文件连接起来。遇到“loss 下降”，在图上指出这句话还缺评估分布与比较对象。

### 两个 CPU 练习

**E00-A：建立一份可复核记录。**用 Python 标准库读取自己写的十条短句，保存规范化规则、文件 SHA-256、记录数和生成时间；建立一个含 seed、数据标识、目标与停止条件的 JSON。改变一条短句并重新计算，确认记录反映了内容变化。交付原始记录和修改后的差异；不要把当前日期或文件名当成内容身份。

**E00-B：复现一次统计判断。**生成两组模拟分数，计算均值、样本标准差，画出每个点，解释均值是否足以证明改进。构造“平均分上升但子类下降”的例子，明确标为合成数据。为一条仓库事实写“声称什么／来源／能验证什么／不能验证什么”四格卡。

**GPU 升级：**本模块没有必要的 GPU 练习。把记录规范迁移到未来 GPU 实验即可。

### 验收问题与错误理解

不用提示能否回答：为什么相同 seed 不保证跨版本逐位相同？一份可推理权重为什么可能无法完整续训？同名数据目录为什么不足以定义相同实验？若答复只是“固定随机数即可”，回到证据关系图。把 README 描述直接写成个人已验证结果、把单次分数当作确定收益，均视为未通过。

**进入 M01 的数学出口：先手算，再核对。**三题均为教学构造，不是模型实验；代码只能用于事后检查。

1. 令 `A=[[1,0,2],[0,1,1]]`、`B=[[1,2],[3,4],[5,6]]`。写出各自 shape、`AB` 的 shape，并展开一个元素的求和。参考结果为 `[[11,14],[8,10]]`；只报结果而不能解释内维和输出轴不算通过。
2. 一百项模拟任务中，四十项是代码任务，其中三十项成功。求“已知是代码任务时成功”的概率，以及“既是代码任务又成功”的比例。参考值分别为 `0.75` 和 `0.30`；必须说明分母为何分别为四十和一百，不能只套公式。
3. 令 `f(w)=(2w−1)²`，在 `w=1` 求函数值和导数，再预测参数略微增大时函数怎样变。参考值为 `1` 和 `4`；应能拆成内外两层解释链式法则，并说明导数不是任意步长下的精确变化量。

三题都能独立解释，且复现验收通过，才进入 M01。错一题只回对应桥接；换一组数字再做，避免记住答案冒充掌握。首课的手写梯度与有限差分可辅助第三题，但不等同于已经掌握 autograd 或完成 decoder。

**产物：**`M00-evidence-card.md`、两个数据版本的 manifest、统计脚本与图、先修缺口清单。**建议工作量：**8–16 小时；数学或 PyTorch 桥接另留 12–24 小时，均为规划。**下一模块：**[M01](#m01)。

<a id="m01"></a>
## M01｜从零写出一个 decoder，理解 tensor 与 autograd

### 先修与可跳过诊断

完成 M00；会矩阵乘法、条件概率和局部导数即可开始。先在纸上追踪 embedding、Q/K/V、attention 输出和词表 logits 的形状。能独立实现一个小型因果 decoder、证明未来 token 不影响过去输出，并解释反向传播与梯度累积的学习者，可只交练习和源码对照，不重复逐节阅读。

### 学习目标细分

- 从 `p(x₁…x_T)=∏p(x_t|x_<t)` 理解 next-token prediction；区分训练时同时计算多个位置，与生成时逐步采样。
- 理解 embedding 查表、线性投影、多头拆分、缩放点积、因果 mask、残差、归一化、MLP 和输出头各自改变什么。
- 能追踪 `[B,T] → [B,T,D] → [B,H,T,d] → [B,H,T,T] → [B,T,V]`，解释转置和广播不会凭空创造语义。
- 理解 autograd 记录的是实际运算图，参数共享会累积梯度；`detach`、无梯度上下文、清零时点会改变更新。
- 将“从零写关键计算”与“从随机权重训练”分开；前者用于学习，后者可以使用现成框架。

### 核心阅读：按顺序，最多三项

1. [CS336 2025 Assignment 1 官方讲义](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf)：本模块只选模型架构与训练循环部分；BPE 留到 M03。可对照官方仓库的 `tests/adapters.py` 理解接口，不要求一次完成全部作业。
2. [PyTorch autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)：追一次 forward、backward 和梯度累积；先自己写出预期梯度，再运行验证。
3. [OLMo Transformer.forward](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/nn/transformer/model.py#L523-L610)：按 embedding → blocks → lm_head 读，与 [已有笔记](../notes/repositories/olmo-core.md) 对照。先跳过 CP、TP、compile 分支，只注明其接口边界。

**选读：**[Attention Is All You Need](https://arxiv.org/abs/1706.03762) 的 attention 与架构定义；它的原始 encoder-decoder 设计不等于今天所有 decoder-only LLM。需要图示时看 [CS336 2025 架构讲义固定版本](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf)。

### 应画或推导的对象

手算一组三 token、单头 attention：先写 QKᵀ，再加入因果约束，最后得到加权 V。指出 softmax 沿哪个轴进行。画一条参数被两个位置共同使用的反向路径，说明为什么其梯度要相加。列出 embedding 与线性层的参数量，区分参数量和中间 attention 矩阵大小。

### 两个 CPU 练习

**E01-A：搭一个可解释的玩具 decoder。**用固定 byte 或字符词表，自写 attention、MLP、残差、位置表与输出头；可用 PyTorch 基本算子和自动微分，不直接调用 Transformer 模块。两层、宽度 64、四头、长度 32 可作为调试起点，仅是玩具尺寸。尝试拟合自写短句的小子集，保留曲线及失败记录。失败时先缩小样本、检查目标，不靠扩大模型掩盖错误。

**E01-B：给模型找反例。**固定权重、关闭 dropout，修改后半段，比较前半段 logits；改变 batch 中另一条序列，检查第一条是否按预期不变。故意去掉 causal mask 或转错轴，记录捕获错误的断言。把三个函数映射到 OLMo，并注明省略的机制，不声称实现等价。

**GPU 升级：**正确性通过后才扩展语料和模型；同输入比较 CPU 与 GPU 的误差，再记录吞吐和显存。GPU 结果不是本模块验收前提。

### 验收问题与错误理解

为什么训练可以并行计算整段，而采样仍有顺序依赖？为什么 logits 的最后一轴是词表，不能当作概率直接相加？为什么“训练 loss 很低”无法排除偷看未来？为什么改变 batch 排列可能有数值微差，却不该改变数学目标？把 attention 当作检索原文的固定字典、把自动微分当作数值差分，都需返工解释。

**产物：**玩具 decoder、shape 表、因果性检查、失败注入报告、与 OLMo 的接口对照。**建议工作量：**20–35 小时。**下一模块：**[M02](#m02)，先使更新正确，再追更大规模。

<a id="m02"></a>
## M02｜优化、数值与 loss 正确性

### 先修与可跳过诊断

完成 M01，能指出可训练参数与 loss 的依赖。诊断：两个 microbatch 有效长度不同，如何合并平均 loss？标签应该在什么位置左移？连续两次 `backward` 后梯度是什么？能以手算、断言和反例回答，并解释 AdamW 状态与重启关系，可缩短基础阅读。

### 学习目标细分

- 将交叉熵写成有效目标上的负对数概率之和，识别 target shift、ignore index、padding 与文档边界。
- 区分“局部平均再平均”和“求和后除以有效 token 总数”；知道样本权重和梯度尺度会被 reduction 改变。
- 从 SGD 走到 AdamW：理解一阶/二阶状态、偏差修正、epsilon、解耦 weight decay、学习率调度和裁剪位置。
- 分清浮点动态范围、精度和累加顺序；能用 log-sum-exp 稳定计算解释溢出，而不是把所有 NaN 归因于学习率。
- 理解权重、梯度、优化器状态可以用不同 dtype；混合精度不是把整个模型无差别转换成一种类型。

### 核心阅读：按顺序，最多三项

1. [首课的 loss 与更新推导](../lessons/01-one-token-to-update.md)，再回到 A1 讲义的优化器与训练部分。先用小数字理解运算，再看调度器 API。
2. [OLMo train_batch](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/train/train_module/transformer/train_module.py#L345-L441)，向上追 `get_labels`。尤其读 instance filtering 的注释：该路径将某些被忽略实例的 token 加回分母，会使报告 loss 人为偏低，不能拿自己的理想公式覆盖实际配方。
3. [PyTorch 数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)：重点是非结合性、极值与后端差异；文档版本会变化，记录阅读日期，不照抄硬件开关作为通用建议。

**选读：**[OLMo RMSNorm](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/nn/layer_norm.py#L210-L236)，解释为什么内部计算类型与输出类型分开；[PyTorch autograd mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html) 中无效运算即使事后 mask，也可能污染反向传播的例子。

### 应画或推导的对象

写出 `L=(Σ_j S_j)/(Σ_j n_j)`，其中 `S_j` 是第 j 个片段有效负对数概率之和，`n_j` 是有效目标数；推导每片段平均值的正确加权方式。再画 `梯度清零 → microbatch 累积 → 裁剪 → optimizer → scheduler` 的状态流，明确 AdamW 的动量状态不等于模型权重。

### 两个 CPU 练习

**E02-A：让不等长 batch 暴露归一化错误。**构造两个有效长度不同、平均难度也不同的小 batch；分别计算整体梯度、逐 microbatch 的正确加权梯度、错误的局部平均梯度。关闭 dropout，以 FP64 或 FP32 小张量比较误差并解释差别；再加入 padding、EOS、全部被 mask 的实例，规定空目标 batch 是拒绝还是跳过，禁止出现未解释的零分母。另选少量参数，以中心差分检查 autograd 梯度，说明步长太大和太小各有什么问题。

**E02-B：制作数值与恢复故障集。**用大幅度 logits 对比直接 exponentiation 与稳定交叉熵，记录有限值检查。然后在玩具模型上比较“连续训练”和“保存后恢复”，恢复权重、优化器、计数、随机状态及取数位置；再故意仅恢复权重，观察下一次更新是否改变。额外注入漏清梯度和双重除 batch 两类错误，要求每种错误至少有一个能捕获它的检查，而非只看最终 loss。

**GPU 升级：**在同一输入与参数下比较精度方案；分别记录算子误差、梯度偏差和短训练趋势。CPU 模拟不能证明真实 FP8/FP4 kernel、通信归约或混合精度性能正确。

### 验收问题与错误理解

为什么 FP64 有帮助却不保证修复错误目标？为什么同一 batch 拆分方式改变后，数学更新应一致而浮点值未必逐位一致？weight decay 与 L2 项在自适应优化器中为何不能随意互换？为什么 clipping 应检查累积和缩放后的整体语义？把低 loss 当作健康、把“可以继续运行”当作恢复正确、把更多 padding 造成的统计变化当作模型进步，均不通过。

**产物：**loss/梯度对照表、数值故障样例、恢复状态清单、正确与错误实现的最小反例。**建议工作量：**18–30 小时。**下一模块：**[M03](#m03)，此时才扩大数据管线。

<a id="m03"></a>
## M03｜数据、tokenizer 与评估隔离

### 先修与可跳过诊断

完成 M02 的 labels/mask 检查，能读取 JSONL 和计算内容 hash。诊断：同一网页不同 URL 是否算重复？BPE 词表变了，旧 token shards 是否还能复用？随机划分文档为什么可能把同一代码仓库的副本分到两侧？能提供带反例的数据流与版本记录者，可直接进入练习。

### 学习目标细分

- 区分采集、抽取、语言/领域判断、质量筛选、精确/近似去重、去污染和混合采样的职责。
- 理解 byte 编码、预分词、BPE merge 与特殊 token；比较词表大小、压缩率、语言覆盖与模型成本，知道 token 越少并不必然能力越强。
- 追踪来源、许可/使用条件、文档标识、处理版本和过滤原因；对合成内容增加教师、提示与验证器身份。
- 分清文档数、唯一内容、tokenized tokens、采样 tokens 和有效监督 targets；用实际消费统计审查 mixture。
- 将评估隔离落实到来源簇、题目变体和答案，知道零采样权重只隔离组件，不证明内容没有近重复。

### 核心阅读：按顺序，最多三项

1. [CS336 Assignment 4 官方讲义](https://github.com/stanford-cs336/assignment4-data/blob/main/cs336_assignment4_data.pdf)：阅读抽取、过滤和去重的任务定义；本课程只使用自建微型语料，不要求下载完整 Common Crawl。BPE 机制回看 A1 对应部分。
2. [OLMo tokenizer 与词表约定](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/olmo_core/data/tokenizer.py#L48-L107)，接着读 [数据与 mask 手册](../handbook/01-data-model-pretraining-design.md) 的 Gate 2、3、5。确认真实词元数与为吞吐补齐的 embedding 行数不同。
3. [Marin ArtifactStep 身份实现](https://github.com/marin-community/marin/blob/5e2436d0f61462983003bd8b6eaef8235ecab78c/lib/marin/src/marin/execution/lazy.py#L213-L237)，结合 [Marin 笔记](../notes/repositories/marin.md) 画依赖图。注意配置 fingerprint 不自动替代全部源文件与数据内容校验。

**选读：**CS336 归档页的 Lecture 12–14 评估与数据讲义；[SmolLM 笔记](../notes/repositories/smollm.md) 的 DataTrove tokenization 与多来源混合入口。源码中的内网路径只是作者环境信息，不是公开可下载的数据承诺。

### 应画或推导的对象

画出一个文档经过每道过滤的去向及拒绝理由；再画训练、开发、保留三侧与去重簇的关系。手推几轮 BPE 合并，写出 ties 的确定性规则。计算一个小数据源在目标 mixture 下的预期消费 tokens 与重复轮数，并指出原始 byte 长度为何不能直接当作训练 tokens。

### 两个 CPU 练习

**E03-A：建设可审计的微型语料。**编写 60–100 条中英文、代码和数学样本，加入重复、改写、模板噪声、同源版本及评估答案。做规范化、精确去重及字符/词片段集合的近似匹配，按来源簇划分训练与评估。标注已知重复对，统计误报、漏报和保留率；保存来源、变换链与拒绝理由。小样本阈值不代表适用于互联网语料。

**E03-B：追踪分词到监督目标。**在微型语料上实现小词表 BPE，保留 byte 基线作对照；检查中英文、缩进、数字、特殊标记和未见字符的往返行为。记录每类 bytes/token 与长度分布，按不同 token 比例采样成小 batch，再明确文档注意力边界、位置编号和 label mask。最后把每一个有效 target 追回原文位置或声明的特殊 token；对保留集只做最终约定的检查，不拿它反复调 merge 和过滤阈值。

**GPU 升级：**固定 tokenizer、模型、预算和评估协议，比较两份数据处理方案的训练结果；若同时改词表和混合，必须承认无法归因。GPU 训练前仍要通过全部数据追踪检查。

### 验收问题与错误理解

为什么高质量过滤可能损害少数语言？为什么重复十轮不等于十倍新信息？为什么跨 tokenizer 的 per-token loss 不能直接排行？为什么去重与去污染不是一个指标？为什么 packing 的边界不是纯存储细节？“所有规则越严格越好”“零匹配等于无污染”“tokenizer 只是预处理，可以随时更换”都应能被自己的反例驳回。

**产物：**微型 corpus manifest、去重与隔离报告、tokenizer 文件和版本、采样统计、原文到 target 的追踪图。**建议工作量：**18–32 小时。**下一模块：**[M04](#m04)。

<a id="m04"></a>
## M04｜Scaling、实验设计与完整 pretraining recipe

### 先修与可跳过诊断

完成 M00–M03，拥有能解释的玩具模型、有效 loss 与数据记录。诊断：能否区分等 tokens、等理论 FLOPs、等墙钟时间三种比较？小模型上的最佳 mixture 为什么可能无法直接外推？两份同名 YAML 为何不一定产生同一训练？已有完整实验报告者，可用报告回答这些问题后直接补大规模配方审读。

### 学习目标细分

- 把能力、部署延迟、训练预算与数据可用性变成候选方案，不能先选“最大参数”再寻找理由。
- 理解 scaling law 是依赖配方与分布的经验模型；区分拟合、插值、外推和未参与拟合的检验。
- 设计单因素消融和必要的交互实验，记录重复种子、误差、选择次数与停止规则，接受“证据不足”结论。
- 区分 dense 总参数与 MoE 总/激活参数、计算量与存储/通信成本；把 `6ND` 当有限近似而非预算真值。
- 从一份正式配置追到模型、tokenizer、数据、优化器、schedule、batch、checkpoint 和 eval，理解 pretrain、midtraining、long-context 之间的接口。

### 核心阅读：按顺序，最多三项

1. [CS336 2025 Scaling 基础讲义固定版本](https://github.com/stanford-cs336/spring2025-lectures/blob/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf)：先理解曲线和实验设计，再阅读公式；此处只将其作为概念导读。
2. [Delphi 作者的 scaling 实验记录](https://openathena.ai/blog/delphi/)：重点读第一次失败、修订与留出规模验证；接着对照 [Marin 535B 案例](../handbook/04-marin-535b-live-case-study.md)，分开历史预测、当前代码和进行中的计划。
3. [OLMo 3 7B 预训练配置](https://github.com/allenai/OLMo-core/blob/92870a33c3fee060d57c3faec52b0b369ece85a6/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py#L40-L108)，结合 [OLMo 笔记](../notes/repositories/olmo-core.md) 追到后续阶段。读实际参数组、dtype 与 duration，不仅抄学习率；当前文件的计划延长注释也不应替代最终运行报告。

**选读：**[Chinchilla 原论文](https://arxiv.org/abs/2203.15556) 的固定计算预算问题；[SmolLM 4K 配方](https://github.com/huggingface/smollm/blob/a041759883ec7152d18fb985ea49be641a0bceef/text/pretraining/smollm3/stage1_8T.yaml#L163-L255) 与已有 SmolLM 笔记中的长上下文配置，比较阶段如何联合改数据和系统。CS336 [Assignment 3 仓库](https://github.com/stanford-cs336/assignment3-scaling) 当前主分支已写为 Spring 2026，不能把它冒充固定 2025 版本；它的课程 API 有访问条件，本模块不依赖该服务。

### 应画或推导的对象

画三个预算下“模型大小—训练 tokens—验证 loss”的等算力比较示意，并用不同标记区分拟合点与留出点。画一张候选模型资源表，列参数、激活、优化器、attention 中间量和预估通信。最后画正式 recipe 的依赖图，标明哪个产物失效会迫使重新 tokenization、重新拟合或重新编译。

### 两个 CPU 练习

**E04-A：设计并审判一个小实验。**先写问题、唯一主要变量、固定条件、评价指标、预算和会推翻假设的结果。可在玩具模型上用极小预算比较两档宽度与三档 token horizon，至少重复两次；CPU 太慢时缩小任务，不把计划写成完成。另用明确标注的合成表练习拟合与外推：隐藏最大预算点，比较不同拟合选择的预测，展示拟合好却预测失败的情况。真实小跑与合成拟合必须分图、分结论；二者均不支持宣称得到可迁移的 frontier scaling law。

**E04-B：做一次公开配方评审。**以 OLMo 3 7B 填写模型、词表、mix、tokens/update、参数组、精度、停止、恢复与评估表；给出源码行号，区分默认值与覆盖。用 SmolLM 长上下文或 Marin hero 对照，列三项可直接比较、三项不可直接比较的量。写一页“下一笔预算该验证什么”，不把复制的大跑参数当作已验证方案。

**GPU 升级：**执行预先登记的少量 pilot，并用接近目标形状的短跑校正吞吐预算；实际集群数值、恢复和性能验收进入 M05 及后续系统模块。GPU 多了也不能跳过留出规模与独立评估。

### 验收问题与错误理解

为什么训练计算最优不一定服务成本最优？为什么小规模稳定不证明长 horizon 稳定？为什么能力分数提高可能来自更长推理或 agent 预算？为什么 changing tokenizer 使旧 scaling 曲线不能无条件接续？能否解释 OLMo 的 midtraining 数据和长上下文阶段为何是研究选择，而不是所有模型必经的统一步骤？把相关当因果、把 curve fitting 当定律、把单个 kernel 加速当整项训练加速，均需用反例返工。

**产物：**预登记实验卡、真实/合成分离的曲线、公开 recipe 评审表、资源预算及下一步决策备忘录。**建议工作量：**20–35 小时。**下一模块：**[M05：分布式并行](02-training-systems.md)。

## 本篇完成后，应能独立交付什么

你应能追踪一条文档怎样成为监督目标，解释因果性、loss 合并、梯度更新与 checkpoint 状态，再映射到公开配方，区分亲手验证、源码阅读与缺失的证据。

检查顺序是产物 → 反例 → 解释 → 决策。无法解释更新就回 M02，数据或指标不可信就回 M03；小测试通过后再进 M05。课程正文不会自动增加个人进度，实际产物与独立验证才是完成依据。
