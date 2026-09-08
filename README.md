# Frontier LLM Handbook

**本学习资料旨在系统提供前沿 LLM 的端到端训练知识，帮助学习者建立工程能力基础，并逐步提升到能够实现、验证、诊断和改进训练系统。**

面向希望进入这一领域的初学者，以及希望深化训练与系统能力的工程师、研究者，尤其是没有机会进入 frontier lab 的学习者。以公开的一手论文、源码、训练配方和运行记录为依据，将数据、模型设计、预训练、后训练、推理与评估连接成有先修、有练习、有验收的中英双语学习路线。

创建日期：2026-09-08。课程 v0.1：**16 个核心模块、6 个前沿专题、20 个固定源码仓库**，配合外部一级资源、五篇全流程讲解和可运行首课。课程设计、实际源码审读和已执行实验分别标记；完整覆盖地图与待补内容公开维护。

主仓库：[yubol-bobo/frontier_llm_handbook](https://github.com/yubol-bobo/frontier_llm_handbook)，主分支：`main`。后续学习笔记、实验与知识树在这里持续积累；本地工作目录目前保留名称 `frontier-llm-lab`。

**[进入交互学习网站 →](https://yubol-bobo.github.io/frontier_llm_handbook/)** 按模块阅读、全文搜索、探索知识连接，动手调整训练公式，并保存个人笔记与验收记录。网站由本仓库资料生成，随 `main` 更新发布到 GitHub Pages；[网站开发说明](website/README.md)。

[中文网站](https://yubol-bobo.github.io/frontier_llm_handbook/?lang=zh#/learn) · [English website](https://yubol-bobo.github.io/frontier_llm_handbook/?lang=en#/learn)。顶部可切换界面与全部站内教材语言，个人记录共用；[中英术语表](GLOSSARY.md)。

## 两条贯穿全程的学习目标

- **端到端训练知识：**理解目标与预算、数据与 tokenizer、架构与规模实验、预训练与阶段训练、SFT/偏好优化/RL、推理、评估与发布之间的关系，能解释每个阶段的输入、输出、设计依据与取舍。
- **工程能力的基础与提升：**从数学、张量、梯度、Git 和可复现实验起步，逐步掌握数据管线、训练循环、分布式与 MoE、性能分析、故障恢复和 agent 系统；用实现、测量、调试与综合项目检验进步。

每个模块都要同时回答“为什么这样训练”和“怎样把它实现并验证”。学习产物与能力层级见 [学习路线](ROADMAP.md)；各项内容的实际建设深度见 [覆盖地图](COVERAGE.md)。

## 从这里开始

| 入口 | 用途 |
|---|---|
| **[学习路线：从这里开始](ROADMAP.md)** | 入门诊断、M00–M15 顺序、先修图、不同方向和硬件条件的路径 |
| [资源地图](RESOURCE_ATLAS.md) | 每份资源为什么读、在哪个模块读、读到哪里停、公开程度与快照状态 |
| [知识覆盖与缺口](COVERAGE.md) | 区分课程设计、教材讲解、源码审读、实测与尚未公开的材料 |
| [首课：一个 token 到一次更新](lessons/01-one-token-to-update.md) | 用普通电脑理解概率、loss、梯度、mask 与分片归一化 |
| [超大 LLM 从零训练全流程](handbook/00-end-to-end.md) | 从目标、数据与 scaling pilots，到分布式预训练、agent RL 和发布；含四篇详章 |
| [仓库学习索引](LEARNING_LIST.md) | 20 个固定项目的登记编号与源码笔记；编号不代表先修顺序 |
| [知识树](KNOWLEDGE_TREE.md) | 按知识组织项目：从任务与数据到训练、调度、GPU 内核 |
| [仓库关系图](REPO_RELATIONSHIPS.md) | 区分真实依赖、可选后端、示例集成、项目谱系和概念对应 |
| [内容建设与维护者进度](PROGRESS.md) | 本仓库实际完成了什么；读者另用学习记录模板 |
| [源码快照](SOURCE_INDEX.md) | 本地目录、固定 commit、克隆完整性与版本边界 |
| [学习方法](HOW_TO_STUDY.md) | 每次如何读代码、做实验、记笔记和更新知识树 |
| [术语表](GLOSSARY.md) | 连接训练与系统设计的共用概念 |
| [贡献指南](CONTRIBUTING.md) | 怎样增加有教学价值的资源、lesson 和实验 |

## 学习顺序

**基础与梯度 → 数据与 scaling → 分布式与 MoE → 内核/运行/恢复 → 推理 → SFT/偏好 → 同步 RL → Harness/任务/轨迹 → 异步 agent RL → 独立评估与综合项目。** 评估原则从数据阶段开始使用，最后再做整体验收。

| 课程 | 模块 | 核心交付 |
|---|---|---|
| [基础到预训练](curriculum/01-foundations-to-pretraining.md) | M00–M04 | 数学桥接、decoder、正确更新、数据与实验设计 |
| [训练与推理系统](curriculum/02-training-systems.md) | M05–M09 | 并行布局、MoE、数值/性能、恢复与 serving |
| [后训练与 agent](curriculum/03-posttraining-and-agents.md) | M10–M15 | 目标函数、同步到异步、环境/轨迹、评估与 capstone |
| [前沿专题](curriculum/04-frontier-seminars.md) | F01–F06 | 新架构、蒸馏、多模态、数据研究、深层 infra 与持续学习 |

每个核心模块有先修与跳过诊断、顺序阅读、两项练习、GPU 升级条件、验收和产物。普通电脑可从数学、代码和小实验入门；真实 GPU 性能与规模复现需要另行验证。完整时间安排和专项路线见 [ROADMAP](ROADMAP.md)。

## 当前成果

- 20/20 仓库已克隆，全部记录来源和固定 SHA。
- 20 份项目初读笔记：每份至少追踪一条具体代码/配置路径；它们不是全仓库审计或“已经学完”。
- Pi 的低层 agent loop、Harbor Cookbook 的局部任务/评分接口已经进一步展开。
- 已执行 [实验 001：Harbor 多维奖励](experiments/001-harbor-reward-contract/README.md) 与 [实验 002：token loss 与梯度归一化](experiments/002-token-weighted-loss/README.md)，均为局部 CPU 实验；没有运行完整 agent RL 或大模型训练。
- 新增五篇 [训练全流程手册](handbook/00-end-to-end.md)，结合固定源码与当前一手报告，另以 [Marin 535B 进行中的训练](handbook/04-marin-535b-live-case-study.md) 追踪真实决策。源码、作者报告、工程综合与未验证事项分别标注。

## 目录

16 个核心模块另提供 32 项练习设计；它们尚未由本仓库完整执行，不因提供课程就标为结业。

```text
frontier-llm-lab/
├── ROADMAP.md                   学习顺序、先修、分流与验收
├── RESOURCE_ATLAS.md            核心与扩展资源、用途与公开程度
├── COVERAGE.md                  知识覆盖深度与待补内容
├── curriculum/                 16 核心模块与 6 个前沿专题
├── lessons/                    带 worked example 的教学单元
├── website/                    交互学习网站、内容生成与自动检查
├── .github/workflows/          GitHub Pages 构建与发布
├── LEARNING_LIST.md              学习清单
├── KNOWLEDGE_TREE.md             概念知识树
├── REPO_RELATIONSHIPS.md         有证据的项目关系
├── PROGRESS.md                   阶段状态和续学入口
├── HOW_TO_STUDY.md               持续学习流程
├── GLOSSARY.md                   术语与接口语义
├── repos.json                   20 个源码仓库注册表
├── sources.lock.json            本次学习的源码快照记录
├── SOURCE_INDEX.md              可点击的源码索引
├── handbook/                    全流程、三篇机制详章与 Marin 535B 案例
├── sources/                     独立 upstream clones，外层 Git 忽略
├── notes/repositories/          逐仓库源码笔记
├── notes/connections/           跨仓库专题与证据
├── notes/sessions/              每次学习记录
├── experiments/                 自有实验、README 和小型结果
├── templates/                   可复用笔记模板
└── tools/                       克隆、快照与完整性检查工具
```

源码使用 depth=1 的浅克隆，当前工作树可读，未获取完整 Git 历史、submodule 内容、LFS 大文件、模型权重或训练数据。GitHub 主仓库保存学习资料、实验和源码版本清单；`sources/` 下的 20 个独立 upstream checkouts 不重复打包上传，可通过下面的命令恢复。笔记中的本地源码链接需要先恢复 `sources/`；固定 SHA 的 GitHub 链接可以直接在线阅读。

## 在另一台电脑恢复

```powershell
git clone https://github.com/yubol-bobo/frontier_llm_handbook.git
cd frontier_llm_handbook
python tools/clone_repos.py --restore-lock
python tools/validate_learning_repo.py
```

这里的恢复覆盖登记的源码快照，不自动安装训练环境、下载权重/数据或执行 upstream 安装脚本。

## 常用命令

在本目录执行，工具仅使用 Python 标准库与 Git：

```powershell
python tools/clone_repos.py --restore-lock
python tools/validate_learning_repo.py
python experiments/001-harbor-reward-contract/run.py
```

`clone_repos.py` 默认补齐缺失仓库，检查已有来源，不更新或重置已有源码。若在新电脑恢复已记录的版本，使用 `python tools/clone_repos.py --restore-lock`。更新源码后，旧笔记继续引用原始 SHA；为新的学习创建记录，再按需要运行 `snapshot_sources.py`。

## 下一次学习

新读者从 [ROADMAP 的诊断](ROADMAP.md) 与 [M00](curriculum/01-foundations-to-pretraining.md#m00) 开始，复制 [个人学习记录模板](templates/learner-progress.md)。已有基础按诊断跳过；不要把维护者的阅读进度当成自己的结业记录。内容建设下一步优先补 M01 完整 decoder lesson，详见 [维护进度](PROGRESS.md#下一次从这里继续)。

## Claude Code harness 学习入口

从 [M12 进阶讲解](handbook/05-claude-code-harness.md) 开始：Pi → Claude Code 的执行循环、上下文、权限与恢复 → Harbor/APEX 的轨迹与评估。配套 [固定源码与历史快照笔记](notes/repositories/claude-agent-sdk.md) 和 [CPU 状态机实验](experiments/harness-state-machine/README.md)。官方 SDK 为新增的第 20 个独立源码项目；历史镜像有单独的来源核验边界。实验使用自有 mock provider，不调用 Claude，也不代表复现其内部实现。
