# Frontier LLM Handbook

从源码、实验配方与可重复的小实验，学习 LLM training、agent RL、harness 和系统基础设施。

创建日期：2026-09-08。范围：此前清单的项目、最初四个项目，以及 NVIDIA Megatron-LM，共 **19 个独立源码仓库**。

主仓库：[yubol-bobo/frontier_llm_handbook](https://github.com/yubol-bobo/frontier_llm_handbook)，主分支：`main`。后续学习笔记、实验与知识树在这里持续积累；本地工作目录目前保留名称 `frontier-llm-lab`。

## 从这里开始

| 入口 | 用途 |
|---|---|
| [超大 LLM 从零训练全流程](handbook/00-end-to-end.md) | 从目标、数据与 scaling pilots，到分布式预训练、agent RL 和发布；含四篇详章 |
| [学习清单](LEARNING_LIST.md) | 19 个项目的学习顺序、核心问题、源码入口和阶段产物 |
| [知识树](KNOWLEDGE_TREE.md) | 按知识组织项目：从任务与数据到训练、调度、GPU 内核 |
| [仓库关系图](REPO_RELATIONSHIPS.md) | 区分真实依赖、可选后端、示例集成、项目谱系和概念对应 |
| [学习进度](PROGRESS.md) | 实际完成了什么，接下来从哪里继续 |
| [源码快照](SOURCE_INDEX.md) | 本地目录、固定 commit、克隆完整性与版本边界 |
| [学习方法](HOW_TO_STUDY.md) | 每次如何读代码、做实验、记笔记和更新知识树 |
| [术语表](GLOSSARY.md) | 连接训练与系统设计的共用概念 |

## 当前成果

- 19/19 仓库已克隆，全部记录来源和固定 SHA。
- 19 份项目初读笔记：每份至少追踪一条具体代码/配置路径；它们不是全仓库审计或“已经学完”。
- Pi 的低层 agent loop、Harbor Cookbook 的局部任务/评分接口已经进一步展开。
- 已执行 [实验 001：Harbor 多维奖励](experiments/001-harbor-reward-contract/README.md)，结果与脚本均保存；没有运行完整 agent RL 或大模型训练。
- 新增五篇 [训练全流程手册](handbook/00-end-to-end.md)，结合固定源码与当前一手报告，另以 [Marin 535B 进行中的训练](handbook/04-marin-535b-live-case-study.md) 追踪真实决策。源码、作者报告、工程综合与未验证事项分别标注。

## 目录

```text
frontier-llm-lab/
├── LEARNING_LIST.md              学习清单
├── KNOWLEDGE_TREE.md             概念知识树
├── REPO_RELATIONSHIPS.md         有证据的项目关系
├── PROGRESS.md                   阶段状态和续学入口
├── HOW_TO_STUDY.md               持续学习流程
├── GLOSSARY.md                   术语与接口语义
├── repos.json                   19 个源码仓库注册表
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

源码使用 depth=1 的浅克隆，当前工作树可读，未获取完整 Git 历史、submodule 内容、LFS 大文件、模型权重或训练数据。GitHub 主仓库保存学习资料、实验和源码版本清单；`sources/` 下的 19 个独立 upstream checkouts 不重复打包上传，可通过下面的命令恢复。笔记中的本地源码链接需要先恢复 `sources/`；固定 SHA 的 GitHub 链接可以直接在线阅读。

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

当前优先沿 [全流程手册](handbook/00-end-to-end.md) 追踪一条预训练样本从数据到 loss / checkpoint 的完整路径，见 [进度文件中的下一步](PROGRESS.md#下一次从这里继续)。Pi session/runtime 与 Harbor 奖励加载保留为 agent RL 方向的续学入口。
