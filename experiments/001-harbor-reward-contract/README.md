# 实验 001：奖励向量并不自动等于可靠训练信号

日期：2026-09-08。状态：已运行，本地 CPU 局部实验。

## 问题与方法

使用 Harbor Cookbook 的 multi-reward 任务。固定来源为 `e093c9a860b988d9d74901010ddddb9c7f124f92`。读过 reference solution、correctness 与 performance 的全部 Python 断言后，脚本从 `solve.sh` 的 heredoc 提取 Python 函数，直接执行这些断言。**没有执行 shell 脚本、安装任何包、调用模型或启动容器。**

对照组是 upstream reference；实验组仅把忽略大小写的邮箱 key 改成区分大小写。其他代码与测试输入不变。上游代码只从已克隆的快照读取，未复制进学习仓库；其原始许可与署名保留在源码仓库。

## 重跑

在学习仓库根目录执行：

```powershell
python experiments/001-harbor-reward-contract/run.py
```

脚本要求源码 SHA 与首次阅读一致，并用 `git show SHA:path` 从固定 Git 对象读取实际执行的三个文件；工作区未提交修改不会被误记成该版本。脚本拒绝 `python -O` / `PYTHONOPTIMIZE`，防止断言被移除后出现虚假通过。源码更新后应先审阅，再建立新的实验版本。运行结果写入 [results.json](results.json)，包括输入哈希、运行时刻、每个断言的结果与耗时。

## 观测

| 候选 | 正确性测试 | 性能测试 | 奖励向量 |
|---|---|---|---|
| 参考解 | 7/7 通过 | 1/1 通过 | correctness=1, performance=1 |
| 大小写错误解 | 5/7 通过 | 1/1 通过 | correctness=0, performance=1 |

性能断言处理 100,000 条记录，要求输出 50,000 条且耗时低于 2 秒；其输入邮箱均是相同大小写形式。因此这个测试可以验证该 workload 的速度和数量，却不能独立验证邮箱大小写语义。耗时是本机结果，不是跨机器性能结论。

此外，本实验确认原始奖励字典没有 `reward` 字段；直接使用 `.get("reward", 0.0)` 会返回零。脚本计算的 `correctness * (0.5 + 0.5 * performance)` 仅用来演示显式 scalarization，不是上游公式或推荐通用目标。

## 结论的边界

- 已验证：固定快照的 Python correctness/performance 断言对这两个候选的结果。
- 未验证：pytest 插件、容器隔离、Harbor 对 reward.json 的完整加载过程、训练 adapter 的归一化以及 RL 学习效果。
- 没有从本实验推导出具体上游集成 bug。真正连接任务与 trainer 时，应确认消费端实际收到的 reward 字段和聚合策略。

## 与知识树的连接

任务规格 → verifier 覆盖范围 → 奖励向量 → scalarization → RL objective → held-out evaluation。第一轮学习的实际成果是把这些接口分开验证，而不是把一个成功运行的脚本直接称为成功训练。
