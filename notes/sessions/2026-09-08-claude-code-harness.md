# Claude Code harness 学习单元

日期：2026-09-08。

## 本次产物

- 新增双语 Claude Code harness 讲解，接入 M12、资源页与知识树。
- 官方 Python Agent SDK 固定于 f1315c69a74db1c15fed2e5974918495d90b7d57，本地浅克隆 144 个 tracked files；登记为第 20 个源码项目。
- 历史镜像按初始导入 f5a40b86dede580f6543bf8926c9af017eea9409 阅读三条局部路径；未认证其与 Anthropic 原始产物逐字节一致。
- 历史镜像未导入主仓库，也未安装或执行其代码。官方 SDK 的版本不被当作历史 CLI 的配套版本。
- 原创 CPU 状态机执行 8 项 unittest 检查，并生成拒绝、预算、未知结果与压缩遗漏的事件记录。模拟工具只改变内存，record.py 写入教学结果 JSON。
- 中文和英文同步维护，原有学习进度与笔记存储结构未改变。

## 验证范围

官方 SDK 只做静态阅读；没有启动 Claude、调用 API、运行上游测试、进行 GPU 训练或真实 SDK 故障恢复。CPU 成功仅限于自有状态机的契约。未进行浏览器交互或视觉测试。

## 校验结果

- `python experiments/harness-state-machine/run.py`：8 项通过。
- `npm run check`：54 篇文档、20 个源码项目，24 项网站检查通过。
- `python tools/validate_learning_repo.py`：20 份源码笔记、974 个本地链接、525 个固定源码引用，零错误。
- `git diff --check`：无空白错误。
