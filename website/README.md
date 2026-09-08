# 交互学习网站

网站入口：[Frontier LLM Handbook](https://yubol-bobo.github.io/frontier_llm_handbook/)。

网站从仓库现有 Markdown、`repos.json` 和 `sources.lock.json` 生成，原始教材仍是唯一内容来源。课程顺序以 [ROADMAP](../ROADMAP.md) 为准，资源问题与入口来自 [RESOURCE_ATLAS](../RESOURCE_ATLAS.md)。网站不要求恢复 `sources/`，本地源码链接会转成固定 SHA 的 upstream 链接。

## 学习体验

- 深色学习空间与浅色阅读主题，三个阶段的课程卡片、继续学习入口与个人验收计数。
- 按三个阶段浏览 M00–M15，直接阅读课程、前沿专题、全流程手册和源码笔记；阅读目录跟随当前章节，顶部显示阅读位置。
- 中文/英文全文搜索，按模块问题定位资料，保留论文、固定源码和原文入口。
- 在知识地图中选择问题，通过连线与高亮查看跨层连接；课程先修使用有方向的边，主题与模块的关联使用无方向的虚线。窄屏改为模块列表，关系详情仍可阅读。
- 用浏览器调整 logits、标签和学习率，观察概率条、更新前标记与本次操作的 loss 轨迹；切换 mask 与分组，直观看到有效目标和加权差异；完成四道自测。修改单 token 参数会重置轨迹，恢复实验默认值保留当前页自测作答。
- 保存模块笔记、工程记录与自主验收证据，导出或导入 JSON 备份。

笔记只保存在当前浏览器的 `localStorage`，不会上传 GitHub，也没有账号或跨设备自动同步。标记为“已自主验收”需要证据说明；自测与页面访问都不会自动结业。导入合并时，同名模块使用备份中的记录，其他模块保留。

主题偏好单独保存，升级沿用原有学习记录格式。交互支持键盘焦点和减少动态效果偏好；切换主题会更新文档关系图，不重建笔记输入区域。

## 开发与更新

需要 Node.js 22 或更高版本。第一次运行：

```powershell
npm ci
npm run check
npm run dev
```

预览地址是 `http://127.0.0.1:4173/frontier_llm_handbook/`。每次修改文档或网站代码后先运行 `npm run build` 再刷新；预览服务器读取 `_site/`，不自动监听重建。

`npm run check` 执行构建与 Node 测试，覆盖课程身份、站内文档和锚点、源码链接转换、HTML 清理、梯度有限差分、已有 Python 实验数值、全部非空 mask 组合的分组不变性、学习记录校验及搜索，并检查课程卡片和继续学习、关系类型、loss 曲线、分组展示与字体产物。它不包含浏览器视觉测试，也不代表运行了真实模型训练。

已有完整 upstream clones 的维护者另运行 `python tools/validate_learning_repo.py`。CI 不下载 19 个上游仓库，只检查网站依赖的资料及生成结果。

## GitHub Pages 发布

[发布工作流](../.github/workflows/pages.yml) 在 `main` 更新时构建、检查并发布 `_site/`，PR 只构建检查。首次需在仓库 Settings → Pages 选择 **GitHub Actions**；管理员也可通过 GitHub Pages API 配置 `build_type=workflow`。

所有站内入口使用 hash 路由，资源路径相对于站点目录，因此支持 GitHub 项目站点的 `/frontier_llm_handbook/` 前缀与直接分享阅读链接。生成文件不提交 Git；包版本和完整依赖由 `package-lock.json` 固定，正文、Markdown 渲染与 Mermaid 图依赖随站点一起打包，不依赖第三方 CDN。

界面使用 Geist 与 Noto Sans SC 可变字体，图标来自 Lucide。字体经 Fontsource 的 Unicode 子集 CSS 在本站按需加载；构建会将字体和图标的原始许可证复制到 `assets/licenses/`。两套字体为 SIL Open Font License 1.1；Lucide 的完整许可和归属说明随站点保留。

## 文件职责

- `content.mjs`：从原文提取模块、专题、资源、先修；渲染及清理 Markdown，重写源码和站内链接。
- `app.mjs`：路由、阅读、知识地图、互动教学及个人记录。
- `design.mjs` / `interactions.mjs` / `lab-view.mjs`：课程卡片、可信图标、主题、阅读跟踪、地图连线与实验结果展示。
- `styles.css`：字体与样式入口；`base.css` 保留基础文档和控件布局，`modern.css` 定义主题与现代界面。
- `learning.mjs`：独立可测试的数学、进度格式与搜索逻辑。
- `build.mjs` / `serve.mjs`：静态输出和带项目路径的本地预览。
- `tests/`：内容与核心交互逻辑的自动检查。

新增知识连接必须指向足以支撑该表述的原文，不根据界面效果虚构软件集成。新增实验应明确浏览器模拟、CPU 检查、真实 GPU 与集群验证的边界。
