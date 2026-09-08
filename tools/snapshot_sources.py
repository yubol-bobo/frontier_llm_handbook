"""Record current source commits and produce a browseable source inventory."""
from datetime import datetime, timezone
import json
from pathlib import Path
from clone_repos import inspect

ROOT = Path(__file__).resolve().parents[1]
repos = json.loads((ROOT / "repos.json").read_text(encoding="utf-8"))["repositories"]
snapshots = [inspect(repo) for repo in repos]
payload = {
    "schema_version": 1,
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "clone_policy": {"depth": 1, "tags": False, "submodules_initialized": False, "git_lfs_smudge": False},
    "repositories": snapshots,
}
(ROOT / "sources.lock.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
rows = ["# Source snapshots", "", "记录每份初读笔记对应的源码版本。不是跨项目兼容性 lockfile；实际训练仍应使用各项目自己的依赖锁定。", "", "所有源码为 shallow clone，保留当前工作树；不含完整历史、未初始化的 submodules 或 Git LFS 大文件内容。模型权重与数据集不在本目录。", "", "| 项目 | 本地源码 | 固定版本 | 文件数 | Submodules | LFS pointers |", "|---|---|---|---:|---:|---:|"]
for repo, snap in zip(repos, snapshots):
    url = repo["url"].removesuffix(".git")
    rows.append(f'| {repo["name"]} | [{snap["path"]}]({snap["path"]}/) | [{snap["commit"][:12]}]({url}/tree/{snap["commit"]}) | {snap["tracked_files"]} | {snap["submodules"]} | {len(snap["lfs_pointer_files"])} |')
rows += ["", "## 使用说明", "", "- 本地源码互相独立，学习仓库通过 `.gitignore` 排除 `sources/`，避免嵌套仓库误提交。", "- 学习笔记引用固定 SHA。之后手工更新源码时，应新建学习记录并重新记录快照，不覆盖过去实验依据。", "- `python tools/clone_repos.py` 只克隆缺失仓库并检查已有 origin，不 pull/reset 已有源码。", "- `python tools/snapshot_sources.py` 重新记录当前版本；只有在需要更新来源清单时才执行。", "- Submodule 仅记录 gitlink；需要编译某个项目时再按其说明获取对应依赖。", ""]
(ROOT / "SOURCE_INDEX.md").write_text("\n".join(rows), encoding="utf-8")
print(json.dumps({"repositories":len(snapshots),"tracked_files":sum(s["tracked_files"] for s in snapshots),"dirty":[s["id"] for s in snapshots if s["working_tree_changes"]]}, ensure_ascii=False))
