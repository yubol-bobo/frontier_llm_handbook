"""Clone the registered source snapshots without executing upstream code.

Usage: python tools/clone_repos.py [--group harness|training|rl|infra] [--workers 2]
Existing checkouts are inspected, never reset or pulled. LFS content and submodules
are intentionally not downloaded; Git history is shallow. See sources.lock.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                   encoding="utf-8", errors="replace",
                                   env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")).strip()


def inspect(repo):
    path = ROOT / "sources" / repo["id"]
    return {
        "id": repo["id"], "requested_url": repo["url"],
        "origin": git("remote", "get-url", "origin", cwd=path),
        "path": path.relative_to(ROOT).as_posix(),
        "commit": git("rev-parse", "HEAD", cwd=path),
        "branch": git("branch", "--show-current", cwd=path),
        "commit_date": git("show", "-s", "--format=%cI", "HEAD", cwd=path),
        "shallow": git("rev-parse", "--is-shallow-repository", cwd=path) == "true",
        "tracked_files": len(git("ls-files", cwd=path).splitlines()),
        "submodules": git("ls-files", "--stage", cwd=path).count("160000 "),
        "lfs_pointer_files": git("grep", "-l", "--cached", "version https://git-lfs.github.com/spec/v1", cwd=path).splitlines() if has_lfs(path) else [],
        "working_tree_changes": git("status", "--porcelain", "--untracked-files=no", cwd=path),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "cloned"
    }


def has_lfs(path):
    p = subprocess.run(["git", "grep", "-q", "--cached", "version https://git-lfs.github.com/spec/v1"],
                       cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.returncode == 0


def clone_one(repo, expected_commit=None):
    path = ROOT / "sources" / repo["id"]
    log_dir = ROOT / ".cache" / "clone"
    log_dir.mkdir(parents=True, exist_ok=True)
    result = {"id": repo["id"], "status": "failed"}
    try:
        if path.exists() and not (path / ".git").exists():
            raise RuntimeError("Destination exists without .git; refusing to overwrite")
        fresh_checkout = not path.exists()
        if fresh_checkout:
            env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
            with (log_dir / (repo["id"] + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(["git", "-c", "core.longpaths=true", "clone", "--depth", "1",
                                "--no-tags", repo["url"], str(path)], env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        origin = git("remote", "get-url", "origin", cwd=path)
        if origin.rstrip("/").removesuffix(".git").lower() != repo["url"].rstrip("/").removesuffix(".git").lower():
            raise RuntimeError("Existing checkout origin does not match registry")
        if expected_commit and git("rev-parse", "HEAD", cwd=path) != expected_commit:
            if not fresh_checkout:
                raise RuntimeError("Existing checkout differs from lock; preserved unchanged. Restore into a fresh directory or inspect manually.")
            # Only the checkout created by this invocation may be moved to its recorded SHA.
            git("fetch", "--depth", "1", "origin", expected_commit, cwd=path)
            git("checkout", "--detach", expected_commit, cwd=path)
        # Per-repository setting only; never modify the user's global Git config.
        git("config", "core.longpaths", "true", cwd=path)
        result = inspect(repo)
    except Exception as exc:
        result["error"] = str(exc)
    (log_dir / (repo["id"] + ".json")).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("id", "status", "commit", "error") if k in result}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["harness", "training", "rl", "infra"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--restore-lock", action="store_true", help="Restore missing checkouts to recorded SHA; never change existing HEAD")
    args = parser.parse_args()
    repos = json.loads((ROOT / "repos.json").read_text(encoding="utf-8"))["repositories"]
    selected = [r for r in repos if not args.group or r["group"] == args.group]
    pinned = {}
    if args.restore_lock:
        pinned = {r["id"]:r["commit"] for r in json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))["repositories"]}
        missing = [r["id"] for r in selected if r["id"] not in pinned]
        if missing:
            raise SystemExit(f"Lock has no recorded commit for: {missing}")
    (ROOT / "sources").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = [f.result() for f in as_completed([pool.submit(clone_one, r, pinned.get(r["id"])) for r in selected])]
    raise SystemExit(0 if all(r["status"] == "cloned" for r in results) else 1)


if __name__ == "__main__":
    main()
