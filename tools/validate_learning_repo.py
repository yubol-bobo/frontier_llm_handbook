"""Read-only validation of the learning inventory, snapshots, notes and links."""
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace").strip()


def main():
    registry = json.loads((ROOT / "repos.json").read_text(encoding="utf-8"))["repositories"]
    snapshots = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))["repositories"]
    lock = {r["id"]:r for r in snapshots}
    known_urls = {urlsplit(r["url"].removesuffix(".git")).path.strip("/").lower():r for r in registry}
    errors = []
    if len({r["id"] for r in registry}) != len(registry):
        errors.append("Duplicate repository IDs")
    if set(lock) != {r["id"] for r in registry}:
        errors.append("Registry and snapshot IDs differ")
    for repo in registry:
        path = ROOT / "sources" / repo["id"]
        if not (path / ".git").is_dir():
            errors.append(f"Missing clone: {repo['id']}")
            continue
        actual = git("rev-parse", "HEAD", cwd=path)
        if actual != lock.get(repo["id"], {}).get("commit"):
            errors.append(f"Source HEAD differs from lock: {repo['id']}")
        if git("status", "--porcelain", "--untracked-files=no", cwd=path):
            errors.append(f"Source tracked files changed: {repo['id']}")
        origin = git("remote", "get-url", "origin", cwd=path)
        if origin.removesuffix(".git").lower() != repo["url"].removesuffix(".git").lower():
            errors.append(f"Origin mismatch: {repo['id']}")
        note = ROOT / "notes" / "repositories" / (repo["id"] + ".md")
        if not note.is_file() or actual not in note.read_text(encoding="utf-8"):
            errors.append(f"Missing note or note SHA: {repo['id']}")

    # Walk only the learning corpus, never recurse into the upstream source trees.
    documents = sorted(ROOT.glob("*.md"))
    for folder in ("notes", "templates", "experiments"):
        documents.extend(sorted((ROOT / folder).rglob("*.md")))
    local_links = 0
    permalinks = 0
    for document in documents:
        content = document.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]\n]*\]\(([^)\n]+)\)", content):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("http://", "https://")):
                parsed = urlsplit(target)
                pieces = unquote(parsed.path).strip("/").split("/")
                if parsed.hostname != "github.com" or len(pieces) < 4 or pieces[2] not in ("blob", "tree"):
                    continue
                repo_key = "/".join(pieces[:2]).lower()
                if repo_key not in known_urls or not re.fullmatch(r"[0-9a-f]{40}", pieces[3]):
                    continue
                repo = known_urls[repo_key]
                if pieces[3] != lock[repo["id"]]["commit"]:
                    errors.append(f"Unexpected permalink SHA in {document.relative_to(ROOT)}: {target}")
                    continue
                path = ROOT / "sources" / repo["id"] / "/".join(pieces[4:])
                permalinks += 1
                if not path.exists():
                    errors.append(f"Missing source permalink target: {target}")
                    continue
                lines = re.fullmatch(r"L(\d+)(?:-L(\d+))?", parsed.fragment)
                if lines and path.is_file():
                    count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
                    start = int(lines[1])
                    end = int(lines[2] or lines[1])
                    if not 1 <= start <= end <= count:
                        errors.append(f"Out-of-range source lines ({count} lines): {target}")
            elif target and not target.startswith(("#", "mailto:", "app:")):
                path_part = unquote(target.split("#", 1)[0])
                local_links += 1
                if not (document.parent / path_part).exists():
                    errors.append(f"Broken local link in {document.relative_to(ROOT)}: {target}")
    result = {"repositories":len(registry),"repository_notes":len(list((ROOT / "notes" / "repositories").glob("*.md"))),"markdown_files":len(documents),"local_links_checked":local_links,"source_permalinks_checked":permalinks,"errors":errors}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
