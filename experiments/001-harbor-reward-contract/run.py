"""Exercise the inspected upstream Python verifiers, without Docker or APIs.

This does not execute Harbor, the upstream shell installer, or an RL trainer.
It loads the exact reference solution embedded in solve.sh, then compares it
with a deliberate case-sensitivity mutation using the upstream assertion tests.
"""
import json
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "sources" / "harbor-cookbook"
RECIPE = REPO / "harbor_cookbook" / "recipes" / "multi-reward"
EXPECTED_COMMIT = "e093c9a860b988d9d74901010ddddb9c7f124f92"


def read_source(path):
    relative = path.relative_to(REPO).as_posix()
    return subprocess.check_output(["git", "show", f"{EXPECTED_COMMIT}:{relative}"],
                                   cwd=REPO, text=True, encoding="utf-8")


def evaluate(source):
    module = types.ModuleType("deduplicate")
    exec(compile(source, "<inspected-upstream-reference>", "exec"), module.__dict__)
    previous = sys.modules.get("deduplicate")
    sys.modules["deduplicate"] = module
    groups = {}
    try:
        for group in ("correctness", "performance"):
            original_path = list(sys.path)
            try:
                test_path = RECIPE / "tests" / f"test_{group}.py"
                namespace = {"__name__": f"upstream_test_{group}", "__file__": str(test_path)}
                exec(compile(read_source(test_path), str(test_path), "exec"), namespace)
            finally:
                sys.path[:] = original_path
            results = []
            for name, function in sorted(namespace.items()):
                if not name.startswith("test_") or not callable(function):
                    continue
                start = time.perf_counter()
                try:
                    function()
                    result = {"name": name, "passed": True}
                except AssertionError as exc:
                    result = {"name": name, "passed": False, "reason": str(exc) or "AssertionError"}
                result["seconds"] = round(time.perf_counter() - start, 6)
                results.append(result)
            if not results:
                raise RuntimeError(f"No tests found for {group}")
            groups[group] = {"reward": int(all(t["passed"] for t in results)), "tests": results}
    finally:
        if previous is None:
            sys.modules.pop("deduplicate", None)
        else:
            sys.modules["deduplicate"] = previous
    return {"rewards": {g: result["reward"] for g, result in groups.items()}, "groups": groups}


def main():
    if sys.flags.optimize:
        raise SystemExit("Refusing optimized Python: upstream assert statements must remain enabled.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit("Source snapshot changed; review upstream files before rerunning this experiment.")
    shell = read_source(RECIPE / "solution" / "solve.sh")
    marker = "cat > /app/deduplicate.py << 'EOF'\n"
    if marker not in shell or "\nEOF" not in shell:
        raise SystemExit("Reference heredoc changed; manual review required.")
    reference = shell.split(marker, 1)[1].split("\nEOF", 1)[0] + "\n"
    needle = 'record["email"].lower()'
    if reference.count(needle) != 1:
        raise SystemExit("Reference mutation target changed; manual review required.")
    incorrect = reference.replace(needle, 'record["email"]')
    reference_result = evaluate(reference)
    incorrect_result = evaluate(incorrect)
    vector = reference_result["rewards"]
    observations = {
        "reference_passes_both": vector == {"correctness": 1, "performance": 1},
        "incorrect_can_pass_performance": incorrect_result["rewards"] == {"correctness": 0, "performance": 1},
        "raw_vector_has_no_reward_key": "reward" not in vector,
        "unadapted_scalar_get_returns_zero": vector.get("reward", 0.0),
        "illustrative_correctness_gated_scalar": vector["correctness"] * (0.5 + 0.5 * vector["performance"]),
    }
    output = {
        "recorded_at": datetime.now(timezone.utc).isoformat(), "source_commit": commit,
        "input_sha256": {str(path.relative_to(REPO).as_posix()):hashlib.sha256(read_source(path).encode("utf-8")).hexdigest()
                         for path in (RECIPE / "solution" / "solve.sh", RECIPE / "tests" / "test_correctness.py", RECIPE / "tests" / "test_performance.py")},
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "scope": "Actual upstream Python assertions, invoked directly; no pytest/Docker/Harbor/RL execution.",
        "candidates": {"upstream_reference": reference_result, "case_sensitive_mutation": incorrect_result},
        "observations": observations,
    }
    destination = Path(__file__).with_name("results.json")
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results_file": str(destination), "observations": observations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
