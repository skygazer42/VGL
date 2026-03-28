import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dependency_audit_prints_runtime_requirements_from_pyproject():
    script = REPO_ROOT / "scripts" / "dependency_audit.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--print-requirements"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert "torch>=2.4" in lines
    assert "typing_extensions>=4.12" in lines


def test_extras_smoke_lists_default_lightweight_extras():
    script = REPO_ROOT / "scripts" / "extras_smoke.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--list-defaults"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert lines == ["networkx", "scipy", "tensorboard"]


def test_ci_workflow_runs_lint_extras_smoke_and_dependency_audit():
    ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python -m ruff check ." in ci_text
    assert "python scripts/extras_smoke.py --extras networkx scipy tensorboard" in ci_text
    assert "python scripts/dependency_audit.py" in ci_text
