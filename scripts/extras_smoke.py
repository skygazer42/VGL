#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_EXTRAS = ("networkx", "scipy", "tensorboard")
EXTRA_IMPORTS = {
    "networkx": "networkx",
    "scipy": "scipy",
    "tensorboard": "tensorboard",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test lightweight optional extras.")
    parser.add_argument(
        "--extras",
        nargs="+",
        default=list(DEFAULT_EXTRAS),
        help="Extras to install and import-check.",
    )
    parser.add_argument(
        "--list-defaults",
        action="store_true",
        help="Print the default lightweight extras and exit.",
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root for editable installs.",
    )
    return parser.parse_args()


def _bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if sys.platform == "win32" else "bin")


def _python_name() -> str:
    return "python.exe" if sys.platform == "win32" else "python"


def _smoke_extra(repo_root: Path, extra: str) -> None:
    import_name = EXTRA_IMPORTS[extra]
    with tempfile.TemporaryDirectory(prefix=f"vgl-extra-{extra}-") as tmp:
        tmp_dir = Path(tmp)
        venv_dir = tmp_dir / "venv"
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)], check=True)
        bin_dir = _bin_dir(venv_dir)
        pip = bin_dir / "pip"
        python_bin = bin_dir / _python_name()
        subprocess.run([str(pip), "install", "--no-deps", "-e", f".[{extra}]"], cwd=repo_root, check=True)
        subprocess.run(
            [str(python_bin), "-c", f"import vgl, {import_name}; print(vgl.__version__)"],
            cwd=tmp_dir,
            check=True,
        )
        print(f"{extra} smoke passed")


def main() -> int:
    args = _parse_args()
    if args.list_defaults:
        for extra in DEFAULT_EXTRAS:
            print(extra)
        return 0

    repo_root = args.repo_root.resolve()
    unknown = [extra for extra in args.extras if extra not in EXTRA_IMPORTS]
    if unknown:
        raise SystemExit(f"unsupported extras: {', '.join(unknown)}")

    for extra in args.extras:
        _smoke_extra(repo_root, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
