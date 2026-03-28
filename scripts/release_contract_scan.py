#!/usr/bin/env python3

from __future__ import annotations

import argparse
import email
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


CheckFn = Callable[[], tuple[bool, str]]


@dataclass(frozen=True)
class ScanTask:
    id: str
    category: str
    description: str
    check: CheckFn


class ArtifactContext:
    def __init__(self, repo_root: Path, artifact_dir: Path):
        self.repo_root = repo_root.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self._pyproject: dict | None = None
        self._version: str | None = None
        self._wheel_path: tuple[Path | None, str] | None = None
        self._sdist_path: tuple[Path | None, str] | None = None
        self._wheel_metadata: tuple[email.message.Message | None, str] | None = None
        self._wheel_names: tuple[list[str] | None, str] | None = None
        self._sdist_names: tuple[list[str] | None, str] | None = None

    def pyproject_value(self, *keys: str) -> object:
        payload: object = self._load_pyproject()
        for key in keys:
            if not isinstance(payload, dict):
                raise KeyError(" -> ".join(keys))
            payload = payload[key]
        return payload

    def repo_version(self) -> str:
        if self._version is None:
            text = (self.repo_root / "vgl" / "version.py").read_text(encoding="utf-8")
            match = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
            if match is None:
                raise RuntimeError("unable to parse __version__ from vgl/version.py")
            self._version = match.group(1)
        return self._version

    def wheel_path(self) -> tuple[Path | None, str]:
        if self._wheel_path is None:
            self._wheel_path = self._single_artifact("*.whl")
        return self._wheel_path

    def sdist_path(self) -> tuple[Path | None, str]:
        if self._sdist_path is None:
            self._sdist_path = self._single_artifact("*.tar.gz")
        return self._sdist_path

    def wheel_metadata(self) -> tuple[email.message.Message | None, str]:
        if self._wheel_metadata is None:
            wheel_path, detail = self.wheel_path()
            if wheel_path is None:
                self._wheel_metadata = (None, detail)
            else:
                with zipfile.ZipFile(wheel_path) as archive:
                    metadata_name = next((name for name in archive.namelist() if name.endswith("METADATA")), None)
                    if metadata_name is None:
                        self._wheel_metadata = (None, "wheel METADATA missing")
                    else:
                        payload = email.message_from_bytes(archive.read(metadata_name))
                        self._wheel_metadata = (payload, metadata_name)
        return self._wheel_metadata

    def wheel_names(self) -> tuple[list[str] | None, str]:
        if self._wheel_names is None:
            wheel_path, detail = self.wheel_path()
            if wheel_path is None:
                self._wheel_names = (None, detail)
            else:
                with zipfile.ZipFile(wheel_path) as archive:
                    names = archive.namelist()
                self._wheel_names = (names, f"{len(names)} wheel entries")
        return self._wheel_names

    def sdist_names(self) -> tuple[list[str] | None, str]:
        if self._sdist_names is None:
            sdist_path, detail = self.sdist_path()
            if sdist_path is None:
                self._sdist_names = (None, detail)
            else:
                with tarfile.open(sdist_path) as archive:
                    names = archive.getnames()
                self._sdist_names = (names, f"{len(names)} sdist entries")
        return self._sdist_names

    def _load_pyproject(self) -> dict:
        if self._pyproject is None:
            with (self.repo_root / "pyproject.toml").open("rb") as handle:
                self._pyproject = tomllib.load(handle)
        return self._pyproject

    def _single_artifact(self, pattern: str) -> tuple[Path | None, str]:
        matches = sorted(self.artifact_dir.glob(pattern))
        if len(matches) != 1:
            return None, f"{pattern} count == {len(matches)} in {self.artifact_dir}"
        return matches[0], matches[0].name


def _artifact_exists_task(
    ctx: ArtifactContext,
    task_id: str,
    description: str,
    resolver: Callable[[], tuple[Path | None, str]],
) -> ScanTask:
    def check() -> tuple[bool, str]:
        path, detail = resolver()
        return path is not None, detail

    return ScanTask(task_id, "artifact", description, check)


def _metadata_header_equals_task(
    ctx: ArtifactContext,
    task_id: str,
    description: str,
    header: str,
    expected: str,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        metadata, detail = ctx.wheel_metadata()
        if metadata is None:
            return False, detail
        value = metadata.get(header)
        return value == expected, f"{header} == {expected!r}"

    return ScanTask(task_id, "metadata", description, check)


def _project_url_task(ctx: ArtifactContext, task_id: str, label: str, url: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        metadata, detail = ctx.wheel_metadata()
        if metadata is None:
            return False, detail
        project_urls = metadata.get_all("Project-URL", [])
        expected = f"{label}, {url}"
        return expected in project_urls, expected

    return ScanTask(task_id, "metadata", f"wheel exposes {label} project URL", check)


def _provides_extra_task(ctx: ArtifactContext, task_id: str, extra: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        metadata, detail = ctx.wheel_metadata()
        if metadata is None:
            return False, detail
        extras = metadata.get_all("Provides-Extra", [])
        return extra in extras, f"{extra!r} in Provides-Extra"

    return ScanTask(task_id, "metadata", f"wheel provides extra {extra}", check)


def _wheel_contains_task(ctx: ArtifactContext, task_id: str, relative_path: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        names, detail = ctx.wheel_names()
        if names is None:
            return False, detail
        return relative_path in names, relative_path

    return ScanTask(task_id, "wheel", f"wheel contains {relative_path}", check)


def _wheel_excludes_task(ctx: ArtifactContext, task_id: str, substring: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        names, detail = ctx.wheel_names()
        if names is None:
            return False, detail
        return not any(substring in name for name in names), f"{substring!r} not present in wheel"

    return ScanTask(task_id, "wheel", f"wheel excludes {substring}", check)


def _sdist_contains_task(ctx: ArtifactContext, task_id: str, suffix: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        names, detail = ctx.sdist_names()
        if names is None:
            return False, detail
        return any(name.endswith(suffix) for name in names), suffix

    return ScanTask(task_id, "sdist", f"sdist contains {suffix}", check)


def _sdist_excludes_task(ctx: ArtifactContext, task_id: str, substring: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        names, detail = ctx.sdist_names()
        if names is None:
            return False, detail
        return not any(substring in name for name in names), f"{substring!r} not present in sdist"

    return ScanTask(task_id, "sdist", f"sdist excludes {substring}", check)


def build_tasks(repo_root: Path, artifact_dir: Path) -> list[ScanTask]:
    ctx = ArtifactContext(repo_root, artifact_dir)
    project_name = str(ctx.pyproject_value("project", "name"))
    requires_python = str(ctx.pyproject_value("project", "requires-python"))
    project_urls = ctx.pyproject_value("project", "urls")
    if not isinstance(project_urls, dict):
        raise RuntimeError("project.urls must be a mapping")

    tasks = [
        _artifact_exists_task(ctx, "001", "built wheel exists", ctx.wheel_path),
        _artifact_exists_task(ctx, "002", "built sdist exists", ctx.sdist_path),
        _metadata_header_equals_task(ctx, "003", "wheel name matches project", "Name", project_name),
        _metadata_header_equals_task(ctx, "004", "wheel version matches repo version", "Version", ctx.repo_version()),
        _metadata_header_equals_task(
            ctx,
            "005",
            "wheel Requires-Python matches pyproject",
            "Requires-Python",
            requires_python,
        ),
        _project_url_task(ctx, "006", "Homepage", str(project_urls["Homepage"])),
        _project_url_task(ctx, "007", "Repository", str(project_urls["Repository"])),
        _project_url_task(ctx, "008", "Documentation", str(project_urls["Documentation"])),
        _project_url_task(ctx, "009", "Issues", str(project_urls["Issues"])),
        _project_url_task(ctx, "010", "Changelog", str(project_urls["Changelog"])),
        _provides_extra_task(ctx, "011", "dev"),
        _provides_extra_task(ctx, "012", "scipy"),
        _provides_extra_task(ctx, "013", "networkx"),
        _provides_extra_task(ctx, "014", "tensorboard"),
        _provides_extra_task(ctx, "015", "dgl"),
        _provides_extra_task(ctx, "016", "pyg"),
        _provides_extra_task(ctx, "017", "full"),
        _wheel_contains_task(ctx, "018", "vgl/__init__.py"),
        _wheel_contains_task(ctx, "019", "vgl/version.py"),
        _wheel_excludes_task(ctx, "020", "/docs/plans/"),
        _wheel_excludes_task(ctx, "021", "examples/"),
        _sdist_contains_task(ctx, "022", "/README.md"),
        _sdist_contains_task(ctx, "023", "/docs/releasing.md"),
        _sdist_contains_task(ctx, "024", "/scripts/release_smoke.py"),
    ]

    if len(tasks) != 24:
        raise RuntimeError(f"expected 24 scan tasks, found {len(tasks)}")
    return tasks


def run_tasks(tasks: list[ScanTask]) -> int:
    passed = 0
    for task in tasks:
        ok, detail = task.check()
        status = "PASS" if ok else "FAIL"
        print(f"{status} {task.id} [{task.category}] {task.description} :: {detail}")
        if ok:
            passed += 1
    print(f"SUMMARY {passed}/{len(tasks)} passed")
    return 0 if passed == len(tasks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan built release artifacts for packaging contracts.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory containing built wheel and sdist artifacts. Defaults to <repo-root>/dist.",
    )
    parser.add_argument("--list", action="store_true", help="List scan tasks without executing them.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan. Defaults to this checkout root.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    artifact_dir = (args.artifact_dir or (repo_root / "dist")).resolve()
    tasks = build_tasks(repo_root, artifact_dir)

    if args.list:
        for task in tasks:
            print(f"SCAN {task.id} [{task.category}] {task.description}")
        return 0
    return run_tasks(tasks)


if __name__ == "__main__":
    raise SystemExit(main())
