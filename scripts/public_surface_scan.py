#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


CheckFn = Callable[[], tuple[bool, str]]

FORBIDDEN_IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+vgl\.core(?:\.|\s+import)\b", re.MULTILINE),
    re.compile(r"^\s*import\s+vgl\.core(?:\.|\b)", re.MULTILINE),
    re.compile(r"^\s*from\s+vgl\.data(?:\.|\s+import)\b", re.MULTILINE),
    re.compile(r"^\s*import\s+vgl\.data(?:\.|\b)", re.MULTILINE),
    re.compile(r"^\s*from\s+vgl\.train(?:\.|\s+import)\b", re.MULTILINE),
    re.compile(r"^\s*import\s+vgl\.train(?:\.|\b)", re.MULTILINE),
)


@dataclass(frozen=True)
class ScanTask:
    id: str
    category: str
    description: str
    check: CheckFn


@dataclass(frozen=True)
class ModuleSurface:
    imports: dict[str, str]
    exports: set[str]


class ScanContext:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self._text_cache: dict[Path, str] = {}
        self._surface_cache: dict[Path, ModuleSurface] = {}

    def resolve(self, relative_path: str) -> Path:
        return self.repo_root / relative_path

    def read_text(self, relative_path: str) -> str:
        path = self.resolve(relative_path)
        cached = self._text_cache.get(path)
        if cached is None:
            cached = path.read_text(encoding="utf-8")
            self._text_cache[path] = cached
        return cached

    def module_surface(self, relative_path: str) -> ModuleSurface:
        path = self.resolve(relative_path)
        cached = self._surface_cache.get(path)
        if cached is not None:
            return cached

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: dict[str, str] = {}
        exports: set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    imports[local_name] = f"{node.module}.{alias.name}"
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        exports = _string_sequence(node.value)
                        break

        cached = ModuleSurface(imports=imports, exports=exports)
        self._surface_cache[path] = cached
        return cached

    def has_main_guard(self, relative_path: str) -> tuple[bool, str]:
        text = self.read_text(relative_path)
        has_guard = 'if __name__ == "__main__":' in text or "if __name__ == '__main__':" in text
        return has_guard, relative_path

    def scan_tree_for_forbidden_imports(self, relative_dir: str) -> tuple[bool, str]:
        root = self.resolve(relative_dir)
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in FORBIDDEN_IMPORT_PATTERNS):
                offenders.append(str(path.relative_to(self.repo_root)))
        if offenders:
            return False, ", ".join(offenders[:5])
        return True, f"{relative_dir} has no vgl.core/vgl.data/vgl.train imports"


def _string_sequence(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set()
    values: set[str] = set()
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.add(item.value)
    return values


def _reexport_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    relative_path: str,
    symbol: str,
    expected_source: str,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        surface = ctx.module_surface(relative_path)
        imported_source = surface.imports.get(symbol)
        if imported_source != expected_source:
            return False, f"{relative_path} imports {symbol!r} from {imported_source!r}"
        if symbol not in surface.exports:
            return False, f"{relative_path} __all__ missing {symbol!r}"
        return True, f"{relative_path} exports {symbol} from {expected_source}"

    return ScanTask(task_id, category, description, check)


def _main_guard_task(ctx: ScanContext, task_id: str, relative_path: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        path = ctx.resolve(relative_path)
        if not path.exists():
            return False, f"{relative_path} missing"
        return ctx.has_main_guard(relative_path)

    return ScanTask(task_id, "example", f"{relative_path} has __main__ guard", check)


def _forbidden_import_task(ctx: ScanContext, task_id: str, relative_dir: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        return ctx.scan_tree_for_forbidden_imports(relative_dir)

    return ScanTask(task_id, "imports", f"{relative_dir} avoids legacy import paths", check)


def build_tasks(repo_root: Path) -> list[ScanTask]:
    ctx = ScanContext(repo_root)
    tasks = [
        _reexport_task(ctx, "001", "root", "vgl exports Graph from vgl.graph", "vgl/__init__.py", "Graph", "vgl.graph.Graph"),
        _reexport_task(
            ctx,
            "002",
            "root",
            "vgl exports GraphBatch from vgl.graph",
            "vgl/__init__.py",
            "GraphBatch",
            "vgl.graph.GraphBatch",
        ),
        _reexport_task(
            ctx,
            "003",
            "root",
            "vgl exports DataLoader from vgl.dataloading",
            "vgl/__init__.py",
            "DataLoader",
            "vgl.dataloading.DataLoader",
        ),
        _reexport_task(
            ctx,
            "004",
            "root",
            "vgl exports Trainer from vgl.engine",
            "vgl/__init__.py",
            "Trainer",
            "vgl.engine.Trainer",
        ),
        _reexport_task(
            ctx,
            "005",
            "root",
            "vgl exports MessagePassing from vgl.nn",
            "vgl/__init__.py",
            "MessagePassing",
            "vgl.nn.MessagePassing",
        ),
        _reexport_task(
            ctx,
            "006",
            "root",
            "vgl exports AGNNConv from vgl.nn",
            "vgl/__init__.py",
            "AGNNConv",
            "vgl.nn.AGNNConv",
        ),
        _reexport_task(
            ctx,
            "007",
            "root",
            "vgl exports LinkPredictionTask from vgl.tasks",
            "vgl/__init__.py",
            "LinkPredictionTask",
            "vgl.tasks.LinkPredictionTask",
        ),
        _reexport_task(
            ctx,
            "008",
            "root",
            "vgl exports TemporalEventPredictionTask from vgl.tasks",
            "vgl/__init__.py",
            "TemporalEventPredictionTask",
            "vgl.tasks.TemporalEventPredictionTask",
        ),
        _reexport_task(
            ctx,
            "009",
            "root",
            "vgl exports __version__ from vgl.version",
            "vgl/__init__.py",
            "__version__",
            "vgl.version.__version__",
        ),
        _reexport_task(
            ctx,
            "010",
            "legacy",
            "vgl.train reexports Evaluator from vgl.engine",
            "vgl/train/__init__.py",
            "Evaluator",
            "vgl.engine.Evaluator",
        ),
        _reexport_task(
            ctx,
            "011",
            "legacy",
            "vgl.train reexports TensorBoardLogger from vgl.engine",
            "vgl/train/__init__.py",
            "TensorBoardLogger",
            "vgl.engine.TensorBoardLogger",
        ),
        _reexport_task(
            ctx,
            "012",
            "legacy",
            "vgl.train reexports Trainer from vgl.engine",
            "vgl/train/__init__.py",
            "Trainer",
            "vgl.engine.Trainer",
        ),
        _reexport_task(
            ctx,
            "013",
            "legacy",
            "vgl.data reexports LinkNeighborSampler from vgl.dataloading",
            "vgl/data/__init__.py",
            "LinkNeighborSampler",
            "vgl.dataloading.LinkNeighborSampler",
        ),
        _reexport_task(
            ctx,
            "014",
            "legacy",
            "vgl.core reexports Graph from vgl.graph",
            "vgl/core/__init__.py",
            "Graph",
            "vgl.graph.Graph",
        ),
        _main_guard_task(ctx, "015", "examples/homo/node_classification.py"),
        _main_guard_task(ctx, "016", "examples/homo/link_prediction.py"),
        _main_guard_task(ctx, "017", "examples/hetero/link_prediction.py"),
        _main_guard_task(ctx, "018", "examples/temporal/event_prediction.py"),
        _forbidden_import_task(ctx, "019", "examples"),
        _forbidden_import_task(ctx, "020", "tests/integration"),
    ]

    if len(tasks) != 20:
        raise RuntimeError(f"expected 20 scan tasks, found {len(tasks)}")
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
    parser = argparse.ArgumentParser(description="Scan public import/export and example script contracts.")
    parser.add_argument("--list", action="store_true", help="List scan tasks without executing them.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan. Defaults to this checkout root.",
    )
    args = parser.parse_args()

    tasks = build_tasks(args.repo_root.resolve())
    if args.list:
        for task in tasks:
            print(f"SCAN {task.id} [{task.category}] {task.description}")
        return 0
    return run_tasks(tasks)


if __name__ == "__main__":
    raise SystemExit(main())
