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
            "root",
            "vgl exports DatasetRegistry from vgl.data",
            "vgl/__init__.py",
            "DatasetRegistry",
            "vgl.data.DatasetRegistry",
        ),
        _reexport_task(
            ctx,
            "011",
            "root",
            "vgl exports KarateClubDataset from vgl.data",
            "vgl/__init__.py",
            "KarateClubDataset",
            "vgl.data.KarateClubDataset",
        ),
        _reexport_task(
            ctx,
            "012",
            "root",
            "vgl exports PlanetoidDataset from vgl.data",
            "vgl/__init__.py",
            "PlanetoidDataset",
            "vgl.data.PlanetoidDataset",
        ),
        _reexport_task(
            ctx,
            "013",
            "root",
            "vgl exports TUDataset from vgl.data",
            "vgl/__init__.py",
            "TUDataset",
            "vgl.data.TUDataset",
        ),
        _reexport_task(
            ctx,
            "014",
            "root",
            "vgl exports RandomWalkSampler from vgl.dataloading",
            "vgl/__init__.py",
            "RandomWalkSampler",
            "vgl.dataloading.RandomWalkSampler",
        ),
        _reexport_task(
            ctx,
            "015",
            "root",
            "vgl exports ShaDowKHopSampler from vgl.dataloading",
            "vgl/__init__.py",
            "ShaDowKHopSampler",
            "vgl.dataloading.ShaDowKHopSampler",
        ),
        _reexport_task(
            ctx,
            "016",
            "legacy",
            "vgl.train reexports Evaluator from vgl.engine",
            "vgl/train/__init__.py",
            "Evaluator",
            "vgl.engine.Evaluator",
        ),
        _reexport_task(
            ctx,
            "017",
            "legacy",
            "vgl.train reexports TensorBoardLogger from vgl.engine",
            "vgl/train/__init__.py",
            "TensorBoardLogger",
            "vgl.engine.TensorBoardLogger",
        ),
        _reexport_task(
            ctx,
            "018",
            "legacy",
            "vgl.train reexports Trainer from vgl.engine",
            "vgl/train/__init__.py",
            "Trainer",
            "vgl.engine.Trainer",
        ),
        _reexport_task(
            ctx,
            "019",
            "legacy",
            "vgl.data reexports LinkNeighborSampler from vgl.dataloading",
            "vgl/data/__init__.py",
            "LinkNeighborSampler",
            "vgl.dataloading.LinkNeighborSampler",
        ),
        _reexport_task(
            ctx,
            "020",
            "legacy",
            "vgl.core reexports Graph from vgl.graph",
            "vgl/core/__init__.py",
            "Graph",
            "vgl.graph.Graph",
        ),
        _main_guard_task(ctx, "021", "examples/homo/node_classification.py"),
        _main_guard_task(ctx, "022", "examples/homo/link_prediction.py"),
        _main_guard_task(ctx, "023", "examples/hetero/link_prediction.py"),
        _main_guard_task(ctx, "024", "examples/temporal/event_prediction.py"),
        _forbidden_import_task(ctx, "025", "examples"),
        _forbidden_import_task(ctx, "026", "tests/integration"),
        _reexport_task(
            ctx,
            "027",
            "data",
            "vgl.data reexports DatasetRegistry from vgl.data.public",
            "vgl/data/__init__.py",
            "DatasetRegistry",
            "vgl.data.public.DatasetRegistry",
        ),
        _reexport_task(
            ctx,
            "028",
            "data",
            "vgl.data reexports KarateClubDataset from vgl.data.public",
            "vgl/data/__init__.py",
            "KarateClubDataset",
            "vgl.data.public.KarateClubDataset",
        ),
        _reexport_task(
            ctx,
            "029",
            "data",
            "vgl.data reexports PlanetoidDataset from vgl.data.public",
            "vgl/data/__init__.py",
            "PlanetoidDataset",
            "vgl.data.public.PlanetoidDataset",
        ),
        _reexport_task(
            ctx,
            "030",
            "data",
            "vgl.data reexports TUDataset from vgl.data.public",
            "vgl/data/__init__.py",
            "TUDataset",
            "vgl.data.public.TUDataset",
        ),
        _reexport_task(
            ctx,
            "031",
            "dataloading",
            "vgl.dataloading reexports RandomWalkSampler from vgl.dataloading.advanced",
            "vgl/dataloading/__init__.py",
            "RandomWalkSampler",
            "vgl.dataloading.advanced.RandomWalkSampler",
        ),
        _reexport_task(
            ctx,
            "032",
            "dataloading",
            "vgl.dataloading reexports Node2VecWalkSampler from vgl.dataloading.advanced",
            "vgl/dataloading/__init__.py",
            "Node2VecWalkSampler",
            "vgl.dataloading.advanced.Node2VecWalkSampler",
        ),
        _reexport_task(
            ctx,
            "033",
            "dataloading",
            "vgl.dataloading reexports GraphSAINTNodeSampler from vgl.dataloading.advanced",
            "vgl/dataloading/__init__.py",
            "GraphSAINTNodeSampler",
            "vgl.dataloading.advanced.GraphSAINTNodeSampler",
        ),
        _reexport_task(
            ctx,
            "034",
            "dataloading",
            "vgl.dataloading reexports ClusterData from vgl.dataloading.advanced",
            "vgl/dataloading/__init__.py",
            "ClusterData",
            "vgl.dataloading.advanced.ClusterData",
        ),
        _reexport_task(
            ctx,
            "035",
            "dataloading",
            "vgl.dataloading reexports ClusterLoader from vgl.dataloading.advanced",
            "vgl/dataloading/__init__.py",
            "ClusterLoader",
            "vgl.dataloading.advanced.ClusterLoader",
        ),
        _reexport_task(
            ctx,
            "036",
            "dataloading",
            "vgl.dataloading reexports ShaDowKHopSampler from vgl.dataloading.advanced",
            "vgl/dataloading/__init__.py",
            "ShaDowKHopSampler",
            "vgl.dataloading.advanced.ShaDowKHopSampler",
        ),
        _main_guard_task(ctx, "037", "examples/homo/planetoid_node_classification.py"),
        _main_guard_task(ctx, "038", "examples/homo/tu_graph_classification.py"),
        _main_guard_task(ctx, "039", "examples/homo/graph_saint_node_classification.py"),
        _main_guard_task(ctx, "040", "examples/homo/cluster_gcn_node_classification.py"),
        _reexport_task(
            ctx,
            "041",
            "root",
            "vgl exports ClusterData from vgl.dataloading",
            "vgl/__init__.py",
            "ClusterData",
            "vgl.dataloading.ClusterData",
        ),
        _reexport_task(
            ctx,
            "042",
            "root",
            "vgl exports ClusterLoader from vgl.dataloading",
            "vgl/__init__.py",
            "ClusterLoader",
            "vgl.dataloading.ClusterLoader",
        ),
        _reexport_task(
            ctx,
            "043",
            "root",
            "vgl exports Node2VecWalkSampler from vgl.dataloading",
            "vgl/__init__.py",
            "Node2VecWalkSampler",
            "vgl.dataloading.Node2VecWalkSampler",
        ),
        _reexport_task(
            ctx,
            "044",
            "root",
            "vgl exports GraphSAINTNodeSampler from vgl.dataloading",
            "vgl/__init__.py",
            "GraphSAINTNodeSampler",
            "vgl.dataloading.GraphSAINTNodeSampler",
        ),
        _reexport_task(
            ctx,
            "045",
            "root",
            "vgl exports GraphSAINTEdgeSampler from vgl.dataloading",
            "vgl/__init__.py",
            "GraphSAINTEdgeSampler",
            "vgl.dataloading.GraphSAINTEdgeSampler",
        ),
        _reexport_task(
            ctx,
            "046",
            "root",
            "vgl exports GraphSAINTRandomWalkSampler from vgl.dataloading",
            "vgl/__init__.py",
            "GraphSAINTRandomWalkSampler",
            "vgl.dataloading.GraphSAINTRandomWalkSampler",
        ),
        _reexport_task(
            ctx,
            "047",
            "dataloading",
            "vgl.dataloading reexports GraphSAINTEdgeSampler from vgl.dataloading.advanced",
            "vgl/dataloading/__init__.py",
            "GraphSAINTEdgeSampler",
            "vgl.dataloading.advanced.GraphSAINTEdgeSampler",
        ),
        _reexport_task(
            ctx,
            "048",
            "dataloading",
            "vgl.dataloading reexports GraphSAINTRandomWalkSampler from vgl.dataloading.advanced",
            "vgl/dataloading/__init__.py",
            "GraphSAINTRandomWalkSampler",
            "vgl.dataloading.advanced.GraphSAINTRandomWalkSampler",
        ),
        _reexport_task(
            ctx,
            "049",
            "legacy",
            "vgl.data reexports PlanStage from vgl.dataloading",
            "vgl/data/__init__.py",
            "PlanStage",
            "vgl.dataloading.PlanStage",
        ),
        _reexport_task(
            ctx,
            "050",
            "legacy",
            "vgl.data reexports SamplingPlan from vgl.dataloading",
            "vgl/data/__init__.py",
            "SamplingPlan",
            "vgl.dataloading.SamplingPlan",
        ),
        _reexport_task(
            ctx,
            "051",
            "legacy",
            "vgl.data reexports PlanExecutor from vgl.dataloading",
            "vgl/data/__init__.py",
            "PlanExecutor",
            "vgl.dataloading.PlanExecutor",
        ),
        _reexport_task(
            ctx,
            "052",
            "legacy",
            "vgl.data reexports MaterializationContext from vgl.dataloading",
            "vgl/data/__init__.py",
            "MaterializationContext",
            "vgl.dataloading.MaterializationContext",
        ),
        _reexport_task(
            ctx,
            "053",
            "legacy",
            "vgl.data reexports GraphSeedRequest from vgl.dataloading",
            "vgl/data/__init__.py",
            "GraphSeedRequest",
            "vgl.dataloading.GraphSeedRequest",
        ),
        _reexport_task(
            ctx,
            "054",
            "legacy",
            "vgl.data reexports NodeSeedRequest from vgl.dataloading",
            "vgl/data/__init__.py",
            "NodeSeedRequest",
            "vgl.dataloading.NodeSeedRequest",
        ),
        _reexport_task(
            ctx,
            "055",
            "legacy",
            "vgl.data reexports materialize_context from vgl.dataloading",
            "vgl/data/__init__.py",
            "materialize_context",
            "vgl.dataloading.materialize_context",
        ),
        _reexport_task(
            ctx,
            "056",
            "legacy",
            "vgl.data reexports materialize_batch from vgl.dataloading",
            "vgl/data/__init__.py",
            "materialize_batch",
            "vgl.dataloading.materialize_batch",
        ),
        _reexport_task(
            ctx,
            "057",
            "legacy",
            "vgl.data.plan reexports PlanStage from vgl.dataloading.plan",
            "vgl/data/plan.py",
            "PlanStage",
            "vgl.dataloading.plan.PlanStage",
        ),
        _reexport_task(
            ctx,
            "058",
            "legacy",
            "vgl.data.executor reexports PlanExecutor from vgl.dataloading.executor",
            "vgl/data/executor.py",
            "PlanExecutor",
            "vgl.dataloading.executor.PlanExecutor",
        ),
        _reexport_task(
            ctx,
            "059",
            "legacy",
            "vgl.data.requests reexports NodeSeedRequest from vgl.dataloading.requests",
            "vgl/data/requests.py",
            "NodeSeedRequest",
            "vgl.dataloading.requests.NodeSeedRequest",
        ),
        _reexport_task(
            ctx,
            "060",
            "legacy",
            "vgl.data.materialize reexports materialize_batch from vgl.dataloading.materialize",
            "vgl/data/materialize.py",
            "materialize_batch",
            "vgl.dataloading.materialize.materialize_batch",
        ),
        _reexport_task(
            ctx,
            "061",
            "legacy",
            "vgl.data.sampler reexports LinkNeighborSampler from vgl.dataloading.sampler",
            "vgl/data/sampler.py",
            "LinkNeighborSampler",
            "vgl.dataloading.sampler.LinkNeighborSampler",
        ),
        _reexport_task(
            ctx,
            "062",
            "legacy",
            "vgl.data.sampler reexports RandomWalkSampler from vgl.dataloading.advanced",
            "vgl/data/sampler.py",
            "RandomWalkSampler",
            "vgl.dataloading.advanced.RandomWalkSampler",
        ),
        _reexport_task(
            ctx,
            "063",
            "legacy",
            "vgl.data.sampler reexports GraphSAINTEdgeSampler from vgl.dataloading.advanced",
            "vgl/data/sampler.py",
            "GraphSAINTEdgeSampler",
            "vgl.dataloading.advanced.GraphSAINTEdgeSampler",
        ),
        _reexport_task(
            ctx,
            "064",
            "legacy",
            "vgl.data.sampler reexports Node2VecWalkSampler from vgl.dataloading.advanced",
            "vgl/data/sampler.py",
            "Node2VecWalkSampler",
            "vgl.dataloading.advanced.Node2VecWalkSampler",
        ),
        _reexport_task(
            ctx,
            "065",
            "legacy",
            "vgl.data.transform reexports BaseTransform from vgl.transforms",
            "vgl/data/transform.py",
            "BaseTransform",
            "vgl.transforms.BaseTransform",
        ),
        _reexport_task(
            ctx,
            "066",
            "legacy",
            "vgl.data.transform reexports NormalizeFeatures from vgl.transforms",
            "vgl/data/transform.py",
            "NormalizeFeatures",
            "vgl.transforms.NormalizeFeatures",
        ),
        _reexport_task(
            ctx,
            "067",
            "legacy",
            "vgl.data.transform reexports RandomNodeSplit from vgl.transforms",
            "vgl/data/transform.py",
            "RandomNodeSplit",
            "vgl.transforms.RandomNodeSplit",
        ),
        _reexport_task(
            ctx,
            "068",
            "legacy",
            "vgl.data.transform reexports RandomLinkSplit from vgl.transforms",
            "vgl/data/transform.py",
            "RandomLinkSplit",
            "vgl.transforms.RandomLinkSplit",
        ),
        _reexport_task(
            ctx,
            "069",
            "legacy",
            "vgl.data reexports ClusterData from vgl.dataloading",
            "vgl/data/__init__.py",
            "ClusterData",
            "vgl.dataloading.ClusterData",
        ),
        _reexport_task(
            ctx,
            "070",
            "legacy",
            "vgl.data reexports ClusterLoader from vgl.dataloading",
            "vgl/data/__init__.py",
            "ClusterLoader",
            "vgl.dataloading.ClusterLoader",
        ),
        _reexport_task(
            ctx,
            "071",
            "legacy",
            "vgl.data reexports RandomWalkSampler from vgl.dataloading",
            "vgl/data/__init__.py",
            "RandomWalkSampler",
            "vgl.dataloading.RandomWalkSampler",
        ),
        _reexport_task(
            ctx,
            "072",
            "legacy",
            "vgl.data reexports Node2VecWalkSampler from vgl.dataloading",
            "vgl/data/__init__.py",
            "Node2VecWalkSampler",
            "vgl.dataloading.Node2VecWalkSampler",
        ),
        _reexport_task(
            ctx,
            "073",
            "legacy",
            "vgl.data reexports GraphSAINTNodeSampler from vgl.dataloading",
            "vgl/data/__init__.py",
            "GraphSAINTNodeSampler",
            "vgl.dataloading.GraphSAINTNodeSampler",
        ),
        _reexport_task(
            ctx,
            "074",
            "legacy",
            "vgl.data reexports GraphSAINTEdgeSampler from vgl.dataloading",
            "vgl/data/__init__.py",
            "GraphSAINTEdgeSampler",
            "vgl.dataloading.GraphSAINTEdgeSampler",
        ),
        _reexport_task(
            ctx,
            "075",
            "legacy",
            "vgl.data reexports GraphSAINTRandomWalkSampler from vgl.dataloading",
            "vgl/data/__init__.py",
            "GraphSAINTRandomWalkSampler",
            "vgl.dataloading.GraphSAINTRandomWalkSampler",
        ),
        _reexport_task(
            ctx,
            "076",
            "legacy",
            "vgl.data reexports ShaDowKHopSampler from vgl.dataloading",
            "vgl/data/__init__.py",
            "ShaDowKHopSampler",
            "vgl.dataloading.ShaDowKHopSampler",
        ),
        _reexport_task(
            ctx,
            "077",
            "legacy",
            "vgl.train reexports ModelCheckpoint from vgl.engine",
            "vgl/train/__init__.py",
            "ModelCheckpoint",
            "vgl.engine.ModelCheckpoint",
        ),
        _reexport_task(
            ctx,
            "078",
            "legacy",
            "vgl.train reexports HitsAtK from vgl.metrics",
            "vgl/train/__init__.py",
            "HitsAtK",
            "vgl.metrics.HitsAtK",
        ),
        _reexport_task(
            ctx,
            "079",
            "legacy",
            "vgl.train reexports FloodingTask from vgl.tasks",
            "vgl/train/__init__.py",
            "FloodingTask",
            "vgl.tasks.FloodingTask",
        ),
        _reexport_task(
            ctx,
            "080",
            "legacy",
            "vgl.train reexports SAM from vgl.engine",
            "vgl/train/__init__.py",
            "SAM",
            "vgl.engine.SAM",
        ),
        _reexport_task(
            ctx,
            "081",
            "legacy",
            "vgl.core reexports EdgeStore from vgl.graph",
            "vgl/core/__init__.py",
            "EdgeStore",
            "vgl.graph.EdgeStore",
        ),
        _reexport_task(
            ctx,
            "082",
            "legacy",
            "vgl.core reexports NodeStore from vgl.graph",
            "vgl/core/__init__.py",
            "NodeStore",
            "vgl.graph.NodeStore",
        ),
        _reexport_task(
            ctx,
            "083",
            "legacy",
            "vgl.core reexports GNNError from vgl.graph",
            "vgl/core/__init__.py",
            "GNNError",
            "vgl.graph.GNNError",
        ),
        _reexport_task(
            ctx,
            "084",
            "legacy",
            "vgl.core reexports SchemaError from vgl.graph",
            "vgl/core/__init__.py",
            "SchemaError",
            "vgl.graph.SchemaError",
        ),
    ]

    if len(tasks) != 84:
        raise RuntimeError(f"expected 84 scan tasks, found {len(tasks)}")
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
