import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = REPO_ROOT / "scripts" / "public_surface_scan.py"


def test_public_surface_scan_lists_stable_catalog():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    listed = [line for line in completed.stdout.splitlines() if line.startswith("SCAN ")]
    assert len(listed) == 84
    assert any("vgl.data reexports PlanStage from vgl.dataloading" in line for line in listed)
    assert any("vgl.data.plan reexports PlanStage from vgl.dataloading.plan" in line for line in listed)
    assert any("vgl.data.materialize reexports materialize_batch from vgl.dataloading.materialize" in line for line in listed)
    assert any("vgl.data.sampler reexports RandomWalkSampler from vgl.dataloading.advanced" in line for line in listed)
    assert any("vgl.data.sampler reexports GraphSAINTEdgeSampler from vgl.dataloading.advanced" in line for line in listed)
    assert any("vgl.data.transform reexports BaseTransform from vgl.transforms" in line for line in listed)
    assert any("vgl.data.transform reexports NormalizeFeatures from vgl.transforms" in line for line in listed)
    assert any("vgl.data reexports ClusterData from vgl.dataloading" in line for line in listed)
    assert any("vgl.data reexports RandomWalkSampler from vgl.dataloading" in line for line in listed)
    assert any("vgl.data reexports GraphSAINTEdgeSampler from vgl.dataloading" in line for line in listed)
    assert any("vgl.data reexports GraphSAINTRandomWalkSampler from vgl.dataloading" in line for line in listed)
    assert any("vgl.train reexports ModelCheckpoint from vgl.engine" in line for line in listed)
    assert any("vgl.train reexports HitsAtK from vgl.metrics" in line for line in listed)
    assert any("vgl.train reexports FloodingTask from vgl.tasks" in line for line in listed)
    assert any("vgl.core reexports EdgeStore from vgl.graph" in line for line in listed)
    assert any("vgl.core reexports SchemaError from vgl.graph" in line for line in listed)


def test_public_surface_scan_passes_on_repository():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY 84/84 passed" in completed.stdout
