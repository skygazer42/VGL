import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_example_modules_import_cleanly():
    example_paths = [
        "examples/homo/planetoid_node_classification.py",
        "examples/homo/tu_graph_classification.py",
        "examples/homo/graph_saint_node_classification.py",
        "examples/homo/cluster_gcn_node_classification.py",
    ]

    for relative_path in example_paths:
        module = _load_module(relative_path)
        assert hasattr(module, "main")
