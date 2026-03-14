# Homogeneous Convolution Expansion Phase 6 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first stable batch of additional homogeneous GNN operators by shipping `GINConv`, `GATv2Conv`, and `APPNPConv` with tests, exports, and one real example.

**Architecture:** Extend the existing `vgl.nn.conv` module rather than creating a second operator namespace. Add one file per operator, export them through `vgl.nn.conv`, `vgl.nn`, and `vgl`, and keep each layer compatible with the current homogeneous `Graph` plus `Trainer` flow. Allow custom per-layer forward logic where the current `MessagePassing` base is too limited, especially for `GATv2Conv`.

**Tech Stack:** Python 3.11+, PyTorch, pytest, ruff, mypy

**Execution Rules:** Use `@test-driven-development` for every code task, keep commits small, and use `@verification-before-completion` before claiming the phase is done.

---

### Task 1: Add Operator Contract Tests and Public Export Coverage

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing operator tests**

Extend `tests/nn/test_convs.py` with:

```python
import pytest
import torch

from vgl import Graph
from vgl.nn.conv.appnp import APPNPConv
from vgl.nn.conv.gatv2 import GATv2Conv
from vgl.nn.conv.gin import GINConv


def _homo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
    )


def _hetero_graph():
    return Graph.hetero(
        nodes={"paper": {"x": torch.randn(2, 4)}, "author": {"x": torch.randn(2, 4)}},
        edges={("author", "writes", "paper"): {"edge_index": torch.tensor([[0, 1], [1, 0]])}},
    )


def test_gin_conv_accepts_graph_input():
    conv = GINConv(in_channels=4, out_channels=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_gin_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GINConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_gatv2_conv_respects_head_output_shapes():
    graph = _homo_graph()

    concat_conv = GATv2Conv(in_channels=4, out_channels=3, heads=2, concat=True)
    mean_conv = GATv2Conv(in_channels=4, out_channels=3, heads=2, concat=False)

    concat_out = concat_conv(graph)
    mean_out = mean_conv(graph)

    assert concat_out.shape == (3, 6)
    assert mean_out.shape == (3, 3)


def test_appnp_conv_accepts_graph_input():
    conv = APPNPConv(in_channels=4, out_channels=3, steps=2, alpha=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


@pytest.mark.parametrize("conv_cls, kwargs", [
    (GINConv, {"in_channels": 4, "out_channels": 3}),
    (GATv2Conv, {"in_channels": 4, "out_channels": 3}),
    (APPNPConv, {"in_channels": 4, "out_channels": 3}),
])
def test_new_homo_convs_reject_hetero_graph_input(conv_cls, kwargs):
    conv = conv_cls(**kwargs)

    with pytest.raises(ValueError, match="homogeneous"):
        conv(_hetero_graph())
```

Update `tests/test_package_exports.py` so the root package also exposes:

```python
from vgl import APPNPConv, GATv2Conv, GINConv

assert APPNPConv.__name__ == "APPNPConv"
assert GATv2Conv.__name__ == "GATv2Conv"
assert GINConv.__name__ == "GINConv"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/nn/test_convs.py tests/test_package_exports.py -v`
Expected: `FAIL` because the three new operator classes and exports do not exist yet.

**Step 3: Write minimal placeholder exports only if needed to clarify failure shape**

Do not write production operator logic yet. If test collection needs import stubs to reach the intended runtime failures, add the thinnest placeholders and re-run the targeted tests.

**Step 4: Re-run tests to verify the failure is about missing operator behavior**

Run: `python -m pytest tests/nn/test_convs.py tests/test_package_exports.py -v`
Expected: `FAIL` for the intended operator or export gaps, not unrelated import problems.

**Step 5: Commit nothing yet**

Do not commit in this task. Wait until the real operator implementations exist.

### Task 2: Implement `GINConv`, `GATv2Conv`, and `APPNPConv` with Public Exports

**Files:**
- Create: `vgl/nn/conv/gin.py`
- Create: `vgl/nn/conv/gatv2.py`
- Create: `vgl/nn/conv/appnp.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`
- Verify: `tests/nn/test_convs.py`
- Verify: `tests/test_package_exports.py`

**Step 1: Write minimal implementation**

Create `vgl/nn/conv/gin.py`:

```python
import torch
from torch import nn


class GINConv(nn.Module):
    def __init__(self, in_channels, out_channels, eps=0.0, train_eps=False):
        super().__init__()
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(float(eps)))
        else:
            self.register_buffer("eps", torch.tensor(float(eps)))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            if not hasattr(graph_or_x, "edge_index"):
                raise ValueError("GINConv requires a homogeneous graph or x with edge_index")
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x

        row, col = edge_index
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, col, x[row])
        return self.mlp((1 + self.eps) * x + aggregated)
```

Create `vgl/nn/conv/gatv2.py`:

```python
import torch
from torch import nn


class GATv2Conv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True):
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.linear = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att = nn.Parameter(torch.randn(heads, out_channels))

    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            if not hasattr(graph_or_x, "edge_index"):
                raise ValueError("GATv2Conv requires a homogeneous graph or x with edge_index")
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x

        row, col = edge_index
        projected = self.linear(x).view(x.size(0), self.heads, self.out_channels)
        src = projected[row]
        dst = projected[col]
        scores = torch.tanh(src + dst)
        scores = (scores * self.att).sum(dim=-1)
        scores = torch.exp(scores - scores.amax())

        out = torch.zeros(
            x.size(0),
            self.heads,
            self.out_channels,
            dtype=projected.dtype,
            device=projected.device,
        )
        normalizer = torch.zeros(x.size(0), self.heads, dtype=projected.dtype, device=projected.device)
        weighted = src * scores.unsqueeze(-1)
        out.index_add_(0, col, weighted)
        normalizer.index_add_(0, col, scores)
        out = out / normalizer.clamp_min(1e-12).unsqueeze(-1)
        if self.concat:
            return out.reshape(x.size(0), self.heads * self.out_channels)
        return out.mean(dim=1)
```

Create `vgl/nn/conv/appnp.py`:

```python
import torch
from torch import nn


class APPNPConv(nn.Module):
    def __init__(self, in_channels, out_channels, steps=10, alpha=0.1):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.steps = steps
        self.alpha = alpha

    def forward(self, graph_or_x, edge_index=None):
        if edge_index is None:
            if not hasattr(graph_or_x, "edge_index"):
                raise ValueError("APPNPConv requires a homogeneous graph or x with edge_index")
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x

        row, col = edge_index
        initial = self.linear(x)
        out = initial
        for _ in range(self.steps):
            propagated = torch.zeros_like(out)
            propagated.index_add_(0, col, out[row])
            out = (1 - self.alpha) * propagated + self.alpha * initial
        return out
```

Before finalizing the code, add an early homogeneous-only check. The minimal consistent rule is:

```python
if edge_index is None and not hasattr(graph_or_x, "edge_index"):
    raise ValueError(...)
if edge_index is None and len(graph_or_x.nodes) != 1:
    raise ValueError("... homogeneous ...")
```

Update exports in:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

**Step 2: Run the operator and export tests**

Run: `python -m pytest tests/nn/test_convs.py tests/test_package_exports.py -v`
Expected: `PASS`

**Step 3: Refine only if needed**

If any failures remain, make the smallest correction needed:

- homogeneous graph validation wording
- `GATv2Conv` output shape handling
- `APPNPConv` propagation tensor shapes

Do not broaden parameters in this phase.

**Step 4: Re-run tests to confirm green**

Run: `python -m pytest tests/nn/test_convs.py tests/test_package_exports.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add vgl/nn/conv/gin.py vgl/nn/conv/gatv2.py vgl/nn/conv/appnp.py vgl/nn/conv/__init__.py vgl/nn/__init__.py vgl/__init__.py tests/nn/test_convs.py tests/test_package_exports.py
git commit -m "feat: add homogeneous conv expansion"
```

### Task 3: Add One Compact Integration Path for Training-Loop Compatibility

**Files:**
- Create: `tests/integration/test_homo_conv_zoo.py`

**Step 1: Write the failing integration test**

Create `tests/integration/test_homo_conv_zoo.py`:

```python
import torch
from torch import nn

from vgl import APPNPConv, GATv2Conv, GINConv, Graph, NodeClassificationTask, Trainer


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
        y=torch.tensor([0, 1, 0]),
        train_mask=torch.tensor([True, True, True]),
        val_mask=torch.tensor([True, True, True]),
        test_mask=torch.tensor([True, True, True]),
    )


def _model(conv):
    class TinyModel(nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op
            out_channels = getattr(op, "out_channels", 2)
            heads = getattr(op, "heads", 1)
            concat = getattr(op, "concat", False)
            hidden = out_channels * heads if concat else out_channels
            self.head = nn.Linear(hidden, 2)

        def forward(self, graph):
            return self.head(self.op(graph))

    return TinyModel(conv)


def test_new_homo_convs_plug_into_training_loop():
    convs = [
        GINConv(in_channels=4, out_channels=4),
        GATv2Conv(in_channels=4, out_channels=4, heads=2, concat=False),
        APPNPConv(in_channels=4, out_channels=4, steps=2, alpha=0.1),
    ]

    for conv in convs:
        trainer = Trainer(
            model=_model(conv),
            task=NodeClassificationTask(
                target="y",
                split=("train_mask", "val_mask", "test_mask"),
                metrics=["accuracy"],
            ),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=1,
        )
        history = trainer.fit(_graph(), val_data=_graph())

        assert history["epochs"] == 1
        assert "loss" in history["train"][0]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_homo_conv_zoo.py -v`
Expected: `FAIL` until the new operators are fully wired and importable from the root package.

**Step 3: Write minimal implementation adjustments only if needed**

If the test exposes integration-specific issues, make the smallest fixes needed:

- root exports missing
- `GATv2Conv` `concat=False` output shape wrong
- homogeneous graph validation too strict

Do not add new features outside the planned operator scope.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_homo_conv_zoo.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add tests/integration/test_homo_conv_zoo.py vgl/nn vgl/__init__.py
git commit -m "feat: integrate new homo convs with training loop"
```

### Task 4: Add a Single Example and Update Docs

**Files:**
- Create: `examples/homo/conv_zoo.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`

**Step 1: Write the failing example-backed smoke test**

Because there is no dedicated example test harness yet, use the integration test below as the spec for the example surface. The example should export the tiny model helpers so the code path is inspectable and runnable.

**Step 2: Write minimal example implementation**

Create `examples/homo/conv_zoo.py`:

```python
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl import APPNPConv, GATv2Conv, GINConv, Graph, NodeClassificationTask, Trainer


def build_demo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
        y=torch.tensor([0, 1, 0]),
        train_mask=torch.tensor([True, True, True]),
        val_mask=torch.tensor([True, True, True]),
        test_mask=torch.tensor([True, True, True]),
    )


class TinyConvModel(nn.Module):
    def __init__(self, conv, hidden_channels):
        super().__init__()
        self.conv = conv
        self.head = nn.Linear(hidden_channels, 2)

    def forward(self, graph):
        return self.head(self.conv(graph))


def run_one(name, conv, hidden_channels):
    graph = build_demo_graph()
    trainer = Trainer(
        model=TinyConvModel(conv, hidden_channels),
        task=NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            metrics=["accuracy"],
        ),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    history = trainer.fit(graph, val_data=graph)
    return {"name": name, "loss": history["train"][-1]["loss"]}


def main():
    results = [
        run_one("gin", GINConv(in_channels=4, out_channels=4), 4),
        run_one("gatv2", GATv2Conv(in_channels=4, out_channels=4, heads=2, concat=False), 4),
        run_one("appnp", APPNPConv(in_channels=4, out_channels=4, steps=2, alpha=0.1), 4),
    ]
    print(results)
    return results


if __name__ == "__main__":
    main()
```

Update docs:

- `README.md`: list `GINConv`, `GATv2Conv`, and `APPNPConv`; add `python examples/homo/conv_zoo.py`
- `docs/quickstart.md`: add one short note that models can swap in `GINConv`, `GATv2Conv`, or `APPNPConv`
- `docs/core-concepts.md`: add one short note that `vgl.nn.conv` is the operator entrypoint for built-in convolution layers

**Step 3: Run the example**

Run: `python examples/homo/conv_zoo.py`
Expected: prints a list of result dictionaries for `gin`, `gatv2`, and `appnp`

**Step 4: Make only the smallest follow-up fixes**

If the example fails, fix only the mismatched hidden sizes, imports, or operator call shapes it reveals.

**Step 5: Commit**

```bash
git add examples/homo/conv_zoo.py README.md docs/quickstart.md docs/core-concepts.md
git commit -m "feat: add homo conv zoo example"
```

### Task 5: Run Full Regression Verification

**Files:**
- Verify only

**Step 1: Run the focused Phase 6 tests**

Run: `python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`
Expected: `PASS`

**Step 2: Run the full test suite**

Run: `python -m pytest -v`
Expected: `PASS`

**Step 3: Run lint and type checking**

Run: `python -m ruff check .`
Expected: `All checks passed`

Run: `python -m mypy vgl`
Expected: `Success: no issues found`

**Step 4: Run the example smoke suite**

Run: `python examples/homo/node_classification.py`
Expected: prints a training-loop result dictionary

Run: `python examples/homo/graph_classification.py`
Expected: prints a training result dictionary

Run: `python examples/homo/link_prediction.py`
Expected: prints a training result dictionary

Run: `python examples/homo/conv_zoo.py`
Expected: prints a list of result dictionaries

Run: `python examples/hetero/node_classification.py`
Expected: prints a training result dictionary

Run: `python examples/hetero/graph_classification.py`
Expected: prints a training result dictionary

Run: `python examples/temporal/event_prediction.py`
Expected: prints a training result dictionary

**Step 5: Commit final polish only if verification required follow-up edits**

```bash
git add README.md docs/quickstart.md docs/core-concepts.md vgl tests examples
git commit -m "chore: verify homo conv expansion"
```

Do not create an empty commit. If verification passes without any follow-up edits, stop after recording the evidence.
