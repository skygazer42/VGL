# Homogeneous Structural Aggregation Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `H2GCNConv` and `EGConv` as stable homogeneous structural-aggregation operators and wire them into the existing export, test, and example surface.

**Architecture:** Add one file per operator under `vgl.nn.conv`, reuse `_homo.py` for tiny aggregation helpers, and extend the compact integration path and `conv_zoo` example instead of introducing new training abstractions. Keep both operators projection-style and homogeneous-only.

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

### Task 1: Add Failing Tests for the Structural Aggregation Operators

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing contract tests**

Extend `tests/nn/test_convs.py` with imports for:

```python
from vgl.nn.conv.h2gcn import H2GCNConv
from vgl.nn.conv.egconv import EGConv
```

Add focused tests:

```python
def test_h2gcn_conv_accepts_graph_input():
    conv = H2GCNConv(in_channels=4, out_channels=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_h2gcn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = H2GCNConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_egconv_accepts_graph_input():
    conv = EGConv(in_channels=4, out_channels=3, aggregators=("sum", "mean", "max"))

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_egconv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = EGConv(in_channels=4, out_channels=3, aggregators=("sum", "mean", "max"))

    out = conv(x, edge_index)

    assert out.shape == (3, 3)
```

Extend the heterogeneous rejection parametrization with:

```python
(H2GCNConv, {"in_channels": 4, "out_channels": 3}),
(EGConv, {"in_channels": 4, "out_channels": 3}),
```

**Step 2: Extend the failing integration test**

Modify `tests/integration/test_homo_conv_zoo.py` so the `convs` list also contains:

```python
H2GCNConv(in_channels=4, out_channels=4),
EGConv(in_channels=4, out_channels=4, aggregators=("sum", "mean", "max")),
```

The hidden-size logic should continue resolving correctly.

**Step 3: Extend the failing export assertions**

Update `tests/test_package_exports.py` so the root package also exposes:

```python
from vgl import H2GCNConv, EGConv

assert H2GCNConv.__name__ == "H2GCNConv"
assert EGConv.__name__ == "EGConv"
```

**Step 4: Run the focused tests to verify RED**

Run:

`python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- collection or import failure for missing operator modules or exports
- no unrelated failures

**Step 5: Do not implement production code yet**

No production operator code should exist before the RED failure is observed.

### Task 2: Implement `H2GCNConv` and `EGConv`

**Files:**
- Create: `vgl/nn/conv/h2gcn.py`
- Create: `vgl/nn/conv/egconv.py`
- Modify: `vgl/nn/conv/_homo.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add only the smallest shared helpers if needed**

Add tiny helpers in `_homo.py` only if genuinely needed, such as:

```python
def sum_propagate(x, edge_index):
    row, col = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row])
    return out


def max_propagate(x, edge_index):
    out = torch.full_like(x, float("-inf"))
    row, col = edge_index
    for idx in range(row.numel()):
        out[col[idx]] = torch.maximum(out[col[idx]], x[row[idx]])
    out[out == float("-inf")] = 0
    return out
```

Prefer reusing `mean_propagate(...)` for the mean aggregator.

**Step 2: Implement `H2GCNConv`**

Create `vgl/nn/conv/h2gcn.py`:

```python
import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, mean_propagate


class H2GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels * 3, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "H2GCNConv")
        hop1 = mean_propagate(x, edge_index)
        hop2 = mean_propagate(hop1, edge_index)
        return self.linear(torch.cat([x, hop1, hop2], dim=-1))
```

**Step 3: Implement `EGConv`**

Create `vgl/nn/conv/egconv.py`:

```python
import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, mean_propagate, max_propagate, sum_propagate


class EGConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggregators=("sum", "mean", "max")):
        super().__init__()
        self.aggregators = tuple(aggregators)
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels * len(self.aggregators), out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "EGConv")
        features = []
        for aggregator in self.aggregators:
            if aggregator == "sum":
                features.append(sum_propagate(x, edge_index))
            elif aggregator == "mean":
                features.append(mean_propagate(x, edge_index))
            elif aggregator == "max":
                features.append(max_propagate(x, edge_index))
            else:
                raise ValueError("EGConv only supports sum, mean, and max aggregators")
        return self.linear(torch.cat(features, dim=-1))
```

**Step 4: Export the operators**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

to export the new operators from the package root.

**Step 5: Run the focused tests to verify GREEN**

Run:

`python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- all focused tests pass

**Step 6: Fix only the smallest issues if needed**

Allowed fixes:

- root export omissions
- output shape mismatches
- hetero rejection wording
- aggregator combination bugs

Do not expand features beyond the approved surface.

### Task 3: Extend `conv_zoo` and Minimal Docs

**Files:**
- Modify: `examples/homo/conv_zoo.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`

**Step 1: Extend the example**

Update `examples/homo/conv_zoo.py` imports and `main()` so it also runs:

```python
run_one("h2gcn", H2GCNConv(in_channels=4, out_channels=4), 4)
run_one("egconv", EGConv(in_channels=4, out_channels=4, aggregators=("sum", "mean", "max")), 4)
```

The example should continue printing one list of result dictionaries.

**Step 2: Update docs minimally**

Update:

- `README.md`
- `docs/quickstart.md`
- `docs/core-concepts.md`

to include `H2GCNConv` and `EGConv` in the built-in operator set and homogeneous backbone examples.

**Step 3: Run the example smoke check**

Run:

`python examples/homo/conv_zoo.py`

Expected:

- prints result dictionaries for the expanded operator set
- exits `0`

**Step 4: Fix only import or hidden-size issues if needed**

Do not turn this into a larger documentation or example refactor.

### Task 4: Stop at Minimal Verification and Do Not Commit

**Files:**
- Verify only

**Step 1: Re-run the focused test suite**

Run:

`python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- `PASS`

**Step 2: Re-run `conv_zoo`**

Run:

`python examples/homo/conv_zoo.py`

Expected:

- `PASS`

**Step 3: Record status without committing**

Do not run `git commit` or `git push`.

Report:

- files changed
- focused verification evidence
- that the branch remains uncommitted pending user approval
