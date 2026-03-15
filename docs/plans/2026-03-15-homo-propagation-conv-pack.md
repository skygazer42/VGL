# Homogeneous Propagation Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `AGNNConv`, `LightGCNConv`, and `FAGCNConv` as stable homogeneous propagation operators and wire them into the existing export, test, and example surface.

**Architecture:** Add one file per operator under `vgl.nn.conv`, reuse `_homo.py` for tiny propagation helpers, and extend the compact integration path and `conv_zoo` example instead of creating new training abstractions. Keep these operators equal-width and propagation-oriented, leaving feature projection to user models.

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

### Task 1: Add Failing Tests for the New Propagation Operators

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing contract tests**

Extend `tests/nn/test_convs.py` with imports for:

```python
from vgl.nn.conv.agnn import AGNNConv
from vgl.nn.conv.lightgcn import LightGCNConv
from vgl.nn.conv.fagcn import FAGCNConv
```

Add focused tests:

```python
def test_agnn_conv_accepts_graph_input():
    conv = AGNNConv(channels=4)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_agnn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = AGNNConv(channels=4)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_lightgcn_conv_accepts_graph_input():
    conv = LightGCNConv()

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_lightgcn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = LightGCNConv()

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_fagcn_conv_accepts_graph_input():
    conv = FAGCNConv(channels=4, eps=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)
```

Extend the heterogeneous rejection parametrization with:

```python
(AGNNConv, {"channels": 4}),
(LightGCNConv, {}),
(FAGCNConv, {"channels": 4}),
```

**Step 2: Extend the failing integration test**

Modify `tests/integration/test_homo_conv_zoo.py` so the `convs` list also contains:

```python
AGNNConv(channels=4),
LightGCNConv(),
FAGCNConv(channels=4, eps=0.1),
```

The hidden-size logic should still resolve to `4` for these equal-width operators.

**Step 3: Extend the failing export assertions**

Update `tests/test_package_exports.py` so the root package also exposes:

```python
from vgl import AGNNConv, LightGCNConv, FAGCNConv

assert AGNNConv.__name__ == "AGNNConv"
assert LightGCNConv.__name__ == "LightGCNConv"
assert FAGCNConv.__name__ == "FAGCNConv"
```

**Step 4: Run the focused tests to verify RED**

Run:

`python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- collection or import failure for missing operator modules or exports
- no unrelated failures

**Step 5: Do not implement production code yet**

No production operator code should exist before the RED failure is observed.

### Task 2: Implement `AGNNConv`, `LightGCNConv`, and `FAGCNConv`

**Files:**
- Create: `vgl/nn/conv/agnn.py`
- Create: `vgl/nn/conv/lightgcn.py`
- Create: `vgl/nn/conv/fagcn.py`
- Modify: `vgl/nn/conv/_homo.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add minimal shared helpers**

Extend `_homo.py` with only the helpers needed for this batch. Valid helper shapes include:

```python
def symmetric_propagate(x, edge_index):
    row, col = edge_index
    src_degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
    dst_degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
    ones = torch.ones(col.size(0), dtype=x.dtype, device=x.device)
    src_degree.index_add_(0, row, ones)
    dst_degree.index_add_(0, col, ones)
    norm = (src_degree[row].clamp_min(1).pow(-0.5) * dst_degree[col].clamp_min(1).pow(-0.5)).unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row] * norm)
    return out


def edge_softmax(scores, edge_index, num_nodes):
    _, col = edge_index
    weights = torch.exp(scores - scores.max())
    normalizer = torch.zeros(num_nodes, dtype=weights.dtype, device=weights.device)
    normalizer.index_add_(0, col, weights)
    return weights / normalizer[col].clamp_min(1e-12)
```

Keep helpers tiny and internal.

**Step 2: Implement `AGNNConv`**

Create `vgl/nn/conv/agnn.py`:

```python
import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


class AGNNConv(nn.Module):
    def __init__(self, channels, beta=1.0, train_beta=False):
        super().__init__()
        self.out_channels = channels
        if train_beta:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "AGNNConv")
        row, col = edge_index
        normalized = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = self.beta * (normalized[row] * normalized[col]).sum(dim=-1)
        weights = edge_softmax(scores, edge_index, x.size(0))
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * weights.unsqueeze(-1))
        return out
```

**Step 3: Implement `LightGCNConv`**

Create `vgl/nn/conv/lightgcn.py`:

```python
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, symmetric_propagate


class LightGCNConv(nn.Module):
    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "LightGCNConv")
        return symmetric_propagate(x, edge_index)
```

**Step 4: Implement `FAGCNConv`**

Create `vgl/nn/conv/fagcn.py`:

```python
import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


class FAGCNConv(nn.Module):
    def __init__(self, channels, eps=0.1):
        super().__init__()
        self.out_channels = channels
        self.eps = eps
        self.gate = nn.Linear(channels * 2, 1)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "FAGCNConv")
        row, col = edge_index
        pair = torch.cat([x[row], x[col]], dim=-1)
        scores = torch.tanh(self.gate(pair)).squeeze(-1)
        weights = edge_softmax(scores, edge_index, x.size(0))
        propagated = torch.zeros_like(x)
        propagated.index_add_(0, col, x[row] * weights.unsqueeze(-1))
        return (1 - self.eps) * propagated + self.eps * x
```

**Step 5: Export the operators**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

to export the new operators from the package root.

**Step 6: Run the focused tests to verify GREEN**

Run:

`python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- all focused tests pass

**Step 7: Fix only the smallest issues if needed**

Allowed fixes:

- root export omissions
- output shape mismatches
- hetero rejection wording
- edge normalization bugs

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
run_one("agnn", AGNNConv(channels=4), 4)
run_one("lightgcn", LightGCNConv(), 4)
run_one("fagcn", FAGCNConv(channels=4, eps=0.1), 4)
```

The example should continue printing one list of result dictionaries.

**Step 2: Update docs minimally**

Update:

- `README.md`
- `docs/quickstart.md`
- `docs/core-concepts.md`

to include `AGNNConv`, `LightGCNConv`, and `FAGCNConv` in the built-in operator set and homogeneous backbone examples.

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
