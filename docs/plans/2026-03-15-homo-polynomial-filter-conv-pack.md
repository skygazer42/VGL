# Homogeneous Polynomial Filter Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `MixHopConv` and `BernConv` as stable homogeneous polynomial-filter operators and wire them into the existing export, test, and example surface.

**Architecture:** Add one file per operator under `vgl.nn.conv`, reuse `_homo.py` for tiny propagation helpers, and extend the compact integration path and `conv_zoo` example instead of introducing new training abstractions. Keep `MixHopConv` projection-style and `BernConv` equal-width.

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

### Task 1: Add Failing Tests for the Polynomial Filter Operators

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing contract tests**

Extend `tests/nn/test_convs.py` with imports for:

```python
from vgl.nn.conv.mixhop import MixHopConv
from vgl.nn.conv.bern import BernConv
```

Add focused tests:

```python
def test_mixhop_conv_accepts_graph_input():
    conv = MixHopConv(in_channels=4, out_channels=3, powers=(0, 1, 2))

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_mixhop_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = MixHopConv(in_channels=4, out_channels=3, powers=(0, 1, 2))

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_bern_conv_accepts_graph_input():
    conv = BernConv(channels=4, steps=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_bern_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = BernConv(channels=4, steps=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)
```

Extend the heterogeneous rejection parametrization with:

```python
(MixHopConv, {"in_channels": 4, "out_channels": 3}),
(BernConv, {"channels": 4}),
```

**Step 2: Extend the failing integration test**

Modify `tests/integration/test_homo_conv_zoo.py` so the `convs` list also contains:

```python
MixHopConv(in_channels=4, out_channels=4, powers=(0, 1, 2)),
BernConv(channels=4, steps=3),
```

The hidden-size logic should continue resolving correctly.

**Step 3: Extend the failing export assertions**

Update `tests/test_package_exports.py` so the root package also exposes:

```python
from vgl import MixHopConv, BernConv

assert MixHopConv.__name__ == "MixHopConv"
assert BernConv.__name__ == "BernConv"
```

**Step 4: Run the focused tests to verify RED**

Run:

`python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- collection or import failure for missing operator modules or exports
- no unrelated failures

**Step 5: Do not implement production code yet**

No production operator code should exist before the RED failure is observed.

### Task 2: Implement `MixHopConv` and `BernConv`

**Files:**
- Create: `vgl/nn/conv/mixhop.py`
- Create: `vgl/nn/conv/bern.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Reuse existing propagation helpers**

Prefer reusing `_homo.py::propagate_steps(...)` rather than adding a larger helper surface.

**Step 2: Implement `MixHopConv`**

Create `vgl/nn/conv/mixhop.py`:

```python
import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, propagate_steps


class MixHopConv(nn.Module):
    def __init__(self, in_channels, out_channels, powers=(0, 1, 2)):
        super().__init__()
        self.out_channels = out_channels
        self.powers = tuple(powers)
        self.linear = nn.Linear(in_channels * len(self.powers), out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "MixHopConv")
        propagated = propagate_steps(x, edge_index, max(self.powers))
        features = [propagated[power] for power in self.powers]
        return self.linear(torch.cat(features, dim=-1))
```

**Step 3: Implement `BernConv`**

Create `vgl/nn/conv/bern.py`:

```python
import math

import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, propagate_steps


class BernConv(nn.Module):
    def __init__(self, channels, steps=3):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.steps = steps
        init = torch.tensor(
            [math.comb(steps, k) / (2**steps) for k in range(steps + 1)],
            dtype=torch.float32,
        )
        self.gamma = nn.Parameter(init)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "BernConv")
        outputs = propagate_steps(x, edge_index, self.steps)
        out = torch.zeros_like(x)
        for weight, propagated in zip(self.gamma, outputs):
            out = out + weight * propagated
        return out
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
- polynomial order accumulation bugs

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
run_one("mixhop", MixHopConv(in_channels=4, out_channels=4, powers=(0, 1, 2)), 4)
run_one("bern", BernConv(channels=4, steps=3), 4)
```

The example should continue printing one list of result dictionaries.

**Step 2: Update docs minimally**

Update:

- `README.md`
- `docs/quickstart.md`
- `docs/core-concepts.md`

to include `MixHopConv` and `BernConv` in the built-in operator set and homogeneous backbone examples.

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
