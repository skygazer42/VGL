# Homogeneous Recursive Propagation Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `ARMAConv` and `GPRGNNConv` as stable homogeneous recursive propagation operators and wire them into the existing export, test, and example surface.

**Architecture:** Add one file per operator under `vgl.nn.conv`, reuse `_homo.py` for tiny propagation helpers, and extend the compact integration path and `conv_zoo` example instead of introducing new training abstractions. Keep these operators equal-width and propagation-oriented, leaving feature projection to user models.

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

### Task 1: Add Failing Tests for the Recursive Propagation Operators

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/integration/test_homo_conv_zoo.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing contract tests**

Extend `tests/nn/test_convs.py` with imports for:

```python
from vgl.nn.conv.arma import ARMAConv
from vgl.nn.conv.gprgnn import GPRGNNConv
```

Add focused tests:

```python
def test_arma_conv_accepts_graph_input():
    conv = ARMAConv(channels=4, stacks=2, layers=2, alpha=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_arma_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = ARMAConv(channels=4, stacks=2, layers=2, alpha=0.1)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_gprgnn_conv_accepts_graph_input():
    conv = GPRGNNConv(channels=4, steps=3, alpha=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_gprgnn_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GPRGNNConv(channels=4, steps=3, alpha=0.1)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)
```

Extend the heterogeneous rejection parametrization with:

```python
(ARMAConv, {"channels": 4}),
(GPRGNNConv, {"channels": 4}),
```

**Step 2: Extend the failing integration test**

Modify `tests/integration/test_homo_conv_zoo.py` so the `convs` list also contains:

```python
ARMAConv(channels=4, stacks=2, layers=2, alpha=0.1),
GPRGNNConv(channels=4, steps=3, alpha=0.1),
```

The hidden-size logic should continue resolving to `4`.

**Step 3: Extend the failing export assertions**

Update `tests/test_package_exports.py` so the root package also exposes:

```python
from vgl import ARMAConv, GPRGNNConv

assert ARMAConv.__name__ == "ARMAConv"
assert GPRGNNConv.__name__ == "GPRGNNConv"
```

**Step 4: Run the focused tests to verify RED**

Run:

`python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- collection or import failure for missing operator modules or exports
- no unrelated failures

**Step 5: Do not implement production code yet**

No production operator code should exist before the RED failure is observed.

### Task 2: Implement `ARMAConv` and `GPRGNNConv`

**Files:**
- Create: `vgl/nn/conv/arma.py`
- Create: `vgl/nn/conv/gprgnn.py`
- Modify: `vgl/nn/conv/_homo.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Add minimal shared helpers**

Extend `_homo.py` only if needed. Acceptable helper shapes include:

```python
def propagate_steps(x, edge_index, steps):
    outputs = [x]
    current = x
    for _ in range(steps):
        current = symmetric_propagate(current, edge_index)
        outputs.append(current)
    return outputs
```

Keep helpers tiny and internal.

**Step 2: Implement `ARMAConv`**

Create `vgl/nn/conv/arma.py`:

```python
import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, symmetric_propagate


class ARMAConv(nn.Module):
    def __init__(self, channels, stacks=1, layers=2, alpha=0.1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.stacks = stacks
        self.layers = layers
        self.alpha = alpha
        self.stack_weights = nn.Parameter(torch.ones(stacks))

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "ARMAConv")
        stack_outputs = []
        for _ in range(self.stacks):
            current = x
            for _ in range(self.layers):
                propagated = symmetric_propagate(current, edge_index)
                current = (1 - self.alpha) * propagated + self.alpha * x
            stack_outputs.append(current)
        weights = torch.softmax(self.stack_weights, dim=0)
        out = torch.zeros_like(x)
        for weight, stack_out in zip(weights, stack_outputs):
            out = out + weight * stack_out
        return out
```

**Step 3: Implement `GPRGNNConv`**

Create `vgl/nn/conv/gprgnn.py`:

```python
import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, propagate_steps


class GPRGNNConv(nn.Module):
    def __init__(self, channels, steps=10, alpha=0.1):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.steps = steps
        self.alpha = alpha
        init = torch.tensor(
            [alpha * (1 - alpha) ** k for k in range(steps)] + [(1 - alpha) ** steps],
            dtype=torch.float32,
        )
        self.gamma = nn.Parameter(init)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GPRGNNConv")
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
- propagation accumulation bugs

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
run_one("arma", ARMAConv(channels=4, stacks=2, layers=2, alpha=0.1), 4)
run_one("gprgnn", GPRGNNConv(channels=4, steps=3, alpha=0.1), 4)
```

The example should continue printing one list of result dictionaries.

**Step 2: Update docs minimally**

Update:

- `README.md`
- `docs/quickstart.md`
- `docs/core-concepts.md`

to include `ARMAConv` and `GPRGNNConv` in the built-in operator set and homogeneous backbone examples.

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
