# Homogeneous LGConv and Grouped Reversible Residual Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `LGConv` and `GroupRevRes` as stable public homogeneous building blocks with focused tests, exports, integration coverage, and example updates.

**Architecture:** Add `LGConv` as a small width-preserving propagation operator under `vgl.nn.conv`, and add `GroupRevRes` as a grouped reversible wrapper under `vgl.nn`. Reuse the existing homogeneous helper functions, keep wrapper semantics local and explicit, and avoid importing a custom invertible autograd runtime into `vgl`.

**Tech Stack:** Python 3.11+, PyTorch, pytest

**Execution Rules:** Use `@test-driven-development` for every implementation task, use `@verification-before-completion` before making success claims, and do not commit because the user has explicitly forbidden commits in this phase.

---

### Task 1: Add Failing Tests for `LGConv`, `GroupRevRes`, and Root Exports

**Files:**
- Modify: `tests/nn/test_convs.py`
- Create: `tests/nn/test_grouprevres.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

Extend `tests/nn/test_convs.py` with:

```python
from vgl.nn.conv.lg import LGConv
```

Add focused tests:

```python
def test_lgconv_accepts_graph_input():
    conv = LGConv()

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_lgconv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = LGConv()

    out = conv(x, edge_index)

    assert out.shape == (3, 4)


def test_lgconv_without_normalization_matches_sum_propagation():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = LGConv(normalize=False)

    out = conv(x, edge_index)

    assert out.shape == (3, 4)
```

Extend the heterogeneous rejection parametrization with:

```python
(LGConv, {}),
```

Create `tests/nn/test_grouprevres.py` with focused tests for:

- `GroupRevRes` on `Graph`
- `GroupRevRes` on `(x, edge_index)`
- inverse round-trip behavior
- invalid `num_groups`
- non-divisible channel width
- unsupported wrapped forward contract
- hetero rejection

Update `tests/test_package_exports.py` so the root package also asserts:

```python
assert LGConv.__name__ == "LGConv"
assert GroupRevRes.__name__ == "GroupRevRes"
```

**Step 2: Run tests to verify RED**

Run:

`python -m pytest tests/nn/test_convs.py tests/nn/test_grouprevres.py tests/test_package_exports.py -v`

Expected:

- collection or import failures because `LGConv`, `GroupRevRes`, and exports do not exist yet

**Step 3: Confirm the failure shape**

Re-run the same command if needed until the failures clearly reflect missing operator behavior rather than unrelated syntax or import mistakes.

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 2: Implement `LGConv`

**Files:**
- Create: `vgl/nn/conv/lg.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `LGConv` minimally**

Create `vgl/nn/conv/lg.py` with a layer that:

- accepts `normalize=True`
- supports `forward(graph)` and `forward(x, edge_index)`
- reuses `coerce_homo_inputs(...)`
- uses `symmetric_propagate(...)` when `normalize=True`
- uses `sum_propagate(...)` when `normalize=False`
- preserves feature width

Suggested implementation shape:

```python
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, sum_propagate, symmetric_propagate


class LGConv(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "LGConv")
        if self.normalize:
            return symmetric_propagate(x, edge_index)
        return sum_propagate(x, edge_index)
```

**Step 2: Export the layer**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so `LGConv` is publicly available from the flat root surface.

**Step 3: Run focused tests for `LGConv`**

Run:

`python -m pytest tests/nn/test_convs.py -k "lgconv" -v`

Expected:

- `LGConv` tests pass
- unrelated `GroupRevRes` tests may still fail

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 3: Implement `GroupRevRes`

**Files:**
- Create: `vgl/nn/grouprevres.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `GroupRevRes` minimally**

Create `vgl/nn/grouprevres.py` with a wrapper that:

- accepts either:
  - one seed module plus `num_groups`
  - an explicit `nn.ModuleList`
- validates:
  - at least `2` groups
  - feature width divisible by group count
  - the wrapped operator can run on `conv(x, edge_index)` without extra mandatory runtime tensors
- deep-copies the seed module when needed
- supports `forward(graph)` and `forward(x, edge_index)`
- exposes `inverse(graph)` and `inverse(x, edge_index)`

Suggested implementation shape:

```python
class GroupRevRes(nn.Module):
    def __init__(self, conv, num_groups=None, split_dim=-1):
        ...

    def forward(self, graph_or_x, edge_index=None):
        ...

    def inverse(self, graph_or_x, edge_index=None):
        ...
```

The grouped forward should follow the additive reversible recurrence and call each wrapped module with standard homogeneous `(x, edge_index)` inputs only.

**Step 2: Export the wrapper**

Update:

- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so `GroupRevRes` is publicly available from the flat root surface.

**Step 3: Run focused tests for `GroupRevRes`**

Run:

`python -m pytest tests/nn/test_grouprevres.py -v`

Expected:

- `GroupRevRes` tests pass

**Step 4: Fix only the smallest issues if needed**

Allowed fixes:

- divisibility validation
- hetero rejection wording
- inverse recurrence correctness
- wrapped forward signature validation

Do not expand features beyond the approved surface.

**Step 5: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 4: Extend the Compact Integration Path

**Files:**
- Modify: `tests/integration/test_homo_conv_zoo.py`

**Step 1: Extend the integration test**

Add the new building blocks to the exercised surface:

```python
LGConv(),
GroupRevRes(LGConv(), num_groups=2),
```

The hidden-size derivation should continue resolving to `4` for both entries in the current tiny training setup.

**Step 2: Run the integration test**

Run:

`python -m pytest tests/integration/test_homo_conv_zoo.py -v`

Expected:

- pass once the new additions are correctly integrated into the tiny training loop

**Step 3: Make the smallest integration fixes**

Adjust only what the integration test reveals:

- import surface
- hidden size derivation
- training loop wiring

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 5: Update `conv_zoo` and Minimal Docs

**Files:**
- Modify: `examples/homo/conv_zoo.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`

**Step 1: Extend the example surface**

Update `examples/homo/conv_zoo.py` so it runs:

```python
run_one("lgconv", LGConv(), 4)
run_one("grouprevres", GroupRevRes(LGConv(), num_groups=2), 4)
```

**Step 2: Update documentation minimally**

Adjust docs so the built-in surface now includes:

- `LGConv`
- `GroupRevRes`

**Step 3: Run the example smoke**

Run:

`python examples/homo/conv_zoo.py`

Expected:

- the script prints result dictionaries that include `lgconv` and `grouprevres`

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 6: Run Focused Verification for the Whole Batch

**Files:**
- Verify only

**Step 1: Run the focused operator and export regression**

Run:

`python -m pytest tests/nn/test_convs.py tests/nn/test_grouprevres.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- pass

**Step 2: Run the example smoke**

Run:

`python examples/homo/conv_zoo.py`

Expected:

- the full conv zoo runs and prints losses for the expanded surface

**Step 3: Record evidence and stop**

Report the exact commands and outcomes. Do not claim success without fresh output.

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.
