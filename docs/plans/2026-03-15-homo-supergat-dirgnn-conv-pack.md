# Homogeneous Supervised Attention and Directed Wrapper Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `SuperGATConv` and `DirGNNConv` as stable public homogeneous operators with focused tests, exports, integration coverage, and example updates.

**Architecture:** Extend the flat `vgl.nn.conv` operator surface with one self-supervised attention operator and one directed wrapper operator. Reuse existing homogeneous helpers, keep heterogeneous rejection consistent across operators, and avoid widening the graph data contract to edge attributes, tuple inputs, or special training hooks in `Trainer`.

**Tech Stack:** Python 3.11+, PyTorch, pytest

**Execution Rules:** Use `@test-driven-development` for every implementation task, use `@verification-before-completion` before making success claims, and do not commit because the user has explicitly forbidden commits in this phase.

---

### Task 1: Add Failing Tests for the New Operator Contracts

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

Extend `tests/nn/test_convs.py` with imports for:

```python
from vgl.nn.conv.dirgnn import DirGNNConv
from vgl.nn.conv.supergat import SuperGATConv
```

Add focused tests:

```python
def test_supergat_conv_accepts_graph_input():
    conv = SuperGATConv(in_channels=4, out_channels=3, heads=2, concat=False)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_supergat_conv_concat_shape():
    conv = SuperGATConv(in_channels=4, out_channels=3, heads=2, concat=True)

    out = conv(_homo_graph())

    assert out.shape == (3, 6)


def test_dirgnn_conv_accepts_graph_input():
    conv = DirGNNConv(GraphConv(in_channels=4, out_channels=3), alpha=0.5, root_weight=True)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)
```

Also add:

- `(x, edge_index)` tests for both operators
- `SuperGATConv.get_attention_loss()` scalar coverage
- `DirGNNConv` alpha-mixing path coverage
- hetero rejection coverage for both operators
- narrow validation tests:
  - invalid `SuperGATConv` `heads`
  - invalid `SuperGATConv` `dropout`
  - invalid `SuperGATConv` `attention_type`
  - invalid `DirGNNConv` `alpha`
  - invalid wrapped base operator contract

Update `tests/test_package_exports.py` so the root package also asserts:

```python
assert SuperGATConv.__name__ == "SuperGATConv"
assert DirGNNConv.__name__ == "DirGNNConv"
```

**Step 2: Run tests to verify RED**

Run:

`python -m pytest tests/nn/test_convs.py tests/test_package_exports.py -v`

Expected:

- collection or runtime failures because the new operators and exports do not exist yet

**Step 3: Confirm the failure shape**

Re-run the same command if needed until the failures clearly reflect missing operator behavior rather than unrelated syntax or import mistakes.

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 2: Implement `SuperGATConv`

**Files:**
- Create: `vgl/nn/conv/supergat.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `SuperGATConv` minimally**

Create `vgl/nn/conv/supergat.py` with a layer that:

- accepts `in_channels`, `out_channels`, `heads`, `concat`, `negative_slope`, `dropout`, `add_self_loops`, `bias`, and `attention_type`
- validates:
  - `heads >= 1`
  - `0.0 <= dropout <= 1.0`
  - `attention_type` in `{"MX", "SD"}`
- supports `forward(graph)` and `forward(x, edge_index)`
- optionally adds self-loops
- computes one linear projection shared across heads
- computes destination-normalized attention
- caches a compact self-supervised attention objective
- exposes `get_attention_loss()` as a scalar tensor

Suggested implementation shape:

```python
class SuperGATConv(nn.Module):
    def __init__(...):
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
```

The first version should not expose negative-edge or batch inputs.

**Step 2: Export the layer**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so `SuperGATConv` is publicly available from the flat root surface.

**Step 3: Run focused tests for `SuperGATConv`**

Run:

`python -m pytest tests/nn/test_convs.py -k "supergat or hetero or package_exposes" -v`

Expected:

- `SuperGATConv` tests pass
- unrelated missing operator tests may still fail

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 3: Implement `DirGNNConv`

**Files:**
- Create: `vgl/nn/conv/dirgnn.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `DirGNNConv` minimally**

Create `vgl/nn/conv/dirgnn.py` with a wrapper that:

- accepts `conv`, `alpha`, and `root_weight`
- validates:
  - `0.0 <= alpha <= 1.0`
  - the wrapped base operator can run on `conv(x, edge_index)` without extra mandatory runtime tensors
  - the wrapped base operator exposes a stable output width via `out_channels` or `channels`
- deep-copies the wrapped operator into:
  - `conv_in`
  - `conv_out`
- disables `add_self_loops` or `root_weight` on the wrapped branches when those attributes exist
- runs:
  - `conv_in(x, edge_index)`
  - `conv_out(x, edge_index.flip(0))`
- mixes them with `alpha`
- optionally adds a learned root transform

Suggested implementation shape:

```python
class DirGNNConv(nn.Module):
    def __init__(self, conv, alpha=0.5, root_weight=True):
        self.conv_in = copy.deepcopy(conv)
        self.conv_out = copy.deepcopy(conv)
```

**Step 2: Export the layer**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so `DirGNNConv` is publicly available from the flat root surface.

**Step 3: Run focused tests for `DirGNNConv`**

Run:

`python -m pytest tests/nn/test_convs.py -k "dirgnn or hetero or package_exposes" -v`

Expected:

- `DirGNNConv` tests pass
- unrelated missing operator tests may still fail

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 4: Extend the Compact Integration Path

**Files:**
- Modify: `tests/integration/test_homo_conv_zoo.py`

**Step 1: Extend the integration test**

Add the new operators to the `convs` list:

```python
SuperGATConv(in_channels=4, out_channels=4, heads=2, concat=False)
DirGNNConv(GraphConv(in_channels=4, out_channels=4), alpha=0.5, root_weight=True)
```

The hidden-size derivation should continue resolving to:

- `4` for `SuperGATConv` when `concat=False`
- `4` for `DirGNNConv`

**Step 2: Run the integration test**

Run:

`python -m pytest tests/integration/test_homo_conv_zoo.py -v`

Expected:

- pass once the new operators are correctly integrated into the tiny training loop

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
run_one("supergat", SuperGATConv(in_channels=4, out_channels=4, heads=2, concat=False), 4)
run_one("dirgnn", DirGNNConv(GraphConv(in_channels=4, out_channels=4), alpha=0.5, root_weight=True), 4)
```

**Step 2: Update documentation minimally**

Adjust docs so the built-in operator list now includes:

- `SuperGATConv`
- `DirGNNConv`

**Step 3: Run the example smoke**

Run:

`python examples/homo/conv_zoo.py`

Expected:

- the script prints result dictionaries that include `supergat` and `dirgnn`

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 6: Run Focused Verification for the Whole Batch

**Files:**
- Verify only

**Step 1: Run the focused operator and export regression**

Run:

`python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- pass

**Step 2: Run the example smoke**

Run:

`python examples/homo/conv_zoo.py`

Expected:

- the full conv zoo runs and prints losses for the expanded operator set

**Step 3: Record evidence and stop**

Report the exact commands and outcomes. Do not claim success without fresh output.

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.
