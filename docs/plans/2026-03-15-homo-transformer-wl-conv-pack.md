# Homogeneous Transformer and Weisfeiler-Lehman Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `TransformerConv` and `WLConvContinuous` as stable public homogeneous operators with focused tests, exports, integration coverage, and example updates.

**Architecture:** Extend the flat `vgl.nn.conv` operator surface with one multi-head attention operator and one equal-width continuous WL refinement operator. Reuse existing homogeneous helpers, keep heterogeneous rejection consistent across operators, and avoid widening the graph data contract to edge attributes or tuple inputs.

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
from vgl.nn.conv.transformer import TransformerConv
from vgl.nn.conv.wlconv import WLConvContinuous
```

Add focused tests:

```python
def test_transformer_conv_accepts_graph_input():
    conv = TransformerConv(in_channels=4, out_channels=3, heads=2, concat=False)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_transformer_conv_concat_shape():
    conv = TransformerConv(in_channels=4, out_channels=3, heads=2, concat=True)

    out = conv(_homo_graph())

    assert out.shape == (3, 6)


def test_wlconv_continuous_accepts_graph_input():
    conv = WLConvContinuous()

    out = conv(_homo_graph())

    assert out.shape == (3, 4)
```

Also add:

- `(x, edge_index)` tests for both operators
- `TransformerConv(beta=True, root_weight=True)` path coverage
- hetero rejection coverage for both operators
- narrow validation tests:
  - invalid `TransformerConv` `heads`
  - invalid `TransformerConv` `dropout`

Update `tests/test_package_exports.py` so the root package also asserts:

```python
assert TransformerConv.__name__ == "TransformerConv"
assert WLConvContinuous.__name__ == "WLConvContinuous"
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

### Task 2: Implement `TransformerConv`

**Files:**
- Create: `vgl/nn/conv/transformer.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `TransformerConv` minimally**

Create `vgl/nn/conv/transformer.py` with a layer that:

- accepts `in_channels`, `out_channels`, `heads`, `concat`, `beta`, `dropout`, `bias`, and `root_weight`
- validates:
  - `heads >= 1`
  - `0.0 <= dropout <= 1.0`
- supports `forward(graph)` and `forward(x, edge_index)`
- computes multi-head:
  - destination queries
  - source keys
  - source values
- uses destination-normalized attention with `edge_softmax(...)`
- applies dropout to attention weights
- aggregates values to destination nodes
- returns either concatenated or averaged heads
- adds an optional root branch when `root_weight=True`
- applies beta-gated root/message fusion when `beta=True`

Suggested implementation shape:

```python
class TransformerConv(nn.Module):
    def __init__(...):
        self.query_linear = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.key_linear = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.value_linear = nn.Linear(in_channels, heads * out_channels, bias=bias)
```

The first version should not include `edge_attr` or attention-weight returns.

**Step 2: Export the layer**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so `TransformerConv` is publicly available from the flat root surface.

**Step 3: Run focused tests for `TransformerConv`**

Run:

`python -m pytest tests/nn/test_convs.py -k "transformer or hetero or package_exposes" -v`

Expected:

- `TransformerConv` tests pass
- unrelated missing operator tests may still fail

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 3: Implement `WLConvContinuous`

**Files:**
- Create: `vgl/nn/conv/wlconv.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `WLConvContinuous` minimally**

Create `vgl/nn/conv/wlconv.py` with a layer that:

- takes no required constructor arguments
- supports `forward(graph)` and `forward(x, edge_index)`
- uses homogeneous-only input coercion
- computes one degree-normalized neighborhood summary
- combines it with the root feature in a continuous WL-style update

Suggested implementation shape:

```python
class WLConvContinuous(nn.Module):
    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "WLConvContinuous")
        neigh = mean_propagate(x, edge_index)
        return 0.5 * (x + neigh)
```

Keep the layer parameter-free and equal-width.

**Step 2: Export the layer**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so `WLConvContinuous` is publicly available from the flat root surface.

**Step 3: Run focused tests for `WLConvContinuous`**

Run:

`python -m pytest tests/nn/test_convs.py -k "wlconv or hetero or package_exposes" -v`

Expected:

- `WLConvContinuous` tests pass
- unrelated missing operator tests may still fail

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 4: Extend the Compact Integration Path

**Files:**
- Modify: `tests/integration/test_homo_conv_zoo.py`

**Step 1: Extend the integration test**

Add the new operators to the `convs` list:

```python
TransformerConv(in_channels=4, out_channels=4, heads=2, concat=False, beta=True)
WLConvContinuous()
```

The hidden-size derivation should continue resolving to:

- `4` for `TransformerConv` when `concat=False`
- `4` for `WLConvContinuous`

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
run_one(
    "transformerconv",
    TransformerConv(in_channels=4, out_channels=4, heads=2, concat=False, beta=True),
    4,
)
run_one("wlconv", WLConvContinuous(), 4)
```

**Step 2: Update documentation minimally**

Adjust docs so the built-in operator list now includes:

- `TransformerConv`
- `WLConvContinuous`

**Step 3: Run the example smoke**

Run:

`python examples/homo/conv_zoo.py`

Expected:

- the script prints result dictionaries that include `transformerconv` and `wlconv`

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
