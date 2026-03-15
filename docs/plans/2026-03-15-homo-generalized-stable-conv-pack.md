# Homogeneous Generalized and Stable Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `PNAConv`, `GeneralConv`, and `AntiSymmetricConv` as stable public homogeneous operators with focused tests, exports, integration coverage, and example updates.

**Architecture:** Extend the flat `vgl.nn.conv` operator surface with one degree-aware multi-aggregation layer, one design-space general message-passing layer, and one stable equal-width anti-symmetric propagation layer. Reuse `_homo.py` for tiny helper functions, keep heterogeneous rejection consistent across operators, and avoid refactoring `MessagePassing` or widening the graph contract to edge attributes.

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
from vgl.nn.conv.antisymmetric import AntiSymmetricConv
from vgl.nn.conv.generalconv import GeneralConv
from vgl.nn.conv.pna import PNAConv
```

Add focused tests:

```python
def test_pna_conv_accepts_graph_input():
    conv = PNAConv(
        in_channels=4,
        out_channels=3,
        aggregators=("sum", "mean", "max"),
        scalers=("identity", "amplification", "attenuation"),
    )

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_general_conv_accepts_graph_input():
    conv = GeneralConv(
        in_channels=4,
        out_channels=4,
        aggr="add",
        heads=2,
        attention=True,
    )

    out = conv(_homo_graph())

    assert out.shape == (3, 4)


def test_antisymmetric_conv_accepts_graph_input():
    conv = AntiSymmetricConv(channels=4, num_iters=2, epsilon=0.1, gamma=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 4)
```

Also add:

- `(x, edge_index)` tests for all three
- hetero rejection coverage for all three
- narrow validation tests:
  - unsupported `PNAConv` aggregators
  - unsupported `PNAConv` scalers
  - unsupported `GeneralConv.aggr`
  - invalid `GeneralConv` `heads`
  - unsupported `AntiSymmetricConv.act`

Update `tests/test_package_exports.py` so the root package also asserts:

```python
assert PNAConv.__name__ == "PNAConv"
assert GeneralConv.__name__ == "GeneralConv"
assert AntiSymmetricConv.__name__ == "AntiSymmetricConv"
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

### Task 2: Add Tiny Homogeneous Helpers for Degree-Aware Aggregation

**Files:**
- Modify: `vgl/nn/conv/_homo.py`

**Step 1: Add the smallest helper surface**

Add only tiny helpers genuinely shared by the new operators, for example:

```python
def node_degree(edge_index, num_nodes, dtype, device):
    _, col = edge_index
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.index_add_(0, col, torch.ones(col.numel(), dtype=dtype, device=device))
    return degree


def degree_reference_from_histogram(deg, dtype, device):
    if deg is None:
        return None
    bins = torch.arange(deg.numel(), dtype=dtype, device=device)
    total = deg.to(dtype=dtype, device=device).sum().clamp_min(1.0)
    mean_degree = (bins * deg.to(dtype=dtype, device=device)).sum() / total
    return torch.log1p(mean_degree).clamp_min(1.0)
```

The helper boundary should stay tiny. Do not build a generalized aggregation runtime.

**Step 2: Run a narrow import check**

Run:

`python -m pytest tests/nn/test_convs.py::test_simple_conv_accepts_graph_input -v`

Expected:

- still pass once helpers compile

**Step 3: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 3: Implement `PNAConv`

**Files:**
- Create: `vgl/nn/conv/pna.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `PNAConv` minimally**

Create `vgl/nn/conv/pna.py` with a layer that:

- accepts `in_channels`, `out_channels`, `aggregators`, `scalers`, and `deg`
- supports `forward(graph)` and `forward(x, edge_index)`
- uses homogeneous-only input coercion
- validates supported aggregators and scalers
- computes requested aggregations with existing helpers
- computes destination-node degree
- derives a scaler reference from either:
  - the provided `deg` histogram
  - the current graph degree statistics if `deg is None`
- applies every scaler to every aggregation output
- concatenates the results
- combines them with a root branch and projects to `out_channels`

Suggested implementation shape:

```python
class PNAConv(nn.Module):
    def __init__(...):
        self.aggregators = tuple(aggregators)
        self.scalers = tuple(scalers)
        self.message_linear = nn.Linear(in_channels * len(self.aggregators) * len(self.scalers), out_channels)
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "PNAConv")
        degree = node_degree(...)
        stats = self._degree_reference(...)
        features = []
        for aggregated in self._aggregate_all(...):
            for scaler in self.scalers:
                features.append(self._scale(aggregated, degree, stats, scaler))
        return self.message_linear(torch.cat(features, dim=-1)) + self.root_linear(x)
```

**Step 2: Export the layer**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so `PNAConv` is publicly available from the flat root surface.

**Step 3: Run focused tests for `PNAConv`**

Run:

`python -m pytest tests/nn/test_convs.py -k "pna or hetero or package_exposes" -v`

Expected:

- `PNAConv` tests pass
- unrelated missing operator tests may still fail

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 4: Implement `GeneralConv`

**Files:**
- Create: `vgl/nn/conv/generalconv.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `GeneralConv` minimally**

Create `vgl/nn/conv/generalconv.py` with a layer that:

- accepts `in_channels`, `out_channels`, `aggr`, `skip_linear`, `directed_msg`, `heads`, `attention`, and `l2_normalize`
- validates:
  - `aggr` in `{"add", "mean", "max"}`
  - `heads >= 1`
  - `out_channels % heads == 0`
- supports `forward(graph)` and `forward(x, edge_index)`
- constructs messages from:
  - `x_src`
  - or `torch.cat([x_src, x_dst], dim=-1)` when `directed_msg=True`
- projects messages into `heads * head_channels`
- optionally computes destination-normalized attention with `edge_softmax(...)`
- aggregates to destination nodes using the selected aggregator
- merges heads back to `[num_nodes, out_channels]`
- adds a root branch
- applies optional `l2` normalization

Suggested implementation shape:

```python
class GeneralConv(nn.Module):
    def __init__(...):
        self.head_channels = out_channels // heads
        self.message_linear = nn.Linear(message_in_channels, out_channels)
        self.root_linear = nn.Identity() if not skip_linear else nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GeneralConv")
        row, col = edge_index
        src = x[row]
        dst = x[col]
        msg_in = torch.cat([src, dst], dim=-1) if self.directed_msg else src
        messages = self.message_linear(msg_in).view(x.size(0), ...)
```

For `max`, avoid in-place autograd hazards. Prefer a scatter-style reduction like the fixed `EdgeConv` path.

**Step 2: Export the layer**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so `GeneralConv` is publicly available from the flat root surface.

**Step 3: Run focused tests for `GeneralConv`**

Run:

`python -m pytest tests/nn/test_convs.py -k "general or hetero or package_exposes" -v`

Expected:

- `GeneralConv` tests pass
- unrelated missing operator tests may still fail

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 5: Implement `AntiSymmetricConv`

**Files:**
- Create: `vgl/nn/conv/antisymmetric.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `AntiSymmetricConv` minimally**

Create `vgl/nn/conv/antisymmetric.py` with a layer that:

- accepts `channels`, `num_iters`, `epsilon`, `gamma`, and `act`
- validates supported activations, for example:
  - `"tanh"`
  - `"relu"`
- supports `forward(graph)` and `forward(x, edge_index)`
- computes one graph propagation branch with a stable homogeneous helper such as `symmetric_propagate(...)`
- learns:
  - one square matrix `weight`
  - one bias vector
- forms:

```python
operator = weight - weight.transpose(0, 1) - gamma * eye
```

- repeatedly updates:

```python
state = state + epsilon * act(state @ operator + propagated + bias)
```

for `num_iters` steps

**Step 2: Export the layer**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so `AntiSymmetricConv` is publicly available from the flat root surface.

**Step 3: Run focused tests for `AntiSymmetricConv`**

Run:

`python -m pytest tests/nn/test_convs.py -k "antisymmetric or hetero or package_exposes" -v`

Expected:

- `AntiSymmetricConv` tests pass
- unrelated missing operator tests may still fail

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 6: Extend the Compact Integration Path

**Files:**
- Modify: `tests/integration/test_homo_conv_zoo.py`

**Step 1: Extend the integration test**

Add the new operators to the `convs` list:

```python
PNAConv(
    in_channels=4,
    out_channels=4,
    aggregators=("sum", "mean", "max"),
    scalers=("identity", "amplification", "attenuation"),
)
GeneralConv(
    in_channels=4,
    out_channels=4,
    aggr="add",
    heads=2,
    attention=True,
)
AntiSymmetricConv(channels=4, num_iters=2, epsilon=0.1, gamma=0.1)
```

The hidden-size derivation should continue resolving to:

- `4` for `PNAConv`
- `4` for `GeneralConv`
- `4` for `AntiSymmetricConv`

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

### Task 7: Update `conv_zoo` and Minimal Docs

**Files:**
- Modify: `examples/homo/conv_zoo.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`

**Step 1: Extend the example surface**

Update `examples/homo/conv_zoo.py` so it runs:

```python
run_one(
    "pna",
    PNAConv(
        in_channels=4,
        out_channels=4,
        aggregators=("sum", "mean", "max"),
        scalers=("identity", "amplification", "attenuation"),
    ),
    4,
)
run_one(
    "generalconv",
    GeneralConv(
        in_channels=4,
        out_channels=4,
        aggr="add",
        heads=2,
        attention=True,
    ),
    4,
)
run_one("antisymmetric", AntiSymmetricConv(channels=4, num_iters=2, epsilon=0.1, gamma=0.1), 4)
```

**Step 2: Update documentation minimally**

Adjust docs so the built-in operator list now includes:

- `PNAConv`
- `GeneralConv`
- `AntiSymmetricConv`

**Step 3: Run the example smoke**

Run:

`python examples/homo/conv_zoo.py`

Expected:

- the script prints result dictionaries that include `pna`, `generalconv`, and `antisymmetric`

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 8: Run Focused Verification for the Whole Batch

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
