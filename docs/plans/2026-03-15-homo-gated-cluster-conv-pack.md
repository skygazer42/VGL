# Homogeneous Gated and Cluster Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `ResGatedGraphConv`, `GatedGraphConv`, and `ClusterGCNConv` as stable public homogeneous operators with focused tests, exports, integration coverage, and example updates.

**Architecture:** Extend the flat `vgl.nn.conv` operator surface with one gated projection layer, one recurrent equal-width propagation layer, and one cluster-style projection layer. Reuse `_homo.py` helpers where possible, keep heterogeneous rejection consistent across operators, and avoid refactoring `MessagePassing`.

**Tech Stack:** Python 3.11+, PyTorch, pytest

**Execution Rules:** Use `@test-driven-development` for every implementation task, use `@verification-before-completion` before making success claims, and do not commit because the user has explicitly forbidden commits in this phase.

---

### Task 1: Add Failing Tests for the New Operator Contracts

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

Extend `tests/nn/test_convs.py` with:

- `test_resgated_graph_conv_accepts_graph_input`
- `test_resgated_graph_conv_accepts_x_and_edge_index`
- `test_gated_graph_conv_accepts_graph_input`
- `test_gated_graph_conv_accepts_x_and_edge_index`
- `test_clustergcn_conv_accepts_graph_input`
- `test_clustergcn_conv_accepts_x_and_edge_index`
- hetero rejection coverage for `ResGatedGraphConv`, `GatedGraphConv`, and `ClusterGCNConv`

Update `tests/test_package_exports.py` so the root package asserts:

- `ResGatedGraphConv.__name__ == "ResGatedGraphConv"`
- `GatedGraphConv.__name__ == "GatedGraphConv"`
- `ClusterGCNConv.__name__ == "ClusterGCNConv"`

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/nn/test_convs.py tests/test_package_exports.py -v`

Expected:

- collection or runtime failures because the new operators and exports do not exist yet

**Step 3: Confirm the failure shape**

Re-run the same command if needed until the failures clearly reflect missing operator behavior rather than unrelated syntax or import mistakes.

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 2: Implement `ResGatedGraphConv`, `GatedGraphConv`, and `ClusterGCNConv`

**Files:**
- Create: `vgl/nn/conv/resgated.py`
- Create: `vgl/nn/conv/gatedgraph.py`
- Create: `vgl/nn/conv/clustergcn.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `ResGatedGraphConv` minimally**

Create `vgl/nn/conv/resgated.py` with a layer that:

- accepts `in_channels` and `out_channels`
- supports `forward(graph)` and `forward(x, edge_index)`
- uses homogeneous-only input coercion
- computes a sigmoid gate from source and destination projections
- applies the gate to source value projections
- aggregates to destination nodes
- adds one root-node projection branch

**Step 2: Implement `GatedGraphConv` minimally**

Create `vgl/nn/conv/gatedgraph.py` with a layer that:

- accepts `channels` and `steps`
- supports `forward(graph)` and `forward(x, edge_index)`
- requires `x.size(-1) == channels`
- iteratively applies one learned transform, one `sum` aggregation, and one `GRUCell`

**Step 3: Implement `ClusterGCNConv` minimally**

Create `vgl/nn/conv/clustergcn.py` with a layer that:

- accepts `in_channels`, `out_channels`, and `diag_lambda`
- aggregates neighbors with a compact helper such as `mean_propagate(...)`
- adds one root-node projection scaled by `1 + diag_lambda`
- returns the sum of neighborhood and root branches

**Step 4: Export the layers**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so the new operators are publicly available from the flat surface.

**Step 5: Run tests to verify green**

Run: `python -m pytest tests/nn/test_convs.py tests/test_package_exports.py -v`

Expected:

- the new contract and export tests pass

**Step 6: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 3: Extend the Compact Integration Path

**Files:**
- Modify: `tests/integration/test_homo_conv_zoo.py`

**Step 1: Write the integration extension**

Extend the integration test so the new operator list includes:

- `ResGatedGraphConv(in_channels=4, out_channels=4)`
- `GatedGraphConv(channels=4, steps=2)`
- `ClusterGCNConv(in_channels=4, out_channels=4, diag_lambda=0.0)`

**Step 2: Run the integration test**

Run: `python -m pytest tests/integration/test_homo_conv_zoo.py -v`

Expected:

- pass once the new operators are properly integrated into the example-training path

**Step 3: Make the smallest integration fixes**

Adjust only what the integration test reveals:

- import surface
- hidden size derivation
- operator list wiring

**Step 4: Re-run the integration test**

Run: `python -m pytest tests/integration/test_homo_conv_zoo.py -v`

Expected:

- pass

**Step 5: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 4: Update `conv_zoo` and Minimal Docs

**Files:**
- Modify: `examples/homo/conv_zoo.py`
- Modify: `README.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/core-concepts.md`

**Step 1: Extend the example surface**

Update `examples/homo/conv_zoo.py` so it runs:

- `resgated`
- `gatedgraph`
- `clustergcn`

Use the same tiny training-loop style already present.

**Step 2: Update documentation minimally**

Adjust docs so the built-in operator list now includes:

- `ResGatedGraphConv`
- `GatedGraphConv`
- `ClusterGCNConv`

**Step 3: Run the example smoke**

Run: `python examples/homo/conv_zoo.py`

Expected:

- the script prints result dictionaries that include `resgated`, `gatedgraph`, and `clustergcn`

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 5: Run Focused Verification for the Whole Batch

**Files:**
- Verify only

**Step 1: Run the focused operator and export regression**

Run: `python -m pytest tests/nn/test_convs.py tests/integration/test_homo_conv_zoo.py tests/test_package_exports.py -v`

Expected:

- pass

**Step 2: Run the example smoke**

Run: `python examples/homo/conv_zoo.py`

Expected:

- the full conv zoo runs and prints losses for the expanded operator set

**Step 3: Record evidence and stop**

Report the exact commands and outcomes. Do not claim success without fresh output.

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.
