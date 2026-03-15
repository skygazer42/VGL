# Homogeneous Residual and Basic Convolution Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `GCN2Conv`, `GraphConv`, and `LEConv` as stable public homogeneous operators with focused tests, exports, integration coverage, and example updates.

**Architecture:** Extend the flat `vgl.nn.conv` operator surface with one equal-width residual propagation layer and two projection-style aggregation layers. Reuse `_homo.py` helpers where possible, keep heterogeneous rejection consistent across operators, and avoid refactoring `MessagePassing`.

**Tech Stack:** Python 3.11+, PyTorch, pytest

**Execution Rules:** Use `@test-driven-development` for every implementation task, use `@verification-before-completion` before making success claims, and do not commit because the user has explicitly forbidden commits in this phase.

---

### Task 1: Add Failing Tests for the New Operator Contracts

**Files:**
- Modify: `tests/nn/test_convs.py`
- Modify: `tests/test_package_exports.py`

**Step 1: Write the failing tests**

Extend `tests/nn/test_convs.py` with:

- `test_gcn2_conv_accepts_graph_input_with_x0`
- `test_gcn2_conv_accepts_x_edge_index_and_x0`
- `test_graph_conv_accepts_graph_input`
- `test_graph_conv_accepts_x_and_edge_index`
- `test_leconv_accepts_graph_input`
- `test_leconv_accepts_x_and_edge_index`
- hetero rejection coverage for `GCN2Conv`, `GraphConv`, and `LEConv`

Update `tests/test_package_exports.py` so the root package asserts:

- `GCN2Conv.__name__ == "GCN2Conv"`
- `GraphConv.__name__ == "GraphConv"`
- `LEConv.__name__ == "LEConv"`

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/nn/test_convs.py tests/test_package_exports.py -v`

Expected:

- collection or runtime failures because the new operators and exports do not exist yet

**Step 3: Confirm the failure shape**

Re-run the same command if needed until the failures clearly reflect missing operator behavior rather than unrelated syntax or import mistakes.

**Step 4: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 2: Implement `GCN2Conv`, `GraphConv`, and `LEConv` with Minimal Helper Support

**Files:**
- Create: `vgl/nn/conv/gcn2.py`
- Create: `vgl/nn/conv/graphconv.py`
- Create: `vgl/nn/conv/leconv.py`
- Modify: `vgl/nn/conv/_homo.py`
- Modify: `vgl/nn/conv/__init__.py`
- Modify: `vgl/nn/__init__.py`
- Modify: `vgl/__init__.py`

**Step 1: Implement `GCN2Conv` minimally**

Create `vgl/nn/conv/gcn2.py` with a layer that:

- accepts `channels`, `alpha`, `theta`, `layer`, and `shared_weights`
- supports `forward(graph, x0=...)` and `forward(x, edge_index, x0=...)`
- uses homogeneous-only input coercion
- applies one `symmetric_propagate(...)` step
- mixes propagated state with `x0`
- applies one or two learned linear transforms depending on `shared_weights`
- returns shape `[num_nodes, channels]`

**Step 2: Implement `GraphConv` minimally**

Create `vgl/nn/conv/graphconv.py` with a layer that:

- accepts `in_channels` and `out_channels`
- aggregates neighbors with `sum_propagate(...)`
- applies a neighbor linear branch and a root linear branch
- returns their sum

**Step 3: Implement `LEConv` minimally**

Create `vgl/nn/conv/leconv.py` with a layer that:

- accepts `in_channels` and `out_channels`
- projects node features with small linear branches
- builds edge messages from `src - dst`
- aggregates messages to destination nodes
- adds a self branch

**Step 4: Add only tiny shared helpers if truly needed**

If `GCN2Conv` needs a shared helper for resolving `x0` across the two call styles, add only a small helper in `vgl/nn/conv/_homo.py`.

Do not add a generalized framework for optional auxiliary tensors.

**Step 5: Export the layers**

Update:

- `vgl/nn/conv/__init__.py`
- `vgl/nn/__init__.py`
- `vgl/__init__.py`

so the new operators are publicly available from the flat surface.

**Step 6: Run tests to verify green**

Run: `python -m pytest tests/nn/test_convs.py tests/test_package_exports.py -v`

Expected:

- the new contract and export tests pass

**Step 7: No commit checkpoint**

Do not commit. The user has explicitly forbidden commits.

### Task 3: Extend the Compact Integration Path

**Files:**
- Modify: `tests/integration/test_homo_conv_zoo.py`

**Step 1: Write the failing integration extension**

Extend the integration test so the new operator list includes:

- `GCN2Conv(channels=4, alpha=0.1, theta=1.0, layer=1)`
- `GraphConv(in_channels=4, out_channels=4)`
- `LEConv(in_channels=4, out_channels=4)`

The tiny model wrapper should pass `x0=graph.x` when the operator is `GCN2Conv`.

**Step 2: Run the integration test to verify it fails**

Run: `python -m pytest tests/integration/test_homo_conv_zoo.py -v`

Expected:

- failure because the new operators are not yet integrated into the example-training path

**Step 3: Make the smallest integration fixes**

Adjust only what the integration test reveals:

- import surface
- hidden size derivation
- `GCN2Conv` call path with explicit `x0`

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

- `gcn2`
- `graphconv`
- `leconv`

Use the same tiny training loop style already present. For `gcn2`, pass the initial representation explicitly in the tiny model.

**Step 2: Update documentation minimally**

Adjust docs so the built-in operator list and example list now include:

- `GCN2Conv`
- `GraphConv`
- `LEConv`
- `python examples/homo/conv_zoo.py`

**Step 3: Run the example smoke**

Run: `python examples/homo/conv_zoo.py`

Expected:

- the script prints result dictionaries that include `gcn2`, `graphconv`, and `leconv`

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
