# External Adjacency Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-style `adj_external(...)` exports to backend sparse tensors or SciPy sparse matrices.

**Architecture:** Extend `vgl.ops.query` with one external adjacency exporter that resolves relations like the existing query ops, keeps visible COO ordering aligned with public `e_id`, and emits either `torch.sparse_coo_tensor` or SciPy COO / CSR matrices with unit values. Bridge the new op through `Graph` and `vgl.ops`, then document the external sparse interop surface.

**Tech Stack:** Python 3.10+, PyTorch, SciPy, pytest

---

### Task 1: Add failing external-adjacency regressions

**Files:**
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `adj_external(...)` returns `torch.sparse_coo_tensor` by default
- `transpose=True` swaps orientation and shape
- `scipy_fmt="coo"` and `"csr"` return SciPy sparse matrices
- unsupported `scipy_fmt` fails clearly
- heterogeneous graphs work with explicit `edge_type`
- featureless storage-backed graphs preserve declared shape
- `Graph` bridges and `vgl.ops` exports expose the new API

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_external"`
Expected: FAIL because the new op, Graph bridge, and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_external"`
Expected: FAIL on missing imports, missing Graph methods, or missing export behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement external adjacency export

**Files:**
- Modify: `vgl/ops/query.py`
- Modify: `tests/ops/test_query_ops.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "adj_external"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/query.py`:

- `adj_external(graph, transpose=False, *, scipy_fmt=None, edge_type=None)`

Implementation rules:

- resolve the target relation with the same helper as other query ops
- preserve visible COO order via public `e_id`
- default to `torch.sparse_coo_tensor` with unit values
- support SciPy `"coo"` and `"csr"` outputs
- reject unsupported `scipy_fmt` values clearly
- preserve declared node counts in exported matrix shape

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "adj_external"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Bridge Graph methods and stable exports

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_external"`
Expected: FAIL

**Step 3: Write minimal implementation**

Add:

- `Graph.adj_external(...)`

Update `vgl.ops` exports and namespace expectations so `adj_external` becomes part of the stable public surface.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_external"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run the full regression suite

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review adjacency-oriented docs for places that mention internal sparse exports but not external sparse export.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "adj_external"`
Expected: PASS if the external-adjacency path is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- default torch sparse output
- supported SciPy formats
- transpose semantics

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add external adjacency exports"`
