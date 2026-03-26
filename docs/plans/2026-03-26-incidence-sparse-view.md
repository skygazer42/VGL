# Incidence Sparse View Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a DGL-style incidence sparse view so VGL graphs can export `SparseTensor` incidence matrices through `inc(...)`.

**Architecture:** Extend `vgl.ops.query` with one incidence builder that reuses the public `e_id` ordering model, normalizes sparse layouts the same way `Graph.adjacency(...)` does, and builds a COO `SparseTensor` before converting to the requested layout. Bridge the new op through `Graph` and `vgl.ops`, and document the `typestr` semantics plus storage-backed node-count behavior.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing incidence regressions

**Files:**
- Modify: `tests/ops/test_query_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `inc(...)` supports `typestr="in"`, `"out"`, and `"both"` on homogeneous graphs
- `"both"` omits self-loop nonzeros while preserving the edge-column dimension
- incidence columns follow public `e_id` ordering on derived graphs
- bipartite heterogeneous relations reject `typestr="both"`
- invalid `typestr` fails clearly
- featureless storage-backed graphs preserve declared node counts in the row dimension
- `Graph` bridges and `vgl.ops` exports expose the new API

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "inc("`
Expected: FAIL because the new incidence op, Graph method, and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "inc("`
Expected: FAIL on missing imports, missing Graph methods, or missing incidence behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement the incidence sparse view

**Files:**
- Modify: `vgl/ops/query.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_query_ops.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "inc"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/query.py`:

- `inc(graph, typestr="both", *, layout="coo", edge_type=None)`

Implementation rules:

- normalize `layout` from string or `SparseLayout`
- support `typestr` values `"in"`, `"out"`, and `"both"`
- for `"both"`, require a same-type relation
- size row dimensions from `graph._node_count(...)`
- order columns by public `e_id` when present
- represent self-loop `"both"` columns as zero columns by omitting nonzero entries for them
- return a `SparseTensor` in the requested layout

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py -k "inc"`
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

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "inc"`
Expected: FAIL

**Step 3: Write minimal implementation**

Add:

- `Graph.inc(...)`

Update `vgl.ops` exports and namespace expectations so `inc` becomes part of the stable public surface.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "inc"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run the full regression suite

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review graph-query and sparse-layer docs for places that mention adjacency views but not incidence views.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/ops/test_query_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "inc"`
Expected: PASS if the incidence path is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- `typestr="in"|"out"|"both"` semantics
- self-loop behavior for `"both"`
- layout normalization and public-`e_id` column ordering

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add incidence sparse views"`
