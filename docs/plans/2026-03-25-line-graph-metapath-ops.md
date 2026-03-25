# Line Graph And Metapath Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add DGL-class `line_graph(...)` and `metapath_reachable_graph(...)` structure transforms to `vgl.ops` and bridge them through `Graph` convenience methods.

**Architecture:** Add one small `vgl.ops.path` module for the new transforms, keep the implementation tensor-first and non-mutating, and export the operations through both `vgl.ops` and `Graph`. `line_graph(...)` stays intentionally single-relation and homogeneous in this batch, while `metapath_reachable_graph(...)` covers heterogeneous and single-node-type multi-relation path composition.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing graph-ops regressions for line graph and metapath reachability

**Files:**
- Create: `tests/ops/test_path_ops.py`

**Step 1: Write the failing test**

Add tests proving:

- `line_graph(...)` turns homogeneous edges into homogeneous line-graph nodes and adjacency
- `line_graph(..., backtracking=False)` removes immediate reverse traversals
- `line_graph(..., copy_edata=True)` copies edge-aligned tensors onto line-graph nodes and preserves original edge ids as `n_id`
- `metapath_reachable_graph(...)` deduplicates reachable endpoints for a heterogeneous metapath
- `metapath_reachable_graph(...)` also works on a single-node-type graph with multiple relations
- invalid metapath chains fail with a clear `ValueError`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_path_ops.py`
Expected: FAIL because the new ops module and exports do not exist.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_path_ops.py`
Expected: FAIL on missing imports or missing functions.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement line_graph and metapath_reachable_graph in vgl.ops

**Files:**
- Create: `vgl/ops/path.py`
- Modify: `vgl/ops/__init__.py`
- Modify: `tests/ops/test_path_ops.py`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_path_ops.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement in `vgl/ops/path.py`:

- `line_graph(graph, *, edge_type=None, backtracking=True, copy_edata=True)`
- `metapath_reachable_graph(graph, metapath, *, relation_name=None)`

Implementation rules:

- resolve the selected `edge_type` with the same conventions used by existing ops
- `line_graph(...)` only accepts relations whose source and destination node types match
- output line-graph nodes must carry `n_id` equal to the original selected relation edge ids
- when `copy_edata=True`, copy every edge-aligned tensor except `edge_index` onto the line-graph node store
- line-graph edges should use `("node", "line", "node")`
- `metapath_reachable_graph(...)` must validate node-type chaining across the full metapath
- the derived relation name should default to `"__".join(rel for _, rel, _ in metapath)`
- reachable endpoint pairs should be deduplicated in stable discovery order
- preserve the original start/end node stores in the returned graph without compaction

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_path_ops.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Bridge Graph methods and stable exports to the new ops layer

**Files:**
- Modify: `vgl/graph/graph.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Extend coverage to prove:

- `graph.line_graph(...)` delegates correctly
- `graph.metapath_reachable_graph(...)` delegates correctly
- `vgl.ops.__all__` and package-layout tests expose the new functions

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/core/test_graph_ops_api.py tests/test_package_layout.py -k "line_graph or metapath or ops_all"`
Expected: FAIL because the `Graph` bridges and exports are incomplete.

**Step 3: Write minimal implementation**

Add `Graph.line_graph(...)` and `Graph.metapath_reachable_graph(...)` convenience methods that delegate into `vgl.ops`, and update `vgl/ops/__init__.py` plus package-layout expectations so the new functions are part of the stable exported surface.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_path_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 4: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for graph-ops sections that still list only the earlier structure/subgraph primitives.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_path_ops.py tests/core/test_graph_ops_api.py tests/test_package_layout.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document:

- `line_graph(...)` as an edge-centric topology transform
- `metapath_reachable_graph(...)` as a heterogeneous path-composition transform
- the new `Graph` convenience methods in quick examples or API summaries

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add line graph and metapath graph ops"`
