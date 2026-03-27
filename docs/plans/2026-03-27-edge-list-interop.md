# Edge-List Interoperability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class homogeneous edge-list bridge so VGL graphs can import from and export to the simplest common graph representation used across scripts and preprocessing pipelines.

**Architecture:** Follow the existing compat-module pattern used for PyG, DGL, and NetworkX. Add a dedicated `vgl.compat.edgelist` adapter, wire it into `Graph` convenience methods, keep the batch homogeneous-only, and make import preserve explicit node counts even for featureless graphs.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing edge-list compatibility regressions

**Files:**
- Create: `tests/compat/test_edge_list_adapter.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- homogeneous VGL graphs export to ordered `(E, 2)` edge-list tensors
- edge-list import accepts Python pairs and tensor inputs
- explicit `num_nodes` preserves isolated nodes
- transposed `(2, E)` tensor input normalizes correctly
- heterogeneous VGL graphs fail clearly on export
- compat exports expose the new helper functions

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_edge_list_adapter.py tests/test_package_layout.py -k edge_list`
Expected: FAIL because no edge-list compatibility surface exists yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/compat/test_edge_list_adapter.py tests/test_package_layout.py -k edge_list`
Expected: FAIL on missing imports or missing `Graph` methods.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement compat adapter and graph methods

**Files:**
- Create: `vgl/compat/edgelist.py`
- Modify: `vgl/compat/__init__.py`
- Modify: `vgl/graph/graph.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_edge_list_adapter.py tests/test_package_layout.py -k edge_list`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- homogeneous `to_edge_list(graph)` exporting an ordered `(E, 2)` tensor
- `from_edge_list(...)` normalizing Python and tensor inputs into `Graph.homo(...)`
- explicit node-count preservation through imported `n_id` metadata when needed
- `Graph.from_edge_list(...)` and `graph.to_edge_list()`
- compat export wiring consistent with the existing package layout

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/compat/test_edge_list_adapter.py tests/test_package_layout.py -k edge_list`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Run focused and full verification

**Files:**
- Modify: any touched modules as needed

**Step 1: Focused verification**

Run: `python -m pytest -q tests/compat/test_edge_list_adapter.py tests/compat/test_networkx_adapter.py tests/compat/test_dgl_adapter.py tests/compat/test_pyg_adapter.py tests/test_package_layout.py`
Expected: PASS

**Step 2: Full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 3: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains locally and remotely.

**Step 4: Commit**

Run: `git add vgl tests docs && git commit -m "feat: add edge-list graph interoperability"`
