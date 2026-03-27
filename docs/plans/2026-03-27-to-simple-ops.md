# To Simple Graph Transform Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a DGL-class `to_simple(...)` transform that collapses parallel edges into one visible edge per endpoint pair while preserving graph/node context.

**Architecture:** Extend `vgl.ops.structure` with one additive `to_simple(...)` transform that deduplicates a selected relation in stable first-occurrence order, optionally records multiplicity counts, drops ambiguous `e_id` on the simplified relation, and bridges through `Graph` plus `vgl.ops`.

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

### Task 1: Add failing `to_simple(...)` regressions

**Files:**
- Modify: `tests/ops/test_structure_ops.py`
- Modify: `tests/core/test_graph_ops_api.py`
- Modify: `tests/core/test_feature_backed_graph.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- `to_simple(...)` collapses homogeneous parallel edges in stable first-occurrence order
- `count_attr=` records multiplicity counts
- first representative edge features are preserved on the simplified graph
- heterogeneous relation-local simplification only updates the selected relation
- `Graph.to_simple(...)` forwards to the new transform
- `vgl.ops` exports `to_simple`
- featureless storage-backed graphs preserve declared node-space shape after simplification

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_structure_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "to_simple"`
Expected: FAIL because the transform, Graph bridge, and exports do not exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_structure_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "to_simple"`
Expected: FAIL on missing imports, missing Graph methods, or missing deduplication behavior.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement the `to_simple(...)` transform

**Files:**
- Modify: `vgl/ops/structure.py`
- Modify: `vgl/graph/graph.py`
- Modify: `vgl/ops/__init__.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_structure_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "to_simple"`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `to_simple(graph, *, edge_type=None, count_attr=None)` in `vgl.ops.structure`
- `Graph.to_simple(...)`
- `vgl.ops.to_simple`

Implementation rules:

- deduplicate the selected relation by endpoint pair in stable first-occurrence order
- preserve node stores and graph/storage context
- preserve edge-aligned tensor features from the first visible edge for each pair
- drop `e_id` from the simplified relation
- add multiplicity counts when `count_attr` is provided
- keep unrelated relations unchanged on heterogeneous graphs

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_structure_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py -k "to_simple"`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run branch verification

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review graph-ops docs for places that enumerate structure transforms but do not mention simple-graph deduplication.

**Step 2: Run focused verification**

Run: `python -m pytest -q tests/ops/test_structure_ops.py tests/core/test_graph_ops_api.py tests/core/test_feature_backed_graph.py tests/test_package_layout.py`
Expected: PASS if the transform is implemented cleanly.

**Step 3: Write minimal implementation**

Document:

- `to_simple(...)` as a non-mutating edge deduplication transform
- stable first-occurrence semantics
- optional multiplicity counts through `count_attr=`
- dropped `e_id` behavior on simplified relations

**Step 4: Run full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add to_simple graph transform"`
