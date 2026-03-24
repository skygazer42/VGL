# Hetero Graph Ops Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the existing `vgl.ops` structure operators so heterogeneous relation slices work through the current public API instead of failing with homogeneous-only guards.

**Architecture:** Keep the current operator names and expand the implementation in `vgl.ops.subgraph` and `vgl.ops.compact`. Heterogeneous behavior stays relation-local: each op works against one selected `edge_type`, reuses the existing `Graph.hetero(...)` constructor, and preserves the current homogeneous behavior unchanged.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add hetero node_subgraph regression coverage

**Files:**
- Modify: `tests/ops/test_subgraph_ops.py`

**Step 1: Write the failing test**

Add tests showing that `node_subgraph()` accepts hetero node selections keyed by node type, filters edges on the selected relation, relabels source and destination spaces independently, and preserves node/edge features for the selected types.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py -k hetero`
Expected: FAIL because hetero node subgraph is not implemented.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py -k hetero`
Expected: FAIL with the hetero support error path.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement hetero node_subgraph

**Files:**
- Modify: `vgl/ops/subgraph.py`
- Modify: `tests/ops/test_subgraph_ops.py`

**Step 1: Write the failing test**

Use the test from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py -k hetero`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement relation-local hetero node subgraph support. Require `node_ids` to be a dict keyed by the participating source/destination node types, filter selected edges, relabel source/destination ids independently, slice node features for the participating node types, and return `Graph.hetero(...)`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py -k hetero`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/ops/test_subgraph_ops.py vgl/ops/subgraph.py && git commit -m "feat: support hetero node subgraphs"`

### Task 3: Add and implement hetero edge_subgraph coverage

**Files:**
- Modify: `tests/ops/test_subgraph_ops.py`
- Modify: `vgl/ops/subgraph.py`

**Step 1: Write the failing test**

Add tests showing that `edge_subgraph()` on a hetero relation preserves the original node spaces for the participating types, slices edge features by edge id, and returns a hetero graph with the selected relation.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py -k edge_subgraph`
Expected: FAIL because hetero edge subgraph is not implemented.

**Step 3: Write minimal implementation**

Implement relation-local hetero edge subgraph support using the existing edge slicing logic and original node stores for the participating node types.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py -k edge_subgraph`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/ops/test_subgraph_ops.py vgl/ops/subgraph.py && git commit -m "feat: support hetero edge subgraphs"`

### Task 4: Add hetero compact_nodes regression coverage

**Files:**
- Modify: `tests/ops/test_compact_ops.py`

**Step 1: Write the failing test**

Add tests showing that `compact_nodes()` accepts hetero node ids keyed by node type for a selected relation, relabels source/destination spaces independently, slices node features, preserves edge features, and returns per-type mappings.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_compact_ops.py -k hetero`
Expected: FAIL because hetero compacting is not implemented.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/ops/test_compact_ops.py -k hetero`
Expected: FAIL with the homogeneous-only guard.

**Step 5: Commit**

Do not commit yet.

### Task 5: Implement hetero compact_nodes

**Files:**
- Modify: `vgl/ops/compact.py`
- Modify: `tests/ops/test_compact_ops.py`

**Step 1: Write the failing test**

Use the test from Task 4.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_compact_ops.py -k hetero`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement hetero compaction for a selected relation using per-type ordered unique ids, independent relabel maps for source and destination types, sliced node features, preserved edge features, and a mapping payload shaped like `{node_type: {old_id: new_id}}`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_compact_ops.py -k hetero`
Expected: PASS

**Step 5: Commit**

Run: `git add tests/ops/test_compact_ops.py vgl/ops/compact.py && git commit -m "feat: support hetero compact nodes"`

### Task 6: Refresh docs for hetero structure ops

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review docs for places that currently imply structure ops are only homogeneous.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py tests/ops/test_compact_ops.py`
Expected: PASS; documentation gap is manual.

**Step 3: Write minimal implementation**

Document that the existing structure operators now support relation-local hetero slices when `edge_type` is provided.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/ops/test_subgraph_ops.py tests/ops/test_compact_ops.py`
Expected: PASS

**Step 5: Commit**

Run: `git add README.md docs/core-concepts.md docs/quickstart.md && git commit -m "docs: describe hetero graph ops support"`

### Task 7: Run full regression for the hetero ops branch

**Files:**
- Modify: any touched modules as needed

**Step 1: Write the failing test**

No new test file. Use the full repository suite as the branch gate.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q`
Expected: PASS if previous tasks were implemented cleanly; otherwise use failures to close remaining integration gaps.

**Step 3: Write minimal implementation**

Fix only the regressions exposed by the full suite.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: expand hetero graph ops"`
