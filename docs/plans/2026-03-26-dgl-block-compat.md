# DGL Block Compatibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a compatibility bridge between VGL `Block` objects and DGL blocks without changing the existing `Graph` adapter return types.

**Architecture:** Introduce dedicated DGL block import/export helpers plus `Block` convenience methods, and keep the implementation limited to single-relation DGL blocks so it maps cleanly onto VGL's existing relation-local `Block` abstraction.

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: Add failing DGL block adapter regressions

**Files:**
- Modify: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing test**

Add regressions proving:

- homogeneous VGL `Block` exports to a DGL block and imports back with `src_n_id`, `dst_n_id`, `e_id`, and feature tensors preserved
- relation-local heterogeneous VGL `Block` exports to a DGL block and imports back with typed endpoints preserved
- an external single-relation DGL block imports to a VGL `Block`
- multi-relation DGL blocks fail clearly on import

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k block`
Expected: FAIL because `Block` and the DGL compatibility layer do not yet expose block import/export helpers.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k block`
Expected: FAIL on missing block adapter methods or helpers.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement single-relation DGL block import/export

**Files:**
- Modify: `vgl/compat/dgl.py`
- Modify: `vgl/compat/__init__.py`
- Modify: `vgl/graph/block.py`
- Modify: `tests/compat/test_dgl_adapter.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k block`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `block_to_dgl(...)` using `dgl.create_block(...)`
- `block_from_dgl(...)` for single-relation DGL blocks
- `Block.to_dgl()` and `Block.from_dgl(...)` convenience methods
- preservation of block node/edge features and `n_id` / `e_id` metadata
- clear failure on multi-relation DGL block import

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py -k block`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review DGL compatibility docs for places that should mention `Block` round-trips.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/compat/test_dgl_adapter.py`
Expected: PASS if the block adapter implementation is complete.

**Step 3: Write minimal implementation**

Document that DGL compatibility now covers `Block` import/export for single-relation message-flow blocks in addition to graph round-trips.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add dgl block compatibility"`
