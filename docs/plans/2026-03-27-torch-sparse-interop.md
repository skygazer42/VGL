# Torch Sparse Interoperability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-class bridge between VGL `SparseTensor` objects and native PyTorch sparse COO / CSR / CSC tensors.

**Architecture:** Extend `vgl.sparse.convert` with import/export helpers that translate directly between VGL's `SparseTensor` layouts and PyTorch sparse tensor layouts. Keep the batch limited to tensors with exactly two sparse dimensions in COO / CSR / CSC layouts, preserve payload tensors and device placement, and expose the new helpers through `vgl.sparse.__init__` and the stable package-layout test surface.

**Tech Stack:** Python 3.11+, PyTorch, pytest

---

### Task 1: Add failing torch sparse interoperability regressions

**Files:**
- Modify: `tests/sparse/test_sparse_convert.py`
- Modify: `tests/test_package_layout.py`

**Step 1: Write the failing test**

Add regressions proving:

- external PyTorch sparse COO / CSR / CSC tensors import into matching `SparseTensor` layouts
- VGL COO / CSR / CSC tensors export back to matching native sparse layouts
- multi-dimensional sparse payload values survive the bridge
- structure-only VGL sparse tensors export with explicit unit values
- dense tensors and unsupported sparse layouts fail clearly
- `vgl.sparse` exports expose the new helper functions

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_convert.py tests/test_package_layout.py -k torch_sparse`
Expected: FAIL because no torch sparse interoperability helpers exist yet.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/sparse/test_sparse_convert.py tests/test_package_layout.py -k torch_sparse`
Expected: FAIL on missing imports or missing helper functions.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement SparseTensor <-> torch sparse conversion helpers

**Files:**
- Modify: `vgl/sparse/convert.py`
- Modify: `vgl/sparse/__init__.py`

**Step 1: Write the failing test**

Use the regressions from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_convert.py tests/test_package_layout.py -k torch_sparse`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement:

- `from_torch_sparse(...)` for COO / CSR / CSC tensors
- `to_torch_sparse(...)` for VGL COO / CSR / CSC tensors
- unit-value materialization for structure-only export
- clear failures for dense tensors, unsupported layouts, and non-rank-2 inputs
- sparse namespace export wiring consistent with the existing package layout

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/sparse/test_sparse_convert.py tests/test_package_layout.py -k torch_sparse`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Run focused and full verification

**Files:**
- Modify: any touched modules as needed

**Step 1: Focused verification**

Run: `python -m pytest -q tests/sparse/test_sparse_convert.py tests/sparse/test_sparse_base.py tests/sparse/test_sparse_ops.py tests/test_package_layout.py`
Expected: PASS

**Step 2: Full verification**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 3: Prepare integration**

Merge the worktree branch back to `main`, push `origin/main`, and remove the temporary worktree and branch so only `main` remains locally and remotely.

**Step 4: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: add torch sparse interoperability"`
