# Sparse Multi-Value SPMM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `vgl.sparse.spmm(...)` so it supports sparse tensors whose edge values carry trailing payload dimensions.

**Architecture:** Keep the public `spmm(sparse, dense)` API unchanged, preserve current scalar behavior, and generalize only the internal broadcasting/accumulation logic so sparse payload dimensions survive into the output shape. Update docs to reflect that `spmm(...)` now supports multi-value sparse payloads.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add failing sparse regressions for multi-value spmm

**Files:**
- Modify: `tests/sparse/test_sparse_ops.py`

**Step 1: Write the failing test**

Add tests proving:

- scalar `spmm(...)` behavior still matches the current output
- sparse values shaped `(nnz, heads)` produce `(rows, heads, features)` output
- sparse values shaped `(nnz, heads, channels)` produce `(rows, heads, channels, features)` output
- empty sparse tensors with multi-value payloads return correctly shaped zeros
- compressed sparse layouts also work for multi-value `spmm(...)`

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py`
Expected: FAIL because `spmm(...)` currently rejects multi-dimensional sparse values.

**Step 3: Write minimal implementation**

No implementation yet in this task.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py`
Expected: FAIL on the current scalar-only `spmm(...)` validation.

**Step 5: Commit**

Do not commit yet.

### Task 2: Implement multi-value spmm support

**Files:**
- Modify: `vgl/sparse/ops.py`

**Step 1: Write the failing test**

Use the tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `spmm(...)` so it:

- keeps scalar and valueless behavior unchanged
- broadcasts dense row features across sparse payload dimensions
- returns shape `(rows, *payload_dims, features)` for multi-value sparse tensors
- works for COO, CSR, and CSC inputs through `to_coo()`
- preserves the empty-sparse fast path with the correct output shape

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py`
Expected: PASS

**Step 5: Commit**

Do not commit yet.

### Task 3: Refresh docs and run full regression

**Files:**
- Modify: `README.md`
- Modify: `docs/core-concepts.md`
- Modify: `docs/quickstart.md`

**Step 1: Write the failing test**

No code test. Review sparse backend docs for scalar-only `spmm(...)` wording.

**Step 2: Run test to verify the code path state**

Run: `python -m pytest -q tests/sparse/test_sparse_ops.py`
Expected: PASS

**Step 3: Write minimal implementation**

Document that `spmm(...)` now supports sparse edge payloads with trailing dimensions and appends dense feature channels at the end of the output shape.

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q`
Expected: PASS across the repository.

**Step 5: Commit**

Run: `git add vgl tests docs README.md && git commit -m "feat: support multi-value sparse spmm"`
