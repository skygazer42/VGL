# Public Surface Scan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add one executable scan for the public import/export surface and example-script conventions so drift in package exports or example entrypoints fails quickly in CI.

**Architecture:** Add a stdlib-first `scripts/public_surface_scan.py` runner that inspects the live checkout using a declarative task catalog. The runner will validate a focused set of root exports, legacy compatibility re-exports, recommended example import paths, and example `__main__` guards. Lock it with dedicated pytest coverage and wire it into the existing `repo-scan` CI job.

**Tech Stack:** Python 3.10+, `argparse`, `importlib`, `pathlib`, `re`, pytest, GitHub Actions

---

### Task 1: Public Surface Scan Tests

**Files:**
- Create: `tests/test_public_surface_scan.py`

**Step 1: Write the failing test**

Add tests that expect:
- `python scripts/public_surface_scan.py --list` to print a stable non-empty scan catalog.
- `python scripts/public_surface_scan.py` to exit `0` on the current repository.
- the scan summary to confirm every public-surface check passed.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_public_surface_scan.py`
Expected: FAIL because `scripts/public_surface_scan.py` does not exist yet.

### Task 2: Public Surface Scan Runner

**Files:**
- Create: `scripts/public_surface_scan.py`

**Step 1: Write minimal implementation**

Implement a runner that verifies:
- selected `vgl` root exports import successfully
- selected legacy packages still re-export the same objects as domain packages
- `examples/` and `tests/integration/` avoid `vgl.core`, `vgl.data`, and `vgl.train` imports
- all `examples/*.py` files contain a `__main__` guard

**Step 2: Run targeted tests**

Run: `python -m pytest -q tests/test_public_surface_scan.py`
Expected: PASS

### Task 3: CI Wiring and Verification

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_full_scan.py`

**Step 1: Write the failing assertion**

Extend CI assertions so they expect `repo-scan` to run `python scripts/public_surface_scan.py`.

**Step 2: Run targeted tests to verify they fail**

Run: `python -m pytest -q tests/test_full_scan.py tests/test_public_surface_scan.py`
Expected: FAIL until workflow wiring exists.

**Step 3: Write minimal implementation**

Update `.github/workflows/ci.yml` so `repo-scan` executes the new scan after `docs_link_scan.py`.

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_full_scan.py tests/test_public_surface_scan.py`
Expected: PASS

## Final Verification

Run these commands before merging:

- `python scripts/public_surface_scan.py`
- `python -m pytest -q tests/test_public_surface_scan.py tests/test_full_scan.py tests/test_package_exports.py tests/test_preferred_import_paths.py tests/test_package_layout.py`
- `python -m ruff check .`
- `python -m pytest -q`
