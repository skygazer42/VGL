# Release Contract Scans Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add executable scan coverage for built distribution contracts and public Markdown link contracts so packaging regressions fail before a release is published.

**Architecture:** Add two stdlib-first scan runners under `scripts/`: one inspects built wheel/sdist artifacts for metadata and file-layout guarantees, and one scans public Markdown/docs surfaces for broken relative links and stale repository asset URLs. Lock both runners with dedicated pytest coverage and wire them into CI so package and docs regressions are caught by executable checks instead of only prose expectations.

**Tech Stack:** Python 3.10+, `argparse`, `pathlib`, `zipfile`, `tarfile`, `email`, `urllib.parse`, pytest, GitHub Actions

---

### Task 1: Release Artifact Contract Scan

**Files:**
- Create: `tests/test_release_contract_scan.py`
- Create: `scripts/release_contract_scan.py`

**Step 1: Write the failing test**

Add tests that expect:
- `python scripts/release_contract_scan.py --list` to print a stable catalog.
- `python scripts/release_contract_scan.py --artifact-dir <built-dist>` to exit `0` on this repository.
- the script output to include a summary line confirming all artifact checks passed.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_release_contract_scan.py`
Expected: FAIL because `scripts/release_contract_scan.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement a scan runner that:
- locates one wheel and one sdist in the target artifact directory
- inspects wheel metadata for package name, version, Python requirement, extras, and project URLs
- inspects wheel/sdist contents for required files and forbidden repo-only paths
- supports `--list` and executable summary output

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_release_contract_scan.py`
Expected: PASS

### Task 2: Public Docs Link Contract Scan

**Files:**
- Create: `tests/test_docs_link_scan.py`
- Create: `scripts/docs_link_scan.py`

**Step 1: Write the failing test**

Add tests that expect:
- `python scripts/docs_link_scan.py --list` to print a non-empty check list.
- `python scripts/docs_link_scan.py` to exit `0` on the repository root.
- the script to report a passing summary over README and public docs.

**Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_docs_link_scan.py`
Expected: FAIL because `scripts/docs_link_scan.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement a docs scan runner that:
- scans `README.md` plus public `docs/*.md` files excluding `docs/plans`
- validates local Markdown links resolve inside the checkout
- validates same-repo raw asset URLs map to existing files
- ignores unrelated external URLs without network access
- supports `--list` and executable summary output

**Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_docs_link_scan.py`
Expected: PASS

### Task 3: CI Wiring and Aggregate Verification

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_release_packaging.py`

**Step 1: Write the failing assertion**

Extend the packaging/CI test coverage so it expects:
- `package-check` to run `python scripts/release_contract_scan.py --artifact-dir dist`
- CI to run `python scripts/docs_link_scan.py`

**Step 2: Run targeted tests to verify they fail**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_docs_link_scan.py tests/test_release_contract_scan.py`
Expected: FAIL until workflow wiring exists.

**Step 3: Write minimal implementation**

Update CI so:
- `package-check` runs the release artifact contract scan after `python -m build`
- `repo-scan` also runs the docs link scan

**Step 4: Run targeted verification**

Run: `python -m pytest -q tests/test_release_packaging.py tests/test_docs_link_scan.py tests/test_release_contract_scan.py`
Expected: PASS

## Final Verification

Run these commands before merging:

- `python scripts/docs_link_scan.py`
- `python -m build --outdir /tmp/release-contract-dist`
- `python scripts/release_contract_scan.py --artifact-dir /tmp/release-contract-dist`
- `python -m pytest -q tests/test_docs_link_scan.py tests/test_release_contract_scan.py tests/test_release_packaging.py tests/test_full_scan.py tests/test_exec_scan.py`
- `python -m pytest -q`
