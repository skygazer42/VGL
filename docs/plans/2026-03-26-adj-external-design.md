# External Adjacency Export Design

## Context

VGL now exposes three adjacency-oriented surfaces:

- `Graph.adjacency(...)` for internal `SparseTensor` views
- `Graph.adj(...)` for DGL-style weighted adjacency sparse views
- `Graph.adj_tensors(...)` for raw COO / CSR / CSC structural tuples

What is still missing is `adj_external(...)`: a way to export adjacency directly into external sparse formats for interoperability with downstream tooling. DGL exposes this as a lightweight escape hatch that returns either a backend sparse tensor or a SciPy sparse matrix, and it is a natural next step after the internal sparse and raw-tensor layers are already in place.

## Scope Choice

There were two plausible next slices:

1. Add `adj_external(...)` only.
2. Add graph-format management APIs such as `formats()` / `create_formats_()` at the same time.

Option 1 is the right slice. `adj_external(...)` is a pure export boundary with minimal state implications. `formats()` and `create_formats_()` would require graph-level format-state tracking and clone semantics, which is a broader surface.

## Recommended API

Add one public op:

- `adj_external(graph, transpose=False, *, scipy_fmt=None, edge_type=None)`

Add one matching `Graph` method:

- `graph.adj_external(...)`

This intentionally mirrors the DGL name and observable behavior, while omitting `ctx` for now because VGL graphs already carry device-local structure and the immediate goal is parity on CPU and current-device export.

## Semantics

Return modes:

- when `scipy_fmt is None`, return `torch.sparse_coo_tensor`
- when `scipy_fmt == "coo"`, return `scipy.sparse.coo_matrix`
- when `scipy_fmt == "csr"`, return `scipy.sparse.csr_matrix`

Validation:

- reject unsupported `scipy_fmt` values clearly

Shape and orientation:

- default orientation uses source rows and destination columns
- `transpose=True` swaps that orientation
- heterogeneous relations use `(num_src_nodes, num_dst_nodes)` or the transposed shape
- storage-backed declared node counts must carry through to matrix shape

Ordering:

- for torch sparse output, preserve visible COO order from the graph's public `e_id` ordering model
- for SciPy output, `coo` should preserve the same visible order; `csr` can follow SciPy's compressed canonical row ordering

Values:

- nonzero values are ones, matching DGL's observed `adj_external(...)` behavior

## Implementation Approach

The new op can reuse the existing query helpers:

- `_resolve_edge_type(...)`
- `_ordered_edge_tensors(...)`
- graph `_node_count(...)`

Implementation should:

1. resolve the relation
2. fetch ordered endpoints in public-`e_id` order
3. optionally transpose endpoints and shape
4. build either `torch.sparse_coo_tensor` or a SciPy sparse matrix from those endpoints with unit values

This keeps the export path simple and independent from `SparseTensor` caching or graph-level format state.

## Testing Strategy

Extend coverage in:

- `tests/ops/test_query_ops.py`
- `tests/core/test_graph_ops_api.py`
- `tests/core/test_feature_backed_graph.py`
- `tests/test_package_layout.py`

Key regressions:

- default export returns `torch.sparse_coo_tensor`
- `transpose=True` swaps rows / columns and shape
- `scipy_fmt="coo"` and `"csr"` return the right SciPy matrix types
- unsupported `scipy_fmt` fails clearly
- heterogeneous relation selection works
- featureless storage-backed graphs preserve declared shape
- `Graph` bridge and `vgl.ops` export stability

## Non-Goals

This batch should not include:

- graph-level sparse format state tracking
- `formats()` / `create_formats_()`
- CSC SciPy export parity, since observed local DGL rejected `"csc"`
- weighted values or `eweight_name`
- torch compressed sparse export
