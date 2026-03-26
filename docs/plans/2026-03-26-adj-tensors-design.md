# Adjacency Tensor Export Design

## Context

VGL now has two adjacent capabilities in place:

- sparse adjacency views through `Graph.adjacency(...)`
- DGL-style edge-query semantics that consistently respect public `edata["e_id"]` on derived graphs

What is still missing is a low-level export of adjacency structure as raw index tensors. DGL exposes this through `adj_tensors(...)`, and it is a useful substrate for backend interop, custom sparse kernels, and parity work where callers want structural tensors without constructing a `SparseTensor` wrapper first.

## Scope Choice

There were three reasonable slices:

1. Add `adj_tensors(...)` only.
2. Jump to a larger sparse-export batch including torch-native sparse tensors or richer adjacency wrappers.
3. Move sideways into adjacency mutation APIs.

Option 1 is the right slice. It is small, foundational, and leverages the adjacency ordering model already established by `all_edges(...)` and `inc(...)`. Pulling in torch sparse exports or mutation APIs would widen the blast radius too much for one batch.

## Recommended API

Add one public op:

- `adj_tensors(graph, layout="coo", *, edge_type=None)`

Add one matching `Graph` method:

- `graph.adj_tensors(...)`

The op should live beside the other query primitives in `vgl.ops.query` and should be re-exported from `vgl.ops`.

## Semantics

Supported layouts:

- `"coo"` returns `(src, dst)`
- `"csr"` returns `(crow_indices, col_indices, edge_ids)`
- `"csc"` returns `(ccol_indices, row_indices, edge_ids)`

The returned tensors should use `torch.long` and live on the same device as the source `edge_index`.

For `"coo"`, `src` and `dst` should be ordered by public edge id when `edata["e_id"]` exists, matching the visible ordering of `all_edges(order="eid")`. When no public ids exist, preserve the graph’s native edge order.

For `"csr"` and `"csc"`, the pointer tensor and coordinate tensor should describe the compressed adjacency layout, while the third tensor should expose the public edge ids corresponding to each compressed entry. Within the compressed sort order, ties on identical structural coordinates should remain stable with respect to public edge-id ordering.

## Heterogeneous And Storage Rules

`edge_type` should resolve through the same helper used by the existing query ops. On a graph with multiple relations, callers must select the relation explicitly. The row and column dimensions are implicit in the returned pointer tensors, so no extra shape object is needed.

This API does not need direct node-count validation the way `adjacency(...)` or `inc(...)` do, because the exported tensors only describe existing edges. Featureless storage-backed graphs still need to work because their edge stores materialize `edge_index` eagerly.

## Implementation Approach

The query layer already has the pieces needed:

- `_resolve_edge_type(...)`
- `_public_edge_ids(...)`
- `_ordered_edge_positions(..., order="eid")`
- sparse layout normalization through `_normalize_sparse_layout(...)`

The implementation should:

1. resolve the edge type
2. normalize the requested layout
3. build an edge-position permutation in public-`e_id` order
4. for `"coo"`, return reordered row and column tensors directly
5. for `"csr"` and `"csc"`, apply a stable structural sort on top of the public-`e_id` order and build pointer tensors with `torch.bincount(...)` and cumulative sums

This keeps the new API independent from `SparseTensor`, while still matching the ordering behavior of the sparse conversion layer where appropriate.

## Testing Strategy

Extend coverage in:

- `tests/ops/test_query_ops.py`
- `tests/core/test_graph_ops_api.py`
- `tests/core/test_feature_backed_graph.py`
- `tests/test_package_layout.py`

Key regressions:

- homogeneous `coo`, `csr`, and `csc` outputs
- `coo` honoring public `e_id` on frontier subgraphs
- `csr` and `csc` returning compressed edge ids aligned with their structural order
- duplicate edges staying stable under compression
- heterogeneous relation selection
- featureless storage-backed graphs exposing edge tensors cleanly
- `Graph` bridge and `vgl.ops` export stability

## Non-Goals

This batch should not include:

- a new `adj(...)` matrix wrapper
- torch-native sparse tensor export
- adjacency caching for tensor tuples
- graph mutation APIs
- new sparse layouts beyond `coo`, `csr`, and `csc`
