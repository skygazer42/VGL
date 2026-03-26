# Incidence Sparse View Design

## Context

VGL now has a credible sparse backend for adjacency-oriented work: `Graph.adjacency(...)` returns cached COO/CSR/CSC `SparseTensor` views, and `vgl.sparse` already supports conversion, transpose, row/column selection, reductions, `spmm(...)`, `sddmm(...)`, and `edge_softmax(...)`. On the graph-op surface, recent batches filled in DGL-style edge and degree queries such as `all_edges(...)`, `in_edges(...)`, and `in_degrees(...)`.

What is still missing is a basic incidence-matrix view. DGL exposes this through `inc(...)`, and it is a useful substrate for edge-centric linear algebra, flow-style structure checks, and graph-operator parity work that needs a sparse matrix with one column per edge instead of one nonzero per adjacency pair.

## Scope Choice

There were three plausible next slices:

1. Add incidence-matrix views only.
2. Jump straight to a larger sparse-export batch including raw adjacency tensors or multiple sparse-view families.
3. Move upward into graph mutation APIs such as `add_edges(...)` or `remove_edges(...)`.

Option 1 is the right slice. Incidence views are small, foundational, and directly adjacent to the sparse backend already in place. Pulling in broader sparse export or mutation work would widen the surface area too much for one batch.

## Recommended API

Add one public op:

- `inc(graph, typestr="both", *, layout="coo", edge_type=None)`

Add one matching `Graph` method:

- `graph.inc(...)`

The name should intentionally mirror DGL here. It is short and already carries the expected semantics for users coming from DGL. Unlike `edges`, there is no collision with an existing graph attribute.

## Semantics

`inc(...)` should return a `vgl.sparse.SparseTensor` whose second dimension is always the number of edges in the selected relation.

Supported `typestr` values:

- `"in"`: one `+1` per edge at the destination-node row
- `"out"`: one `+1` per edge at the source-node row
- `"both"`: one `-1` at the source row plus one `+1` at the destination row

For `"both"`, self-loops should contribute a zero column, matching DGL’s observable behavior of omitting any nonzero entries for self-loop columns in the incidence matrix.

## Shape Rules

For homogeneous or same-type relations:

- `"in"` shape is `(num_dst_nodes, num_edges)`
- `"out"` shape is `(num_src_nodes, num_edges)`
- `"both"` shape is `(num_nodes, num_edges)`

For bipartite relations:

- `"in"` shape uses the destination node type count
- `"out"` shape uses the source node type count
- `"both"` should fail clearly because positive and negative entries would belong to different row spaces

## Edge Ordering And Public Ids

VGL now has a stable public `e_id` model on derived graphs. The incidence view should stay consistent with that by ordering columns according to public edge ids when present, which matches the default `all_edges(order="eid")` behavior. Columns remain contiguous `0..num_edges-1` in the sparse tensor, but their order should align with the sorted public edge-id space of the current graph view.

This avoids creating enormous sparse matrices with gaps when a derived graph carries non-contiguous `e_id`, while still keeping the visible edge-column order aligned with the public edge identity model.

## Layout And Storage Context

The incidence op should accept the same `layout` normalization pattern as `Graph.adjacency(...)`: strings such as `"coo"`, `"csr"`, and `"csc"` plus `SparseLayout` enum values. Internally, it can build one COO tensor and convert through existing sparse helpers.

Node counts must come from `graph._node_count(...)`, not from the maximum endpoint id observed in `edge_index`, so featureless storage-backed graphs preserve declared but isolated nodes in the row dimension.

## Testing Strategy

Extend coverage in:

- `tests/ops/test_query_ops.py` for homogeneous, self-loop, public-`e_id`, hetero, and validation behavior
- `tests/core/test_graph_ops_api.py` for `Graph.inc(...)`
- `tests/core/test_feature_backed_graph.py` for storage-backed declared node counts
- `tests/test_package_layout.py` for stable `vgl.ops` exports

Key regressions:

- `"in"`, `"out"`, and `"both"` values on a simple homogeneous graph
- `"both"` omitting self-loop nonzeros while preserving the edge-column dimension
- column order following public `e_id` on derived graphs
- bipartite `"both"` rejection
- featureless storage-backed graphs preserving declared row counts

## Non-Goals

This batch should not include:

- raw backend-native sparse tensor export such as torch sparse tensors
- adjacency tensor tuple exports
- graph mutation APIs
- incidence caching
- hypergraph or signed-graph generalizations beyond DGL-style incidence semantics
