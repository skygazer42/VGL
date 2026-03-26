# Graph Query And Reverse Ops Design

## Context

`vgl.ops` now covers structure rewrites, line graphs, metapath reachability, random walks, induced subgraphs, frontier subgraphs, k-hop extraction, compaction, and block conversion. The next DGL-class gap on the graph-operations floor is still the basic edge-query surface: callers cannot yet ask for endpoints by edge id, look up edge ids by endpoint pairs, check pair connectivity directly, or materialize a reversed graph through one stable public API.

Those gaps matter because they sit below more visible features. Samplers, inspection tools, debugging utilities, and DGL compatibility layers all benefit from a small set of stable graph-query primitives. They also become more important now that `in_subgraph(...)` and `out_subgraph(...)` preserve `edata["e_id"]`: once VGL keeps public edge ids alive across derived graphs, the query layer should know how to use them.

## Scope Choice

Several plausible slices were considered:

1. Add only `find_edges(...)` and stop there.
2. Add `find_edges(...)`, `edge_ids(...)`, and `has_edges_between(...)` without graph transforms.
3. Add the edge-query trio plus `reverse(...)` as one compact batch.
4. Jump to larger DGL gaps such as `in_edges(...)`, `out_edges(...)`, `successors(...)`, `predecessors(...)`, or graph mutation APIs.

Option 3 is the right batch. The three query operators share one edge-id resolution model, and `reverse(...)` composes naturally with them while still staying small. This gives VGL one coherent low-level graph-query foundation without widening into adjacency iteration or mutating graph structure in place.

## Recommended API

Add four public ops:

- `find_edges(graph, eids, *, edge_type=None)`
- `edge_ids(graph, u, v, *, return_uv=False, edge_type=None)`
- `has_edges_between(graph, u, v, *, edge_type=None)`
- `reverse(graph, *, copy_ndata=True, copy_edata=False)`

Add matching `Graph` convenience methods:

- `graph.find_edges(...)`
- `graph.edge_ids(...)`
- `graph.has_edges_between(...)`
- `graph.reverse(...)`

This batch should keep the existing VGL edge-type convention: callers may omit `edge_type` for the default or single relation, otherwise pass the canonical edge-type tuple. DGL's string edge-type disambiguation is intentionally out of scope for now because the current VGL ops layer consistently uses canonical tuples.

## Edge Id Semantics

The key design rule is that query ops should resolve against public edge ids when they are available.

Concretely:

- if an edge store contains `edata["e_id"]`, then `find_edges(...)` should treat the input ids as those public ids
- if an edge store does not contain `e_id`, then raw local edge positions remain the public ids
- `edge_ids(...)` should return those same public ids when `return_uv=False`
- `edge_ids(..., return_uv=True)` should return `(u, v, eids)` in query order, where `eids` uses the public-id view

This keeps query behavior stable across derived graphs such as frontier subgraphs and blocks that preserve original edge identity explicitly.

## Query Behavior

`find_edges(...)` should return the source and destination endpoints for the requested edge ids in input order. Unknown edge ids should raise `ValueError`.

`edge_ids(...)` should accept one source collection and one destination collection with the same number of pairs. Behavior should follow DGL closely enough for this batch:

- when `return_uv=False`, return one public edge id per queried pair
- when multiple parallel edges connect the same pair, return the first matching edge in graph order
- when any queried pair has no matching edge, raise `ValueError`
- when `return_uv=True`, return every matching edge for every queried pair in pair order and stable edge order

`has_edges_between(...)` should accept the same pair input shape and return a Python `bool` for scalar-scalar input or a boolean tensor for vector input.

All three query ops should validate pair shapes explicitly and reject out-of-range node ids with `ValueError`.

## Reverse Graph Semantics

`reverse(...)` should flip the selected graph structure edge-wise while preserving node types and node counts.

For homogeneous graphs this means swapping `edge_index[0]` and `edge_index[1]`. For heterogeneous graphs every relation key should become `(dst_type, relation_name, src_type)` and its edge index should be reversed per edge.

Feature-copy rules:

- `copy_ndata=True`: preserve the input node stores as shared data
- `copy_ndata=False`: return empty node stores while retaining schema node types and graph-level node-count context
- `copy_edata=False`: drop ordinary edge features, but still preserve `e_id` as structural identity
- `copy_edata=True`: copy edge-aligned tensors onto the reversed relation in the same edge order

The returned graph should preserve `feature_store` and `graph_store` references so feature-backed or featureless storage-backed graphs keep node-count context and any shared node feature access already present on the input graph.

## Testing Strategy

Add focused coverage in a new `tests/ops/test_query_ops.py` for:

- `find_edges(...)` on plain edge ids and preserved public `e_id`
- `edge_ids(...)` on simple and parallel edges
- `edge_ids(..., return_uv=True)` stable enumeration behavior
- `has_edges_between(...)` scalar and vector forms
- `reverse(...)` on homogeneous and heterogeneous graphs
- `reverse(...)` copy flags for node and edge features, including preserved `e_id`

Add graph-API bridge coverage in `tests/core/test_graph_ops_api.py`, package export coverage in `tests/test_package_layout.py`, and a storage-backed regression in `tests/core/test_feature_backed_graph.py` proving a featureless reversed graph still preserves node-count-sensitive adjacency shape.

## Non-Goals

This batch should not include:

- `in_edges(...)` / `out_edges(...)`
- `successors(...)` / `predecessors(...)`
- graph mutation APIs such as `add_edges(...)` or `remove_edges(...)`
- DGL string edge-type disambiguation
- shared-data toggle aliases such as deprecated `share_ndata` / `share_edata`
- distributed coordinator query helpers
