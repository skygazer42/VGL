# Cardinality And Full-Edge Query Ops Design

## Context

`vgl.ops.query` now covers edge identity lookups, adjacency views, and degree inspection through `find_edges(...)`, `edge_ids(...)`, `has_edges_between(...)`, `in_edges(...)`, `out_edges(...)`, `predecessors(...)`, `successors(...)`, `in_degrees(...)`, and `out_degrees(...)`. What is still missing from the public graph surface is a basic DGL-style way to ask for graph cardinality and to enumerate all edges in a relation with explicit ordering semantics.

That gap shows up immediately when code is ported from DGL. Callers expect to be able to ask for `num_nodes`, `num_edges`, and one shot “give me every edge” queries without reaching into `graph.schema`, summing per-type counts manually, or unpacking `edge_index` directly and rebuilding edge ids themselves.

## Scope Choice

There were three realistic next slices:

1. Add only `num_nodes(...)` and `num_edges(...)`.
2. Add graph cardinality plus DGL-style full-edge enumeration through `all_edges(...)`.
3. Jump into broader sparse-matrix views such as incidence operators.

Option 2 is the right batch. Cardinality and full-edge enumeration are both small, frequently used, and directly adjacent to the current query layer. Incidence or broader sparse views are worth doing later, but they are a larger design surface than this immediate parity gap.

## Recommended API

Add five public ops:

- `num_nodes(graph, node_type=None)`
- `number_of_nodes(graph, node_type=None)`
- `num_edges(graph, edge_type=None)`
- `number_of_edges(graph, edge_type=None)`
- `all_edges(graph, *, form="uv", order="eid", edge_type=None)`

Add matching `Graph` convenience methods:

- `graph.num_nodes(...)`
- `graph.number_of_nodes(...)`
- `graph.num_edges(...)`
- `graph.number_of_edges(...)`
- `graph.all_edges(...)`

`all_edges(...)` is the right DGL-style name for VGL because `graph.edges` is already a store dictionary attribute and cannot become a callable method without breaking the existing object model.

## Cardinality Semantics

`num_nodes(...)` and `number_of_nodes(...)` should follow DGL’s useful heterogeneous behavior:

- if `node_type` is omitted, return the total number of declared nodes across all node types
- if `node_type` is given, return the count for that specific node type

`num_edges(...)` and `number_of_edges(...)` should mirror that pattern:

- if `edge_type` is omitted, return the total number of stored edges across all relations
- if `edge_type` is given, return the count for that specific relation

All four functions should return Python `int`.

## Full-Edge Enumeration Semantics

`all_edges(...)` should support the same forms already used by `in_edges(...)` and `out_edges(...)`:

- `form="uv"` returns `(src, dst)`
- `form="eid"` returns `eid`
- `form="all"` returns `(src, dst, eid)`

It should support two useful orderings:

- `order="eid"` returns edges in public edge-id order
- `order="srcdst"` returns edges lexicographically by source, then destination, then public edge id

Invalid `form` or `order` values should raise `ValueError`.

## Public Edge Id Model

This batch should stay aligned with the public-edge-id rules already established elsewhere in `vgl.ops.query`:

- if an edge store carries `edata["e_id"]`, that tensor defines the public edge-id space
- if `e_id` is absent, local edge positions remain the public ids

That matters for both returned edge ids and ordering. `all_edges(order="eid")` should sort by public edge ids, not merely by local storage position, so derived graphs and storage-backed graphs continue to expose one stable public identity model.

## Validation And Storage Context

Cardinality queries must continue to respect retained storage-backed graph metadata. `num_nodes(...)` should use `graph._node_count(...)`, which already knows how to preserve declared but isolated nodes on featureless storage-backed graphs. `num_edges(...)` can count relation-local `edge_index` columns directly because edge counts do not require feature-store inference.

`all_edges(...)` should remain relation-local. If the graph has multiple relations and `edge_type` is omitted, it should fail clearly instead of trying to concatenate heterogeneous edge spaces into one ambiguous tensor triple.

## Testing Strategy

Extend coverage in:

- `tests/ops/test_query_ops.py` for count ops, aliases, `all_edges(...)` forms, `eid` ordering, `srcdst` ordering, hetero totals, and validation
- `tests/core/test_graph_ops_api.py` for `Graph` method bridges
- `tests/core/test_feature_backed_graph.py` for storage-backed declared node counts
- `tests/test_package_layout.py` for stable `vgl.ops` exports

Key regressions:

- homogeneous and heterogeneous `num_nodes(...)` / `num_edges(...)`
- alias parity through `number_of_nodes(...)` / `number_of_edges(...)`
- `all_edges(...)` returning public `e_id` values
- `all_edges(order="eid")` using public edge-id order instead of local storage order
- `all_edges(order="srcdst")` returning lexicographically sorted endpoints
- featureless storage-backed graphs preserving declared node totals

## Non-Goals

This batch should not include:

- callable `graph.edges(...)` or `graph.nodes(...)` APIs that would conflict with the existing store dictionaries
- incidence-matrix construction
- graph mutation APIs such as `add_edges(...)`
- neighborhood deduplication helpers
- sampler or distributed execution changes
