# Adjacency Query Ops Design

## Context

`vgl.ops.query` now covers direct edge-id and pair-id lookups through `find_edges(...)`, `edge_ids(...)`, and `has_edges_between(...)`, while `reverse(...)`, frontier subgraphs, and blocks preserve public edge identity through `edata["e_id"]`. The next DGL-class gap immediately above that layer is adjacency query ergonomics: callers still cannot ask for all inbound or outbound edges of a node set, or retrieve one-hop predecessor and successor node lists directly from the public graph API.

Those operators matter because they are the natural read-only companion to the query primitives added in the previous batch. They are also useful substrate for later parity work such as `in_degrees/out_degrees`, higher-level neighborhood inspection, sampler debugging, and DGL adapter behaviors that expect edge-form selection instead of only pair-by-pair lookups.

## Scope Choice

Several possible slices were considered:

1. Add only `in_edges(...)` and `out_edges(...)`.
2. Add `in_edges(...)`, `out_edges(...)`, `predecessors(...)`, and `successors(...)` as one compact adjacency batch.
3. Jump directly to larger traversal helpers such as `in_edges/out_edges` plus `in_degrees/out_degrees`.
4. Continue upward into mutation or message-flow APIs.

Option 2 is the right batch. `in_edges(...)` and `out_edges(...)` define the edge-facing adjacency layer, while `predecessors(...)` and `successors(...)` are the node-facing one-hop views over the same ordered edge selections. Together they close a coherent DGL-style gap without dragging degree bookkeeping or graph mutation into the same change set.

## Recommended API

Add four public ops:

- `in_edges(graph, v, *, form="uv", edge_type=None)`
- `out_edges(graph, u, *, form="uv", edge_type=None)`
- `predecessors(graph, v, *, edge_type=None)`
- `successors(graph, v, *, edge_type=None)`

Add matching `Graph` convenience methods:

- `graph.in_edges(...)`
- `graph.out_edges(...)`
- `graph.predecessors(...)`
- `graph.successors(...)`

This batch should keep the existing VGL edge-type convention: callers may omit `edge_type` for the default or single relation, otherwise pass the canonical edge-type tuple.

## Edge Id And Ordering Semantics

These adjacency ops should reuse the same public-edge-id model already established in `vgl.ops.query`:

- if an edge store contains `edata["e_id"]`, adjacency APIs should expose those ids instead of local positions when they return edge ids
- if `e_id` is absent, raw local edge positions remain the public edge ids
- returned edge endpoints and node lists should stay aligned to the selected edge order

Ordering should follow stable edge order in the current graph view. That means frontier subgraphs, reversed graphs, and future derived graphs can keep adjacency query results consistent with their current stored edge order while still surfacing original/public edge ids through `e_id`.

## Form Semantics

`in_edges(...)` and `out_edges(...)` should support the common DGL forms:

- `form="uv"`: return `(src, dst)`
- `form="eid"`: return `eid`
- `form="all"`: return `(src, dst, eid)`

For scalar node input, these functions should still return tensors. For iterable node input, they should concatenate all matching edges into one ordered result. Invalid `form` values should raise `ValueError`.

## Neighbor Semantics

`predecessors(...)` and `successors(...)` should accept one node id at a time and return one-dimensional node-id tensors.

They should preserve duplicates when multiple parallel edges exist, because these functions are adjacency views over edges rather than set-valued neighborhood summaries. For example, if two edges `0 -> 1` exist, then `predecessors(1)` should contain two copies of `0` in stable edge order.

## Validation And Storage Context

All four ops should validate queried node ids against the current graph's node counts. That matters especially for featureless storage-backed graphs, where node counts can exceed the largest node id observed in `edge_index`. Those graphs already preserve `graph_store`, so the new adjacency queries should use that retained context through `graph._node_count(...)` instead of inferring counts from edge structure alone.

This batch should not change graph structure or feature retention. All adjacency queries are read-only tensor projections over the selected edge store.

## Testing Strategy

Extend `tests/ops/test_query_ops.py` with coverage for:

- `in_edges(...)` and `out_edges(...)` on homogeneous graphs for `uv`, `eid`, and `all` forms
- `in_edges(...)` / `out_edges(...)` resolving preserved public `e_id` on derived graphs such as frontier subgraphs
- `predecessors(...)` and `successors(...)` preserving duplicates for parallel edges
- selected-relation heterogeneous adjacency queries
- invalid `form` and out-of-range node validation

Add graph-API bridge coverage in `tests/core/test_graph_ops_api.py`, package export coverage in `tests/test_package_layout.py`, and one featureless storage-backed regression in `tests/core/test_feature_backed_graph.py` proving adjacency queries can still address isolated-but-declared nodes through retained node-count context.

## Non-Goals

This batch should not include:

- degree operators such as `in_degrees(...)` or `out_degrees(...)`
- set-valued neighborhood deduplication helpers
- mutation APIs such as `add_edges(...)` or `remove_edges(...)`
- DGL string edge-type disambiguation
- multi-hop traversal helpers beyond one-hop predecessor/successor queries
