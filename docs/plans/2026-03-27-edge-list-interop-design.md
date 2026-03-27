# Edge-List Interoperability Design

## Context

VGL now has compatibility bridges for PyG, DGL, and homogeneous NetworkX graphs, plus an on-disk dataset container. One practical data-ecosystem gap still remains at the lowest representation level: there is no public edge-list bridge.

That matters because edge lists are the simplest durable graph representation. They are the handoff format used by quick scripts, preprocessing steps, CSV/tabular loaders, and many migration paths before a user graduates to a richer graph container. Without an edge-list surface, VGL can interoperate with large frameworks but still misses the smallest common denominator.

## Scope Choice

Three slices were considered:

1. Full CSV file I/O immediately.
2. In-memory edge-list interoperability first.
3. Heterogeneous edge-list conventions in the same batch.

Option 2 is the right slice.

CSV I/O is built on top of a stable in-memory representation, and heterogeneous edge-list conventions widen the design too quickly. A homogeneous in-memory edge-list layer is the reusable substrate that later file formats can build on cleanly.

## Recommended API

Add:

- `vgl.compat.from_edge_list(edge_list, *, num_nodes=None, node_data=None, edge_data=None)`
- `vgl.compat.to_edge_list(graph)`
- `Graph.from_edge_list(...)`
- `graph.to_edge_list()`

Keep the scope intentionally narrow:

- only homogeneous graphs in this batch
- `edge_list` input may be a Python sequence of `(src, dst)` pairs or a tensor shaped `(E, 2)` or `(2, E)`
- export returns a `torch.Tensor` shaped `(E, 2)` with `dtype=torch.long`
- hetero export fails clearly

## Semantics

### Export

`to_edge_list(graph)` should:

- require a homogeneous graph
- return edges in stored edge order
- preserve parallel edges and self-loops naturally
- return an empty `(0, 2)` tensor for empty graphs

Edge features remain on the graph object; this batch only bridges structure, not a tabular feature bundle.

### Import

`from_edge_list(...)` should:

- normalize the input into canonical `(2, E)` `edge_index`
- accept optional `node_data` and `edge_data`
- infer `num_nodes` from edge endpoints or node-aligned features when omitted
- validate that explicit `num_nodes` is large enough for all endpoints
- preserve isolated declared nodes when `num_nodes` exceeds the maximum endpoint id

To preserve node counts for featureless graphs, the import helper may materialize `n_id` when needed, following the same pattern already used in other compatibility adapters.

## Non-Goals

- CSV file reading/writing
- heterogeneous edge-list conventions
- arbitrary string node labels
- block interoperability
- automatic edge-feature table reconstruction from edge-list export

## Testing Strategy

Add focused coverage for:

- homogeneous graph export to ordered edge-list tensors
- import from Python edge pairs and tensor edge lists
- explicit `num_nodes` preserving isolated nodes
- transposed tensor input normalization
- clear failure on heterogeneous export
- compat and package exports

Then run focused compat/package tests and the full suite.
