# Hetero Graph Ops Expansion Design

## Goal

Expand `vgl.ops` so the existing structural graph operators work on heterogeneous relation slices instead of stopping at homogeneous-only guards. The immediate target is to support hetero `node_subgraph`, hetero `edge_subgraph`, and hetero `compact_nodes` without changing the current `Graph` abstraction or introducing a second graph-ops API.

## Why This Next

After the foundation rollout and sparse runtime expansion, one of the clearest remaining gaps relative to DGL is the public graph-operations surface for heterogeneous graphs. The current code already exposes the right operator names, but several paths stop with explicit `heterogeneous support is not implemented yet` or homogeneous-only constraints. That makes this a high-value, bounded follow-up: the surface already exists, the missing behavior is local, and the tests can be written directly against the current API.

## Scope

This batch keeps the current operator names and extends them in-place:

- `node_subgraph(graph, node_ids, edge_type=...)` supports hetero relation slices when `node_ids` is a dict keyed by node type
- `edge_subgraph(graph, edge_ids, edge_type=...)` supports hetero relation slices while preserving node spaces for the participating node types
- `compact_nodes(graph, node_ids, edge_type=...)` supports hetero relation slices and returns per-type mappings
- docs and regression tests describe the new hetero behavior

## Non-Goals

This batch does not attempt whole-graph hetero subgraph extraction across every edge type simultaneously, distributed graph rewrites, or a DGL-style metagraph utility layer. It also does not change sampling or storage APIs.

## Design Notes

The implementation should stay relation-local: when a specific `edge_type` is selected, the op only needs to materialize the source and destination node types involved in that relation. For hetero `node_subgraph`, node id selection should be provided as `{node_type: ids}` and relabel source and destination spaces independently. For hetero `edge_subgraph`, edge ids continue to be a flat tensor and node spaces remain unchanged. For hetero `compact_nodes`, the result should include one mapping dict per participating node type so downstream code can recover original ids explicitly.
