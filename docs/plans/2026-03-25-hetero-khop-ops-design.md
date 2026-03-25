# Heterogeneous K-Hop Ops Design

## Problem

`vgl.ops.khop` already supported homogeneous graphs, but relation-local heterogeneous graph ops stopped at `node_subgraph(...)` and `compact_nodes(...)`. `khop_nodes(...)` still assumed one flat node-id space, so passing seeds for a bipartite relation like `("author", "writes", "paper")` crashed before users could build a relation-local k-hop subgraph through the public API.

## Goals

- Preserve the existing homogeneous `khop_nodes(...)` and `khop_subgraph(...)` behavior
- Add relation-local heterogeneous support without introducing a second API
- Keep the output shape compatible with `node_subgraph(...)` for bipartite relations
- Document the public contract clearly so the graph-ops surface stays predictable

## Design

### 1. Keep same-type relations on the existing tensor path

If the selected `edge_type` has the same source and destination node type, the current tensor-based implementation remains valid. This keeps homogeneous graphs and single-node-type multi-relation graphs unchanged.

### 2. Use per-type seeds for bipartite relations

For true bipartite heterogeneous relations, `khop_nodes(...)` now requires seeds keyed by node type, for example `{"author": ids}`. The return value is also keyed by node type so callers can preserve source and destination node spaces independently.

### 3. Make direction relation-local and explicit

`direction="out"` expands from the selected relation's source side into the destination side. `direction="in"` expands from destination to source. This keeps the semantics consistent with the chosen relation instead of silently treating the bipartite graph as undirected.

### 4. Reuse the existing hetero subgraph contract

`khop_subgraph(...)` still delegates to `node_subgraph(...)`. Because the returned node ids now use the same per-type shape, relation-local hetero k-hop extraction plugs into the existing subgraph implementation directly.

## Non-goals

- No new `direction="both"` mode
- No whole-graph heterogeneous k-hop expansion across all relations simultaneously
- No changes to sampling-plan `expand_neighbors(...)`, which already has its own hetero contract

## Verification

- Red-green regressions for bipartite hetero `khop_nodes(...)` and `khop_subgraph(...)`
- Focused graph-ops and graph-method wrapper regression suite
- Full repository regression before merge
