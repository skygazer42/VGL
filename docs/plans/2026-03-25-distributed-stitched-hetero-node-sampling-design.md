# Distributed Stitched Hetero Node Sampling Design

## Problem

VGL now stitches shard-local homogeneous node, link, and temporal sampling across partition boundaries through `LocalSamplingCoordinator`, but heterogeneous `NodeNeighborSampler` workloads still expand strictly on the local shard graph. That leaves the remaining large-graph gap on the node-sampling surface: a shard-local hetero seed can fetch routed features correctly, yet it still loses remote frontier structure once a relation crosses a partition boundary.

The local heterogeneous sampler already has clear semantics. It expands one frontier per node type, scans every relation-local edge index, applies one fanout per hop independently per node type, and then materializes one induced heterogeneous subgraph. The missing piece is reproducing those same semantics in typed global-id space through the coordinator before materialization.

## Goals

- Add stitched heterogeneous `NodeNeighborSampler` support for shard-local non-temporal hetero graphs when the loader uses a coordinator-backed feature source
- Keep the public `NodeNeighborSampler`, `Loader`, `NodeBatch`, and coordinator APIs stable for existing callers
- Preserve existing hetero node-sampling semantics for `metadata["node_type"]`, per-hop fanout, and relation-spanning frontier growth
- Reuse the routed typed node/edge feature fetch path so stitched hetero samples keep global `n_id` / `e_id` alignment
- Keep current local-only behavior for homogeneous graphs without a coordinator, link sampling, temporal sampling, and hetero link/temporal workloads

## Recommended Design

Keep the public node sampler plan unchanged and extend `PlanExecutor._expand_neighbors(...)` with one narrow stitched heterogeneous branch that runs before the current local hetero expansion path.

That branch should activate only when:

- the source graph is heterogeneous and non-temporal
- the executor feature source is a coordinator-backed routed source with typed node routing, typed partition membership queries, typed incident-edge queries, and routed typed feature fetch
- every node store on the source graph carries global `n_id`
- those typed `n_id` sets match exactly one coordinator partition, proving the graph is one shard-local partition view rather than a stitched or full graph

When the guard matches, the executor should:

1. Convert the seed local ids for `request.node_type` into global ids through the shard graph `n_id` for that node type.
2. Expand hop-by-hop in typed global-id space. For each relation, route the current frontier for the source node type and destination node type independently, query partition incident edges in global-id form, and collect typed candidate nodes with the same per-hop fanout semantics as the local hetero sampler.
3. After the visited typed node sets are fixed, collect relation-local incident edges, deduplicate them by global `e_id`, and keep only edges whose typed endpoints both remain inside the visited node sets.
4. Build one in-memory stitched heterogeneous graph with relabeled relation-local `edge_index`, typed global `n_id`, typed global `e_id`, and aligned typed node/edge tensors fetched through the coordinator.
5. Build `SampleRecord` payloads against that stitched graph so node materialization can return it directly while preserving `seed`, `node_type`, `sample_id`, and `source_graph_id` metadata.
6. Store typed sampled graph indices in executor state exactly the same way as the current stitched homogeneous branches so later feature-fetch stages continue to align against global ids.

## Non-Goals

- No stitched heterogeneous link sampling in this batch
- No stitched heterogeneous temporal sampling in this batch
- No redesign of `NodeBatch`, hetero tasks, or hetero graph batch contracts
- No new RPC or remote executors
- No mutation of `Graph` objects with partition metadata

## Verification

- Node-sampler regression proving a shard-local hetero seed can stitch a remote typed frontier node and relation edge through the coordinator
- Public sampled hetero node-training regression proving `Loader(..., sampler=NodeNeighborSampler(...), feature_store=coordinator)` can train against the stitched hetero graph path
- Fresh full repository regression before merge
