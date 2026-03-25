# Distributed Stitched Node Sampling Design

## Problem

VGL's local distributed runtime can now persist partition-local graphs, route global node and edge ids through `LocalSamplingCoordinator`, align sampled features against global ids, and expose boundary plus incident edge queries per partition. The missing step is structural stitching during node sampling. When `NodeNeighborSampler` runs against a shard-local graph, `expand_neighbors(...)` still only walks the local shard adjacency, so cross-partition frontiers disappear even if the coordinator can already see them.

That leaves a real bottom-layer gap versus DGL-class large-graph runtimes. A shard-local node batch can fetch the right features through the coordinator, but it still cannot sample the remote node and edge structure needed to continue a frontier across a partition boundary.

## Goals

- Add stitched homogeneous node sampling for shard-local graphs when the loader is given a coordinator-backed feature source
- Keep the public `NodeNeighborSampler`, `Loader`, `Graph`, and coordinator APIs additive for current callers
- Reuse the new incident-edge queries instead of mutating shard graphs to contain halo nodes
- Preserve current local-only sampling behavior for full graphs, hetero graphs, temporal graphs, and non-coordinator feature stores
- Keep stitched samples feature-aligned by global `n_id` and `e_id`

## Recommended Design

Keep the plan shape the same and teach `PlanExecutor._expand_neighbors(...)` one narrow stitched path.

The stitched path should activate only when all of the following are true:

- the source graph is homogeneous and non-temporal
- the executor feature source exposes coordinator-style routing and incident-edge queries
- the source graph carries global `n_id`
- those `n_id` values match one coordinator partition exactly, proving the graph is a shard-local partition view rather than a full graph

When that guard matches, the executor should:

1. Map seed local ids to global ids through `graph.n_id`.
2. Expand hop by hop in global-id space. For each hop, route the current frontier to owning partitions, fetch each partition's incident edge index, gather candidate nodes touching the frontier, and apply the same fanout semantics as the local `expand_neighbors(...)` path.
3. After the visited node set is fixed, collect incident edges for the visited-node owner partitions, deduplicate by global `e_id`, and keep only edges whose endpoints both remain inside the visited node set.
4. Build an in-memory stitched homogeneous graph directly with local relabeled `edge_index`, global `n_id`, global `e_id`, and every aligned tensor feature already present on the shard graph schema fetched through the coordinator.
5. Store that graph plus the seed positions in executor state so node materialization can return it directly instead of trying to subgraph the original shard graph again.

The existing optional feature-fetch stages should remain compatible. They may still run after stitched expansion, but they should preserve global ids in their returned `TensorSlice.index` so any fetched overlays continue to align with the stitched graph's global `n_id` and `e_id`.

## Non-Goals

- No stitched heterogeneous node sampling in this batch
- No stitched link prediction or temporal event sampling in this batch
- No new RPC, multiprocessing, or remote executors
- No mutation of `Graph` objects to add partition metadata or dynamic attributes
- No attempt to expose DistDGL-complete distributed sampling semantics yet

## Verification

- Executor-level or sampler-level regression proving a shard-local homogeneous seed can include a remote frontier node and boundary edge through the coordinator
- Loader regression proving coordinator-backed feature prefetch still aligns with the stitched graph's global ids
- Integration regression proving public sampled training can consume a stitched shard-local node batch through the coordinator-backed loader path
- Fresh full repository regression before merge
