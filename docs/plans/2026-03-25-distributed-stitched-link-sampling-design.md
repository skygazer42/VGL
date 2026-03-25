# Distributed Stitched Link Sampling Design

## Problem

VGL can now stitch shard-local homogeneous node sampling across partition boundaries through `LocalSamplingCoordinator`, but `LinkNeighborSampler` still samples structure strictly from the local shard graph. That leaves an immediate large-graph gap for link prediction workloads: a positive or negative edge whose endpoints live on one shard cannot expand into remote frontier structure, even though the runtime already has partition incident-edge queries, routed feature fetch, and stitched homogeneous node-subgraph construction.

In practice, the coordinator can already answer the structural questions needed for stitched homogeneous link sampling. The missing piece is using those queries during `sample_link_neighbors` so link records materialize against a stitched message-passing graph instead of a shard-local one.

## Goals

- Add stitched homogeneous `LinkNeighborSampler` support for shard-local graphs when the loader uses a coordinator-backed feature source
- Keep the public `LinkNeighborSampler`, `Loader`, `LinkPredictionBatch`, and coordinator APIs stable for existing callers
- Reuse the existing stitched homogeneous node-sampling helpers instead of introducing a second distributed graph format
- Preserve current local-only behavior for full graphs, heterogeneous graphs, temporal graphs, and non-coordinator feature sources
- Keep sampled graph `n_id` / `e_id` globally aligned so existing routed feature materialization continues to work

## Recommended Design

Keep the plan shape unchanged and extend `PlanExecutor._sample_link_neighbors(...)` with one narrow stitched homogeneous branch.

That branch should activate only when:

- all records come from the same homogeneous, non-temporal shard-local graph
- the executor feature source is a coordinator-like routed source that already supports node routing, incident-edge queries, and routed feature fetch
- the source graph carries global `n_id`
- the shard-local graph matches exactly one coordinator partition, the same guard already used for stitched node sampling

When the guard matches, the executor should:

1. Convert each record's local `src_index` / `dst_index` to global node ids through the shard graph `n_id`.
2. Take the union of those global endpoints as the stitched seed set and expand hop-by-hop in global-id space through coordinator incident-edge queries with the same fanout semantics used by stitched node sampling.
3. Collect stitched incident edges whose endpoints both remain inside the visited node set, deduplicate by global `e_id`, and build one in-memory homogeneous graph with relabeled `edge_index`, global `n_id`, global `e_id`, and aligned node/edge tensors fetched through the coordinator.
4. Rebuild each `LinkPredictionRecord` so its `src_index` / `dst_index` point into the stitched graph while preserving existing metadata, labels, and link-task flags.
5. Store the stitched graph indices in executor state exactly the same way as sampled node stitching so any later feature-fetch stages continue to align against global ids.

## Non-Goals

- No stitched heterogeneous link sampling in this batch
- No stitched temporal event sampling in this batch
- No distributed negative-sampler redesign or DistDGL semantics
- No new RPC or remote executors
- No mutation of core `Graph` objects with partition metadata

## Verification

- Link-sampler regression proving a shard-local homogeneous link record can stitch a remote frontier node and edge through the coordinator
- Public sampled link-training regression proving `Loader(..., sampler=LinkNeighborSampler(...), feature_store=coordinator)` can train against the stitched graph path
- Fresh full repository regression before merge
