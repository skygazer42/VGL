# Distributed Stitched Temporal Sampling Design

## Problem

VGL can now stitch shard-local homogeneous node and link sampling across partition boundaries through `LocalSamplingCoordinator`, but `TemporalNeighborSampler` still samples history strictly from the local shard graph. That leaves a remaining large-graph gap for temporal event prediction workloads: an event record on one shard cannot expand into earlier remote history even though the runtime already has partition ownership metadata, global `n_id` / `e_id` alignment, routed feature fetch, and stitched homogeneous graph construction.

Temporal sampling is narrower than node/link sampling because structure must respect time. The sampler first filters eligible history edges by `timestamp`, `strict_history`, `time_window`, and `max_events`, and only then expands the frontier over that history subgraph. The missing piece is reproducing that temporal-history filtering in distributed global-id space before building the stitched sampled graph.

## Goals

- Add stitched homogeneous `TemporalNeighborSampler` support for shard-local temporal graphs when the loader uses a coordinator-backed feature source
- Keep the public `TemporalNeighborSampler`, `Loader`, `TemporalEventBatch`, and coordinator APIs stable for existing callers
- Preserve existing temporal semantics for `strict_history`, `time_window`, and `max_events`
- Reuse the stitched homogeneous data/feature alignment helpers already used by distributed node and link sampling
- Keep sampled temporal graph `n_id`, `e_id`, and edge timestamps globally aligned so later feature materialization continues to work

## Recommended Design

Keep the public temporal sampler plan unchanged and extend `PlanExecutor._sample_temporal_neighbors(...)` with one narrow stitched homogeneous branch.

That branch should activate only when:

- the record graph is homogeneous and temporal
- the feature source is a coordinator-backed routed source for the exact shard that produced the record graph
- the shard graph carries global `n_id`
- the sampled edge type is the homogeneous default edge type

When the guard matches, the executor should:

1. Convert the record's local `src_index` / `dst_index` to global node ids through the shard graph `n_id`.
2. Build a global eligible history edge set for the sampled edge type by scanning owned partition edges once, filtering them by the record timestamp with the same `strict_history` / `time_window` logic as `TemporalNeighborSampler`, and applying the same global `max_events` truncation semantics.
3. Expand the frontier hop-by-hop over that filtered global history edge index with the same fanout semantics as local temporal sampling.
4. Keep only filtered history edges whose endpoints both remain inside the visited node set, relabel them into one stitched homogeneous temporal graph, and hydrate node/edge tensors through the coordinator so `n_id`, `e_id`, `timestamp`, and optional fetched features stay globally aligned.
5. Rebuild the `TemporalEventRecord` against the stitched temporal graph while preserving `timestamp`, `label`, `event_features`, `metadata`, `sample_id`, and `edge_type`.
6. Store sampled graph indices in executor state exactly the same way as the stitched node/link paths so later feature-fetch stages continue to align against global ids.

## Non-Goals

- No stitched heterogeneous temporal sampling in this batch
- No redesign of temporal batching, temporal tasks, or temporal encoders
- No new RPC or remote executors
- No partition-writer format changes
- No mutation of core `Graph` objects with partition metadata

## Verification

- Temporal sampler regression proving a shard-local homogeneous temporal event can stitch earlier remote history through the coordinator
- Public sampled temporal-training regression proving `Loader(..., sampler=TemporalNeighborSampler(...), feature_store=coordinator)` can train against the stitched history graph path
- Fresh full repository regression before merge
