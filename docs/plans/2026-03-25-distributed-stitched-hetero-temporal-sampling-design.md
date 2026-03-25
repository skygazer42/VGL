# Distributed Stitched Hetero Temporal Sampling Design

## Problem

VGL can now stitch shard-local homogeneous temporal sampling across partition boundaries, and it can stitch shard-local non-temporal heterogeneous node/link sampling through `LocalSamplingCoordinator`. The remaining distributed sampler gap is typed heterogeneous temporal sampling: `TemporalNeighborSampler` already supports relation-local temporal history extraction for typed hetero graphs, but when the source graph is a shard-local partition graph it still walks only the shard-local history and therefore loses earlier remote typed history once the sampled relation crosses a partition boundary.

This gap sits at the executor boundary between the existing stitched homogeneous temporal path and the existing relation-local hetero temporal local path. The coordinator already exposes the partition shard payloads, typed global node ids, relation-scoped boundary edges, and routed feature fetches needed to recover remote history. The missing piece is reconstructing the filtered relation-local temporal history in global-id space, then running the same typed frontier expansion and materialization steps against that filtered history.

## Goals

- Add stitched typed heterogeneous `TemporalNeighborSampler` support for shard-local temporal graphs when the loader uses a coordinator-backed feature source
- Keep the public `TemporalNeighborSampler`, `Loader`, `TemporalEventBatch`, and coordinator APIs stable for existing callers
- Preserve existing temporal semantics for `strict_history`, `time_window`, and `max_events` on typed heterogeneous temporal graphs
- Preserve existing relation-local semantics: only the sampled `edge_type` contributes temporal history, and the sampled graph keeps only the source/destination node types for that relation
- Keep sampled typed `n_id`, typed `e_id`, and temporal edge attributes globally aligned so later feature materialization continues to work without special cases

## Recommended Design

Keep the public temporal sampler plan unchanged and extend `PlanExecutor._sample_temporal_neighbors(...)` with one stitched heterogeneous branch that activates after the existing stitched homogeneous branch and before the local fallback.

That branch should activate only when:

- the seed record graph is temporal and heterogeneous rather than homogeneous
- the record has one resolved `edge_type`, and sampling remains relation-local to that edge type
- the feature source is a coordinator-backed routed source for the exact shard that produced that graph
- every node store on the shard graph carries global `n_id`, and the shard graph matches exactly one coordinator partition

When the guard matches, the executor should:

1. Resolve the seed record's typed source and destination node types from `edge_type`, then lift the local `src_index` / `dst_index` to typed global node ids through the shard graph's per-type `n_id`.
2. Build a filtered relation-local global history edge set for that one `edge_type` by scanning owned plus boundary partition edges once, applying the same `strict_history` / `time_window` timestamp mask and the same stable `max_events` truncation used by local temporal sampling.
3. Expand the frontier hop-by-hop over that filtered relation-local global history edge index with the same per-hop fanout semantics used by the current typed hetero temporal local sampler, keeping typed visited node ids in global-id space.
4. Keep only filtered relation-local history edges whose endpoints remain inside the visited typed node set, relabel them into one stitched typed temporal graph, and hydrate node/edge tensors through the coordinator so per-type `n_id`, relation-local `e_id`, timestamps, and optional fetched features stay globally aligned.
5. Rebuild the `TemporalEventRecord` against that stitched graph while preserving `timestamp`, `label`, `event_features`, `metadata`, `sample_id`, and `edge_type`.
6. Store sampled typed node and edge ids in executor state the same way as other stitched hetero paths so later feature-fetch stages continue to align against global ids.

The implementation should stay narrow and relation-local. It should not try to combine multiple temporal relations into one stitched history graph, because that would diverge from current typed hetero temporal semantics and would enlarge the scope into a temporal batching redesign.

## Non-Goals

- No multi-relation stitched temporal history in one sampled graph
- No redesign of `TemporalEventBatch`, temporal tasks, or temporal encoders
- No new RPC or remote execution runtime
- No partition-writer format changes
- No mutation of core `Graph` objects with partition metadata

## Verification

- Temporal sampler regression proving a shard-local typed hetero temporal event can stitch earlier remote typed history through the coordinator
- Public sampled temporal-training regression proving `Loader(..., sampler=TemporalNeighborSampler(...), feature_store=coordinator)` can train against the stitched typed temporal history path
- Focused temporal, link, node, and partition-local regressions around the executor and feature materialization surfaces
- Fresh full repository regression before merge
