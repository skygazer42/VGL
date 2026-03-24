# Distributed Runtime Expansion Design

## Goal

Extend the current local-first distributed foundation so it can coordinate graph structure queries as well as feature fetches. The immediate target is to let shards recover global node/edge structure and let the coordinator serve partition-local node ids, partition edge indices, and partition adjacency views through one stable surface.

## Why This Next

`vgl.distributed` already has partition metadata, deterministic local partition writing, shard loading, store adapters, and node-feature routing. The clearest remaining gap is that the runtime still behaves like a feature gather helper rather than a graph-aware distributed substrate. DGL-class systems need both data-plane pieces: feature lookup and graph structure lookup. This is a good next step because it stays local-first, reuses the existing partition payloads, and strengthens the contracts that future remote execution can preserve.

## Scope

This batch focuses on graph-structure coordination for the current local partition model:

- add shard helpers to map local ids back to global ids
- expose partition-global edge indices from a `LocalGraphShard`
- extend `LocalSamplingCoordinator` with partition node-space and partition graph queries
- keep the current feature-routing surface unchanged
- document the broader coordinator/runtime surface

## Non-Goals

This batch does not add RPC, multiprocessing, remote executors, cross-partition edge stitching, or a full DistDGL runtime. It also does not redesign partition writing or sampling plans.

## Design Notes

The partition payload already stores `n_id` and local edge indices, so global edge recovery can be derived without changing the on-disk format: map each local endpoint through the shard's node-id table. Coordinator graph queries should stay explicitly partition-scoped because the current partition writer only materializes intra-partition edges. The API should make that limitation clear while still shaping the same coordination layer that a future remote backend can implement.
