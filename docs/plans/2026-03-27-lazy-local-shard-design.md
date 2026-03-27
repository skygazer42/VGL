# Lazy LocalGraphShard Loading Design

## Context

VGL now lazy-loads partition-directory stores and storage-backed graph structure, but `LocalGraphShard.from_partition_dir(...)` still eagerly deserializes `part-*.pt` during construction. That leaves the local partition runtime as the remaining eager path in the distributed substrate.

For large partitioned graphs this matters immediately: creating shard objects for local training or evaluation still pays the full payload load cost up front, even when the caller only needs manifest-derived routing metadata first.

## Goals

- Make `LocalGraphShard.from_partition_dir(...)` manifest-only during construction
- Preserve current shard APIs for node/edge id mapping, feature fetches, graph access, and boundary-edge queries
- Reuse the existing partition payload parsing and lazy adapter machinery instead of creating a second on-disk format
- Preserve temporal shard metadata without requiring eager payload reconstruction

## Recommended Design

Use manifest metadata plus the existing lazy partition store cache from `vgl.distributed.store`.

### Manifest-only shard bootstrap

At construction time:

- load only `manifest.json`
- resolve the target `PartitionShard`
- derive `node_ids_by_type` from `partition.node_ranges`
- derive edge-id ownership and global/local mappings from `partition.edge_ids_by_type`
- build shard-local `GraphSchema` from partition metadata

This keeps shard creation free of `torch.load(...)` calls.

### Reuse lazy partition adapters

Construct the shard's `feature_store` and `graph_store` from the same internal lazy adapters already used by `load_partitioned_stores(...)`. They already support manifest-derived `shape(...)`, `edge_types`, `num_nodes(...)`, and cached first-access payload loading.

`Graph.from_storage(...)` can then wrap those adapters without forcing the payload to load.

### Boundary-edge data

`LocalGraphShard` still needs public `boundary_edge_data_by_type` access for the store-backed coordinator path. Extend the internal partition bundle cache so a loaded bundle retains cloned boundary-edge payloads alongside the local adapters, and expose them lazily through the shard.

### Temporal metadata

A manifest-only shard cannot recover `graph.schema.time_attr` from the payload because that would force eager loading. Persist `time_attr` into `PartitionManifest.metadata` when partition graphs are written, then reconstruct the shard schema from manifest metadata alone.

## Non-Goals

- No partition payload schema rewrite
- No eviction policy for loaded shard payloads
- No new public distributed-store type

## Verification

- Regression proving `LocalGraphShard.from_partition_dir(...)` performs zero `torch.load(...)` calls during construction
- Regression proving manifest-only shard methods (`node_ids`, local/global id mapping, edge ids) work before payload load
- Regression proving first real graph/feature/boundary access loads exactly one payload and reuses it
- Regression proving temporal shards still expose `schema.time_attr` through the lazy path
- Fresh focused and full `pytest` runs before merge
