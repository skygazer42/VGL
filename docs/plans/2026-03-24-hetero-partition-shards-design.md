# Heterogeneous Partition Shards Design

## Context

VGL's local distributed runtime now handles homogeneous graphs, temporal homogeneous graphs, and single-node-type multi-relation graphs, but it still rejects true heterogeneous partitions because ownership, node-id routing, and shard reload all assume one global node namespace. That leaves a clear substrate gap versus DGL-class systems: the graph core, samplers, tasks, and structure ops already understand typed node spaces, while the local partition path still collapses to `{"node"}`.

## Options Considered

1. Keep the current manifest shape and encode per-type ranges only in opaque metadata.
2. Extend manifest and shard APIs additively with typed node ownership while preserving the single-type fast path.
3. Introduce a new hetero-only partition format and a separate coordinator.

Option 2 is the right next slice. Option 1 hides core routing data behind unstructured metadata and would make typed ownership harder to validate. Option 3 duplicates nearly the entire local runtime surface. An additive manifest/shard extension lets homogeneous callers keep working while the hetero path becomes explicit and testable.

## Recommended Design

Partition ownership should stay partition-local and contiguous, but become per node type. `PartitionManifest` will gain typed node counts, and each `PartitionShard` will expose typed node ranges. For homogeneous graphs this collapses to the existing `("node", 0:N)` model; for heterogeneous graphs each partition owns one contiguous range per node type, such as `author: [0, 2)` and `paper: [0, 3)` in partition 0. `owner(node_id, node_type="node")` stays additive and becomes the central routing primitive.

`write_partitioned_graph(...)` should stop rejecting multi-node-type graphs. For each partition, slice node features independently by node type, keep global node ids per type in the payload, and retain only edges whose source and destination owners both equal the current partition id for their respective types. Local edge indices are then relabeled against the partition-local source and destination node spaces, preserving all edge features and `time_attr` when present.

`LocalGraphShard.from_partition_dir(...)` should reconstruct typed `FeatureStore` and `InMemoryGraphStore` instances for every node and edge type in the serialized payload. The shard will maintain per-type global-id tables plus reverse lookup maps so `global_to_local(...)`, `local_to_global(...)`, and `global_edge_index(...)` can all become node-type aware without breaking the current homogeneous API.

## Coordinator and API Surface

`LocalSamplingCoordinator` should keep its current homogeneous ergonomics while gaining typed routing. `route_node_ids(node_ids, node_type="node")` and `partition_node_ids(partition_id, node_type="node")` stay additive. `fetch_node_features(key, node_ids)` can infer the node type from the feature key, so existing single-type callers remain unchanged and hetero callers do not need an extra argument when fetching typed node features. Partition edge queries already carry `edge_type`, so they only need to delegate to the shard's typed global-edge reconstruction.

The immediate integration target is hetero node classification on partition-local shard graphs. Each shard graph can stay fully hetero, existing `NodeNeighborSampler` already accepts `metadata["node_type"]`, and the coordinator only needs to route the seed node type requested by the batch metadata. This keeps the scope local-first and avoids redesigning link-routing or cross-partition stitching in the same batch.

## Non-Goals

This batch does not add remote executors, RPC, multiprocessing, or cross-partition edge stitching. It also does not attempt full DistDGL semantics or global hetero sampling across partitions. Edges remain partition-local exactly as in the current runtime; the only change is that node ownership and local/global mapping become typed.

## Testing Strategy

Add red-green coverage in `tests/distributed/test_partition_metadata.py` for typed node ownership. Extend `tests/distributed/test_partition_writer.py` to verify heterogeneous partition payloads preserve per-type node ids, typed node features, relation-local edge features, and partition metadata. Extend `tests/distributed/test_local_shard.py` and `tests/distributed/test_sampling_coordinator.py` so typed node routing, typed feature fetch, and relation-global edge reconstruction are all exercised through the public runtime API.

Finish with an integration regression in `tests/integration/test_foundation_partition_local.py` that runs a hetero node-classification batch through the existing loader/trainer stack using partitioned shards and coordinator-routed paper-node features. That gives a real substrate proof that the new typed partition model works through the current public training path.
