# Heterogeneous Link Neighbor Sampling

## Goal

Close the most obvious remaining gap between the current VGL training stack and mainstream PyG/DGL link mini-batch loaders by extending link prediction sampling from homogeneous graphs to heterogeneous graphs.

## Design

Instead of introducing a second hetero-specific edge sampler, the existing link pipeline now accepts edge-typed supervision records:

```python
LinkPredictionRecord(
    graph=graph,
    src_index=author_id,
    dst_index=paper_id,
    label=1,
    edge_type=("author", "writes", "paper"),
)
```

The existing samplers are extended to respect that `edge_type`:

- `UniformNegativeLinkSampler` samples negatives from the destination node type only
- `HardNegativeLinkSampler` validates hard negatives against that destination type
- `CandidateLinkSampler` evaluates candidate destinations on the same edge type
- `LinkNeighborSampler` expands neighborhoods across all relation types, then remaps typed endpoints into the sampled local node spaces

## Batching

`LinkPredictionBatch` now supports heterogeneous sampled graphs when all records in the batch share a single supervision edge type, mirroring the common `LinkNeighborLoader` pattern in PyG and edge-wise loaders in DGL.

For heterogeneous batches it:

1. disjoint-unions sampled graphs with per-node-type offsets
2. offsets `src_index` and `dst_index` in their own node spaces
3. stores `edge_type`, `src_node_type`, and `dst_node_type`
4. optionally excludes positive supervision edges from the message-passing relation being trained

## Scope

- supports mixed supervision edge types within one heterogeneous batch (records are relation-indexed)
- heterogeneous full-graph and neighbor-sampled link prediction
- no heterogeneous `RandomLinkSplit` yet

## Validation

- sampler tests cover hetero local subgraph extraction
- loader tests cover hetero uniform negative expansion
- trainer tests cover one-epoch hetero link mini-batch training
- integration now includes `examples/hetero/link_prediction.py`
