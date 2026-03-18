# Heterogeneous Node Neighbor Sampling

## Goal

Extend the new node mini-batch path from homogeneous graphs to heterogeneous node classification without introducing a second sampler or batch abstraction.

## Design

Instead of adding a `HeteroNodeNeighborSampler`, the existing `NodeNeighborSampler` now also supports:

```python
(graph, {"seed": node_id, "node_type": "paper"})
```

for heterogeneous graphs.

For each seed node it:

1. expands hop-wise neighborhoods across all incident relation types
2. keeps visited nodes grouped by node type
3. builds a hetero subgraph with per-type `n_id` and per-edge-type `e_id`
4. remaps the supervision seed into the local index space of its target node type

## Batching

`NodeBatch` now supports two batching modes:

- homogeneous sampled subgraphs via the previous disjoint-union path
- heterogeneous sampled subgraphs via per-node-type offsets and per-edge-type edge-index shifting

This preserves the existing trainer contract:

- model sees `batch.graph`
- task reads `batch.seed_index`

For hetero node classification, the task's configured `node_type` determines which node store the seed indices apply to.

## Scope

- heterogeneous node classification mini-batching
- no heterogeneous link sampler yet
- no temporal sampling changes in this step

## Validation

- sampler unit tests cover hetero local subgraph extraction
- batch tests cover hetero `NodeBatch` disjoint-union construction
- end-to-end hetero node classification now runs with neighbor-sampled loaders
