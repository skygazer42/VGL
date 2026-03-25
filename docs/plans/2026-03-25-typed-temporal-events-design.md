# Typed Temporal Events Design

## Problem

The temporal event pipeline already handled homogeneous temporal graphs, but it broke down on typed heterogeneous temporal data. `TemporalEventRecord` could not carry `edge_type`, `TemporalEventBatch` could not represent relation metadata the way `LinkPredictionBatch` already did, and `TemporalNeighborSampler` still explicitly assumed homogeneous temporal graphs. That left multi-relation or multi-node-type temporal workloads outside the promised training path.

## Goals

- Preserve the existing homogeneous temporal event path
- Add typed temporal record and batch metadata without inventing a second batch contract
- Keep strict-history sampling relation-local for heterogeneous temporal graphs
- Materialize fetched node and edge features back into sampled hetero temporal subgraphs with the correct per-type ids
- Document the public contract clearly

## Design

### 1. Mirror the typed link-prediction contract for temporal batches

`TemporalEventRecord` now accepts optional `edge_type`. `TemporalEventBatch` resolves per-record relation ids and exposes `edge_types`, `edge_type_index`, `edge_type`, `src_node_type`, and `dst_node_type` using the same conventions as `LinkPredictionBatch`. This keeps typed temporal models aligned with the existing link-prediction surface instead of creating a special-case API.

### 2. Preserve single-relation edge names and temporal metadata during graph batching

Multi-graph batching for single-relation graphs must not silently rewrite relation names to the default homogeneous edge type or drop `schema.time_attr`. The batching helper therefore preserves the original single relation and, when needed, rebuilds the batched graph through `Graph.temporal(...)` or `Graph.hetero(...)` instead of forcing `Graph.homo(...)`.

### 3. Make temporal neighbor sampling relation-local on hetero graphs

For typed heterogeneous temporal graphs, strict-history extraction is scoped to the selected `edge_type`. The sampler filters candidate history edges by that relation and timestamp, expands relation-local frontiers, builds a relation-local subgraph, and carries the selected `edge_type` back onto the sampled `TemporalEventRecord`. Homogeneous temporal behavior stays on the existing flat-tensor path.

### 4. Use per-type feature ids when materializing sampled hetero temporal graphs

When temporal sampling appends feature-fetch stages, hetero sampled graphs must fetch by `node_ids_by_type` and `edge_ids_by_type` rather than a flat `n_id` or `e_id`. The resulting fetched tensors are then overlaid onto the sampled graph stores so typed temporal models can read features directly from the materialized subgraph.

## Non-goals

- No redesign of `Trainer` or `TemporalEventPredictionTask`
- No whole-graph heterogeneous temporal sampling across all relations simultaneously
- No changes to the homogeneous temporal model contract

## Verification

- Red-green temporal batch, transfer, sampler, loader, and trainer regressions for typed heterogeneous temporal data
- Fresh focused temporal regression suite after implementation
- Fresh full repository regression before merge
