# Core Concepts

## Graph

`Graph` is the canonical graph object in this package. Homogeneous graphs are a special case of the same abstraction used for heterogeneous and temporal graphs.

## GraphView

`GraphView` is a lightweight projection over an existing graph, used for operations such as `snapshot()` and `window()`.

## GraphBatch

`GraphBatch` groups multiple graphs into one training input and tracks node-to-graph membership.

For graph classification it also carries:

- `graph_ptr`
- `labels`
- `metadata`

## MessagePassing

`MessagePassing` is the low-level neural primitive for graph convolutions. `GCNConv`, `SAGEConv`, and `GATConv` build on top of it.

## SampleRecord

`SampleRecord` is the structured pre-collation unit for graph classification. It lets the framework carry:

- a `graph`
- sample metadata
- sample identity
- optional source graph information

This is what makes many-small-graph and sampled-subgraph inputs converge on the same batch contract.

## Readout

Graph classification uses graph-level pooling to convert node representations into graph representations.

The package currently exposes:

- `global_mean_pool`
- `global_sum_pool`
- `global_max_pool`

## Trainer and Task

`Task` defines the supervision contract. `Trainer` runs the optimization loop without taking ownership of the core graph abstraction.

The current training layer supports:

- node classification
- graph classification with labels from graph objects
- graph classification with labels from sample metadata
