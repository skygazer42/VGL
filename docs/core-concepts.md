# Core Concepts

## Graph

`Graph` is the canonical graph object in this package. Homogeneous graphs are a special case of the same abstraction used for heterogeneous and temporal graphs.

## GraphView

`GraphView` is a lightweight projection over an existing graph, used for operations such as `snapshot()` and `window()`.

## GraphBatch

`GraphBatch` groups multiple graphs into one training input and tracks node-to-graph membership.

## MessagePassing

`MessagePassing` is the low-level neural primitive for graph convolutions. `GCNConv`, `SAGEConv`, and `GATConv` build on top of it.

## Trainer and Task

`Task` defines the supervision contract. `Trainer` runs the optimization loop without taking ownership of the core graph abstraction.
