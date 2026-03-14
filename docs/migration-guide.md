# Migration Guide

## From PyG

- `Data(x=..., edge_index=..., y=...)` maps to `Graph.homo(edge_index=..., x=..., y=...)`
- `Graph.from_pyg(data)` imports PyG-style graph data
- `graph.to_pyg()` exports back to a PyG-style `Data` object
- many-graph graph classification maps cleanly to `ListDataset + Loader + GraphClassificationTask`

## From DGL

- `graph.ndata[...]` style access remains available on homogeneous graphs
- `Graph.from_dgl(dgl_graph)` imports a DGL graph
- `graph.to_dgl()` exports back to a DGL graph
- sampled graph-classification records can attach labels to sample metadata instead of mutating the graph object

## Mental Model Shift

The package keeps one internal `Graph` abstraction and treats homogeneous, heterogeneous, and temporal graphs as variations of the same core model instead of separate top-level object families.

For training, the important shift is similar: graph classification does not get its own parallel data model. Many-small-graph samples and sampled subgraphs both converge into the same `GraphBatch` contract before they reach the model or task.
