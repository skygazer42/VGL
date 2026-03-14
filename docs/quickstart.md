# Quickstart

`gnn` is a PyTorch-first graph learning package with one core `Graph` abstraction.

The smallest workflow is:

1. Build a `Graph`
2. Define a `Task`
3. Build a PyTorch model
4. Train it with `Trainer`

For a homogeneous graph:

```python
graph = Graph.homo(edge_index=edge_index, x=x, y=y, train_mask=train_mask)
task = NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask"))
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-3, max_epochs=10)
trainer.fit(graph)
```

For graph classification over many small graphs:

```python
samples = [
    SampleRecord(graph=graph_a, metadata={}, sample_id="a"),
    SampleRecord(graph=graph_b, metadata={}, sample_id="b"),
]
loader = Loader(
    dataset=ListDataset(samples),
    sampler=FullGraphSampler(),
    batch_size=2,
    label_source="graph",
    label_key="y",
)
task = GraphClassificationTask(target="y", label_source="graph")
trainer = Trainer(model=model, task=task, optimizer=torch.optim.Adam, lr=1e-3, max_epochs=10)
trainer.fit(loader)
```

For graph classification from sampled subgraphs of a larger graph:

```python
dataset = ListDataset([
    (source_graph, {"seed": 1, "label": 1, "sample_id": "s1"}),
    (source_graph, {"seed": 2, "label": 0, "sample_id": "s2"}),
])
loader = Loader(
    dataset=dataset,
    sampler=NodeSeedSubgraphSampler(),
    batch_size=2,
    label_source="metadata",
    label_key="label",
)
task = GraphClassificationTask(target="label", label_source="metadata")
trainer.fit(loader)
```
