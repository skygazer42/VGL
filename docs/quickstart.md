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
