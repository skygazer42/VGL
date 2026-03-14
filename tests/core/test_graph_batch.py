import torch

from gnn import Graph
from gnn.core.batch import GraphBatch


def test_graph_batch_tracks_membership():
    g1 = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    g2 = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 4),
        y=torch.tensor([1, 0]),
    )

    batch = GraphBatch.from_graphs([g1, g2])

    assert batch.num_graphs == 2
    assert batch.graph_index.shape[0] == 4
