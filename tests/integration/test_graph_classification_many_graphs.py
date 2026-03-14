import torch
from torch import nn

from gnn import Graph
from gnn.data.dataset import ListDataset
from gnn.data.loader import Loader
from gnn.data.sample import SampleRecord
from gnn.data.sampler import FullGraphSampler
from gnn.nn.readout import global_mean_pool
from gnn.train.tasks import GraphClassificationTask
from gnn.train.trainer import Trainer


class TinyGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, batch):
        x = torch.cat([graph.x for graph in batch.graphs], dim=0)
        node_repr = self.encoder(x)
        graph_repr = global_mean_pool(node_repr, batch.graph_index)
        return self.head(graph_repr)


def test_many_graph_graph_classification_runs():
    samples = [
        SampleRecord(
            graph=Graph.homo(
                edge_index=torch.tensor([[0], [1]]),
                x=torch.randn(2, 4),
                y=torch.tensor([1]),
            ),
            metadata={},
            sample_id="g1",
        ),
        SampleRecord(
            graph=Graph.homo(
                edge_index=torch.tensor([[0], [1]]),
                x=torch.randn(2, 4),
                y=torch.tensor([0]),
            ),
            metadata={},
            sample_id="g2",
        ),
    ]
    dataset = ListDataset(samples)
    loader = Loader(
        dataset=dataset,
        sampler=FullGraphSampler(),
        batch_size=2,
        label_source="graph",
        label_key="y",
    )
    trainer = Trainer(
        model=TinyGraphClassifier(),
        task=GraphClassificationTask(target="y", label_source="graph"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(loader)

    assert result["epochs"] == 1
