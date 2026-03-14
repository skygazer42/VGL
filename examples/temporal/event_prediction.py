from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gnn import Graph
from gnn.data.dataset import ListDataset
from gnn.data.loader import Loader
from gnn.data.sample import TemporalEventRecord
from gnn.data.sampler import FullGraphSampler
from gnn.train.tasks import TemporalEventPredictionTask
from gnn.train.trainer import Trainer


class TinyTemporalEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 2)

    def forward(self, batch):
        src_x = batch.graph.x[batch.src_index]
        dst_x = batch.graph.x[batch.dst_index]
        history_counts = torch.tensor(
            [batch.history_graph(i).edge_index.size(1) for i in range(batch.labels.size(0))],
            dtype=src_x.dtype,
        ).unsqueeze(-1)
        return self.linear(torch.cat([src_x, dst_x, history_counts], dim=-1))


def build_demo_graph():
    return Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )


def build_demo_loader():
    graph = build_demo_graph()
    samples = [
        TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3, label=1),
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5, label=0),
    ]
    return Loader(
        dataset=ListDataset(samples),
        sampler=FullGraphSampler(),
        batch_size=2,
    )


def main():
    trainer = Trainer(
        model=TinyTemporalEventModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    result = trainer.fit(build_demo_loader())
    print(result)
    return result


if __name__ == "__main__":
    main()
