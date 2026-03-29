from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, FullGraphSampler, ListDataset, SampleRecord
from vgl.engine import Trainer
from vgl.nn.readout import global_mean_pool
from vgl.tasks import GraphClassificationTask
from vgl import TUDataset


class TinyTuClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, out_channels)

    def forward(self, batch):
        node_features = torch.cat([graph.x for graph in batch.graphs], dim=0)
        node_repr = self.encoder(node_features)
        graph_repr = global_mean_pool(node_repr, batch.graph_index)
        return self.head(graph_repr)


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = TUDataset(root=Path(tmp_dir) / "datasets", name="MUTAG")
        samples = [
            SampleRecord(graph=dataset[index], metadata={}, sample_id=f"mutag:{index}")
            for index in range(len(dataset))
        ]
        loader = DataLoader(
            dataset=ListDataset(samples),
            sampler=FullGraphSampler(),
            batch_size=min(32, len(samples)),
            label_source="graph",
            label_key="y",
        )
        model = TinyTuClassifier(dataset[0].x.size(1), hidden_channels=16, out_channels=2)
        trainer = Trainer(
            model=model,
            task=GraphClassificationTask(target="y", label_source="graph"),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=1,
        )
        result = trainer.fit(loader)
        print(result)
        return result


if __name__ == "__main__":
    main()
