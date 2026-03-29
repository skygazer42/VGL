from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, ListDataset, NodeNeighborSampler
from vgl.engine import Trainer
from vgl.tasks import NodeClassificationTask
from vgl.transforms import Compose, NormalizeFeatures
from vgl import PlanetoidDataset


class TinyPlanetoidModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, batch):
        graph = batch.graph if hasattr(batch, "graph") else batch
        return self.linear(graph.x)


def _seed_dataset(graph, mask_name):
    seeds = graph.ndata[mask_name].nonzero(as_tuple=False).view(-1).tolist()
    return ListDataset([(graph, {"seed": int(seed), "sample_id": f"{mask_name}:{seed}"}) for seed in seeds])


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = PlanetoidDataset(
            root=Path(tmp_dir) / "datasets",
            name="Cora",
            transform=Compose([NormalizeFeatures()]),
        )
        graph = dataset[0]
        train_loader = DataLoader(_seed_dataset(graph, "train_mask"), NodeNeighborSampler([15, 10]), batch_size=256)
        val_loader = DataLoader(_seed_dataset(graph, "val_mask"), NodeNeighborSampler([15, 10]), batch_size=512)
        test_loader = DataLoader(_seed_dataset(graph, "test_mask"), NodeNeighborSampler([15, 10]), batch_size=512)

        model = TinyPlanetoidModel(graph.x.size(1), int(graph.y.max().item()) + 1)
        trainer = Trainer(
            model=model,
            task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask"), metrics=["accuracy"]),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=1,
            monitor="val_accuracy",
        )
        trainer.fit(train_loader, val_data=val_loader)
        result = trainer.test(test_loader)
        print(result)
        return result


if __name__ == "__main__":
    main()
