from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, GraphSAINTNodeSampler, ListDataset
from vgl.engine import Trainer
from vgl.tasks import NodeClassificationTask
from vgl.transforms import Compose, RandomNodeSplit
from vgl import KarateClubDataset


class TinySaintModel(nn.Module):
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
        transform = Compose([RandomNodeSplit(num_train_per_class=2, num_val=4, num_test=4, seed=0)])
        graph = KarateClubDataset(root=Path(tmp_dir) / "datasets", transform=transform)[0]
        train_loader = DataLoader(_seed_dataset(graph, "train_mask"), GraphSAINTNodeSampler(num_sampled_nodes=12, seed=0), batch_size=8)
        val_loader = DataLoader(_seed_dataset(graph, "val_mask"), GraphSAINTNodeSampler(num_sampled_nodes=12, seed=1), batch_size=8)
        test_loader = DataLoader(_seed_dataset(graph, "test_mask"), GraphSAINTNodeSampler(num_sampled_nodes=12, seed=2), batch_size=8)

        trainer = Trainer(
            model=TinySaintModel(graph.x.size(1), int(graph.y.max().item()) + 1),
            task=NodeClassificationTask(target="y", split=("train_mask", "val_mask", "test_mask"), metrics=["accuracy"]),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=1,
        )
        trainer.fit(train_loader, val_data=val_loader)
        result = trainer.test(test_loader)
        print(result)
        return result


if __name__ == "__main__":
    main()
