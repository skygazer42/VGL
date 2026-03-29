from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import ClusterData, ClusterLoader
from vgl.transforms import Compose, RandomNodeSplit
from vgl import KarateClubDataset


class TinyClusterClassifier(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph):
        return self.linear(graph.x)


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        graph = KarateClubDataset(
            root=Path(tmp_dir) / "datasets",
            transform=Compose([RandomNodeSplit(num_train_per_class=2, num_val=4, num_test=4, seed=0)]),
        )[0]
        clusters = ClusterData(graph, num_parts=4, seed=0)
        loader = ClusterLoader(clusters, batch_size=2)

        model = TinyClusterClassifier(graph.x.size(1), int(graph.y.max().item()) + 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for batch in loader:
            optimizer.zero_grad()
            cluster_losses = []
            for subgraph in batch.graphs:
                logits = model(subgraph)
                mask = subgraph.train_mask
                if int(mask.sum()) == 0:
                    continue
                cluster_losses.append(F.cross_entropy(logits[mask], subgraph.y[mask]))
            if not cluster_losses:
                continue
            loss = torch.stack(cluster_losses).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        result = {"num_clusters": len(clusters), "losses": losses}
        print(result)
        return result


if __name__ == "__main__":
    main()
