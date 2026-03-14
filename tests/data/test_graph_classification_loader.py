import torch

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import SampleRecord
from vgl.data.sampler import FullGraphSampler


def test_loader_collates_graph_samples_with_metadata_labels():
    samples = [
        SampleRecord(
            graph=Graph.homo(
                edge_index=torch.tensor([[0], [1]]),
                x=torch.randn(2, 4),
                y=torch.tensor([1]),
            ),
            metadata={"label": 1},
            sample_id="a",
        ),
        SampleRecord(
            graph=Graph.homo(
                edge_index=torch.tensor([[0], [1]]),
                x=torch.randn(2, 4),
                y=torch.tensor([0]),
            ),
            metadata={"label": 0},
            sample_id="b",
        ),
    ]

    dataset = ListDataset(samples)
    loader = Loader(
        dataset=dataset,
        sampler=FullGraphSampler(),
        batch_size=2,
        label_source="metadata",
        label_key="label",
    )

    batch = next(iter(loader))

    assert batch.num_graphs == 2
    assert torch.equal(batch.labels, torch.tensor([1, 0]))

