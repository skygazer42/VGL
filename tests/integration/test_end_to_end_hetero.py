import torch
from torch import nn

from examples.hetero.link_prediction import TinyHeteroLinkPredictor as DemoHeteroLinkPredictor
from examples.hetero.link_prediction import TinyMixedHeteroLinkPredictor as DemoMixedHeteroLinkPredictor
from examples.hetero.link_prediction import build_demo_loaders as build_hetero_link_demo_loaders
from examples.hetero.link_prediction import build_mixed_demo_loaders as build_mixed_hetero_link_demo_loaders
from examples.hetero.node_classification import build_demo_loaders
from vgl.engine import Trainer
from vgl.graph import Graph
from vgl.nn import HANConv
from vgl.nn import HEATConv
from vgl.nn import RGATConv
from vgl.tasks import LinkPredictionTask
from vgl.tasks import NodeClassificationTask


def _hetero_graph():
    return Graph.hetero(
        nodes={
            "paper": {
                "x": torch.randn(2, 4),
                "y": torch.tensor([0, 1]),
                "train_mask": torch.tensor([True, True]),
                "val_mask": torch.tensor([True, True]),
                "test_mask": torch.tensor([True, True]),
            },
            "author": {
                "x": torch.randn(2, 4),
            },
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "edge_attr": torch.randn(2, 2),
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 0], [0, 1]]),
                "edge_attr": torch.randn(2, 2),
            },
        },
    )


class TinyHeteroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, graph):
        if hasattr(graph, "graph"):
            graph = graph.graph
        return self.linear(graph.nodes["paper"].x)


def test_end_to_end_hetero_training_runs():
    graph = _hetero_graph()
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        node_type="paper",
    )
    trainer = Trainer(model=TinyHeteroModel(), task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)

    result = trainer.fit(graph)

    assert result["epochs"] == 1


def test_end_to_end_hetero_neighbor_sampled_training_runs():
    train_loader, val_loader, test_loader = build_demo_loaders()
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        node_type="paper",
        metrics=["accuracy"],
    )
    trainer = Trainer(model=TinyHeteroModel(), task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)

    history = trainer.fit(train_loader, val_data=val_loader)
    result = trainer.test(test_loader)

    assert history["epochs"] == 1
    assert "accuracy" in result


def test_end_to_end_hetero_training_runs_with_han_conv():
    class TinyHANModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = HANConv(
                in_channels=4,
                out_channels=4,
                node_types=("author", "paper"),
                relation_types=(
                    ("author", "writes", "paper"),
                    ("paper", "written_by", "author"),
                ),
            )
            self.head = nn.Linear(4, 2)

        def forward(self, graph):
            out = self.conv(graph)
            return self.head(out["paper"])

    graph = _hetero_graph()
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        node_type="paper",
    )
    trainer = Trainer(model=TinyHANModel(), task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)

    result = trainer.fit(graph)

    assert result["epochs"] == 1


def test_end_to_end_hetero_training_runs_with_heat_conv():
    class TinyHEATModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = HEATConv(
                in_channels=4,
                out_channels=4,
                node_types=("author", "paper"),
                relation_types=(
                    ("author", "writes", "paper"),
                    ("paper", "written_by", "author"),
                ),
                edge_channels=2,
                heads=2,
            )
            self.head = nn.Linear(4, 2)

        def forward(self, graph):
            out = self.conv(graph)
            return self.head(out["paper"])

    graph = _hetero_graph()
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        node_type="paper",
    )
    trainer = Trainer(model=TinyHEATModel(), task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)

    result = trainer.fit(graph)

    assert result["epochs"] == 1


def test_end_to_end_hetero_training_runs_with_rgat_conv():
    class TinyRGATModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = RGATConv(
                in_channels=4,
                out_channels=4,
                node_types=("author", "paper"),
                relation_types=(
                    ("author", "writes", "paper"),
                    ("paper", "written_by", "author"),
                ),
                edge_channels=2,
                heads=2,
            )
            self.head = nn.Linear(4, 2)

        def forward(self, graph):
            out = self.conv(graph)
            return self.head(out["paper"])

    graph = _hetero_graph()
    task = NodeClassificationTask(
        target="y",
        split=("train_mask", "val_mask", "test_mask"),
        node_type="paper",
    )
    trainer = Trainer(model=TinyRGATModel(), task=task, optimizer=torch.optim.Adam, lr=1e-2, max_epochs=1)

    result = trainer.fit(graph)

    assert result["epochs"] == 1


def test_end_to_end_hetero_link_prediction_runs():
    train_loader, val_loader, test_loader = build_hetero_link_demo_loaders()
    trainer = Trainer(
        model=DemoHeteroLinkPredictor(),
        task=LinkPredictionTask(target="label", metrics=["mrr", "filtered_mrr", "filtered_hits@1"]),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(train_loader, val_data=val_loader)
    result = trainer.test(test_loader)

    assert history["epochs"] == 1
    assert "mrr" in result


def test_end_to_end_mixed_hetero_link_prediction_runs():
    train_loader, val_loader, test_loader = build_mixed_hetero_link_demo_loaders()
    trainer = Trainer(
        model=DemoMixedHeteroLinkPredictor(),
        task=LinkPredictionTask(target="label", metrics=["mrr", "filtered_mrr", "filtered_hits@1"]),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(train_loader, val_data=val_loader)
    result = trainer.test(test_loader)

    assert history["epochs"] == 1
    assert "mrr" in result
