import torch
from torch import nn

from vgl import (
    AGNNConv,
    APPNPConv,
    ARMAConv,
    AntiSymmetricConv,
    BernConv,
    ChebConv,
    ClusterGCNConv,
    DAGNNConv,
    DirGNNConv,
    EdgeConv,
    EGConv,
    FAGCNConv,
    FiLMConv,
    FeaStConv,
    GeneralConv,
    GATv2Conv,
    GCN2Conv,
    GatedGraphConv,
    GENConv,
    GINConv,
    GPRGNNConv,
    Graph,
    GraphConv,
    GroupRevRes,
    H2GCNConv,
    LEConv,
    LGConv,
    LightGCNConv,
    MFConv,
    MixHopConv,
    NodeClassificationTask,
    PNAConv,
    ResGatedGraphConv,
    SGConv,
    SSGConv,
    SimpleConv,
    SuperGATConv,
    TAGConv,
    TransformerConv,
    Trainer,
    WLConvContinuous,
)


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
        y=torch.tensor([0, 1, 0]),
        train_mask=torch.tensor([True, True, True]),
        val_mask=torch.tensor([True, True, True]),
        test_mask=torch.tensor([True, True, True]),
    )


def _model(conv):
    class TinyModel(nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op
            out_channels = getattr(op, "out_channels", None)
            if out_channels is None:
                out_channels = getattr(op, "channels", 4)
            heads = getattr(op, "heads", 1)
            concat = getattr(op, "concat", False)
            hidden = out_channels * heads if concat else out_channels
            self.head = nn.Linear(hidden, 2)

        def forward(self, graph):
            if isinstance(self.op, GCN2Conv):
                return self.head(self.op(graph, x0=graph.x))
            return self.head(self.op(graph))

    return TinyModel(conv)


def test_new_homo_convs_plug_into_training_loop():
    convs = [
        GINConv(in_channels=4, out_channels=4),
        GATv2Conv(in_channels=4, out_channels=4, heads=2, concat=False),
        APPNPConv(in_channels=4, out_channels=4, steps=2, alpha=0.1),
        TAGConv(in_channels=4, out_channels=4, k=2),
        SGConv(in_channels=4, out_channels=4, k=2),
        ChebConv(in_channels=4, out_channels=4, k=3),
        AGNNConv(channels=4),
        LightGCNConv(),
        LGConv(),
        FAGCNConv(channels=4, eps=0.1),
        ARMAConv(channels=4, stacks=2, layers=2, alpha=0.1),
        GPRGNNConv(channels=4, steps=3, alpha=0.1),
        MixHopConv(in_channels=4, out_channels=4, powers=(0, 1, 2)),
        BernConv(channels=4, steps=3),
        SSGConv(channels=4, steps=3, alpha=0.1),
        DAGNNConv(channels=4, steps=3),
        GCN2Conv(channels=4, alpha=0.1, theta=1.0, layer=1),
        GraphConv(in_channels=4, out_channels=4),
        H2GCNConv(in_channels=4, out_channels=4),
        EGConv(in_channels=4, out_channels=4, aggregators=("sum", "mean", "max")),
        LEConv(in_channels=4, out_channels=4),
        ResGatedGraphConv(in_channels=4, out_channels=4),
        GatedGraphConv(channels=4, steps=2),
        ClusterGCNConv(in_channels=4, out_channels=4, diag_lambda=0.0),
        GENConv(in_channels=4, out_channels=4, aggr="softmax", beta=1.0),
        FiLMConv(in_channels=4, out_channels=4),
        SimpleConv(aggr="mean"),
        EdgeConv(in_channels=4, out_channels=4, aggr="max"),
        FeaStConv(in_channels=4, out_channels=4, heads=2),
        MFConv(in_channels=4, out_channels=4, max_degree=4),
        PNAConv(
            in_channels=4,
            out_channels=4,
            aggregators=("sum", "mean", "max"),
            scalers=("identity", "amplification", "attenuation"),
        ),
        GeneralConv(
            in_channels=4,
            out_channels=4,
            aggr="add",
            heads=2,
            attention=True,
        ),
        AntiSymmetricConv(channels=4, num_iters=2, epsilon=0.1, gamma=0.1),
        TransformerConv(in_channels=4, out_channels=4, heads=2, concat=False, beta=True),
        WLConvContinuous(),
        SuperGATConv(in_channels=4, out_channels=4, heads=2, concat=False),
        DirGNNConv(GraphConv(in_channels=4, out_channels=4), alpha=0.5, root_weight=True),
        GroupRevRes(LGConv(), num_groups=2),
    ]

    for conv in convs:
        trainer = Trainer(
            model=_model(conv),
            task=NodeClassificationTask(
                target="y",
                split=("train_mask", "val_mask", "test_mask"),
                metrics=["accuracy"],
            ),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=1,
        )
        history = trainer.fit(_graph(), val_data=_graph())

        assert history["epochs"] == 1
        assert "loss" in history["train"][0]
