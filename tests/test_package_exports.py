from vgl import (
    AGNNConv,
    Accuracy,
    AntiSymmetricConv,
    APPNPConv,
    ARMAConv,
    BernConv,
    ChebConv,
    ClusterGCNConv,
    DAGNNConv,
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
    Graph,
    GraphBatch,
    GraphConv,
    GraphSchema,
    GraphView,
    GINConv,
    GPRGNNConv,
    H2GCNConv,
    LightGCNConv,
    LGConv,
    MixHopConv,
    LinkPredictionBatch,
    LinkPredictionRecord,
    LinkPredictionTask,
    ListDataset,
    Loader,
    LEConv,
    FullGraphSampler,
    MFConv,
    NodeSeedSubgraphSampler,
    ResGatedGraphConv,
    SampleRecord,
    SimpleConv,
    TemporalEventRecord,
    MessagePassing,
    SSGConv,
    PNAConv,
    SuperGATConv,
    Task,
    Metric,
    SGConv,
    TAGConv,
    TransformerConv,
    Trainer,
    DirGNNConv,
    NodeClassificationTask,
    GraphClassificationTask,
    GroupRevRes,
    TemporalEventPredictionTask,
    TemporalEventBatch,
    WLConvContinuous,
    global_mean_pool,
    global_sum_pool,
    global_max_pool,
    __version__,
)


def test_package_exposes_broad_vgl_root_surface():
    assert AGNNConv.__name__ == "AGNNConv"
    assert Accuracy.__name__ == "Accuracy"
    assert AntiSymmetricConv.__name__ == "AntiSymmetricConv"
    assert APPNPConv.__name__ == "APPNPConv"
    assert ARMAConv.__name__ == "ARMAConv"
    assert BernConv.__name__ == "BernConv"
    assert ChebConv.__name__ == "ChebConv"
    assert ClusterGCNConv.__name__ == "ClusterGCNConv"
    assert DAGNNConv.__name__ == "DAGNNConv"
    assert EdgeConv.__name__ == "EdgeConv"
    assert EGConv.__name__ == "EGConv"
    assert FAGCNConv.__name__ == "FAGCNConv"
    assert FiLMConv.__name__ == "FiLMConv"
    assert FeaStConv.__name__ == "FeaStConv"
    assert GeneralConv.__name__ == "GeneralConv"
    assert GATv2Conv.__name__ == "GATv2Conv"
    assert GCN2Conv.__name__ == "GCN2Conv"
    assert GatedGraphConv.__name__ == "GatedGraphConv"
    assert GENConv.__name__ == "GENConv"
    assert Graph.__name__ == "Graph"
    assert GraphBatch.__name__ == "GraphBatch"
    assert GraphConv.__name__ == "GraphConv"
    assert GPRGNNConv.__name__ == "GPRGNNConv"
    assert LinkPredictionBatch.__name__ == "LinkPredictionBatch"
    assert LEConv.__name__ == "LEConv"
    assert LightGCNConv.__name__ == "LightGCNConv"
    assert MFConv.__name__ == "MFConv"
    assert MixHopConv.__name__ == "MixHopConv"
    assert TemporalEventBatch.__name__ == "TemporalEventBatch"
    assert GINConv.__name__ == "GINConv"
    assert GraphSchema.__name__ == "GraphSchema"
    assert GraphView.__name__ == "GraphView"
    assert H2GCNConv.__name__ == "H2GCNConv"
    assert ListDataset.__name__ == "ListDataset"
    assert LGConv.__name__ == "LGConv"
    assert Loader.__name__ == "Loader"
    assert FullGraphSampler.__name__ == "FullGraphSampler"
    assert NodeSeedSubgraphSampler.__name__ == "NodeSeedSubgraphSampler"
    assert ResGatedGraphConv.__name__ == "ResGatedGraphConv"
    assert LinkPredictionRecord.__name__ == "LinkPredictionRecord"
    assert SampleRecord.__name__ == "SampleRecord"
    assert SimpleConv.__name__ == "SimpleConv"
    assert TemporalEventRecord.__name__ == "TemporalEventRecord"
    assert MessagePassing.__name__ == "MessagePassing"
    assert PNAConv.__name__ == "PNAConv"
    assert SSGConv.__name__ == "SSGConv"
    assert SuperGATConv.__name__ == "SuperGATConv"
    assert SGConv.__name__ == "SGConv"
    assert TAGConv.__name__ == "TAGConv"
    assert Task.__name__ == "Task"
    assert Metric.__name__ == "Metric"
    assert TransformerConv.__name__ == "TransformerConv"
    assert Trainer.__name__ == "Trainer"
    assert DirGNNConv.__name__ == "DirGNNConv"
    assert WLConvContinuous.__name__ == "WLConvContinuous"
    assert NodeClassificationTask.__name__ == "NodeClassificationTask"
    assert GraphClassificationTask.__name__ == "GraphClassificationTask"
    assert GroupRevRes.__name__ == "GroupRevRes"
    assert LinkPredictionTask.__name__ == "LinkPredictionTask"
    assert TemporalEventPredictionTask.__name__ == "TemporalEventPredictionTask"
    assert callable(global_mean_pool)
    assert callable(global_sum_pool)
    assert callable(global_max_pool)
    assert __version__ == "0.1.0"
