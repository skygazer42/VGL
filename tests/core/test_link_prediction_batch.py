import pytest
import torch

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch
from vgl.data.sample import LinkPredictionRecord


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )


def test_link_prediction_batch_tracks_fields():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, sample_id="p"),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, sample_id="n"),
        ]
    )

    assert batch.graph is graph
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))
    assert batch.metadata == [{}, {}]


def test_link_prediction_batch_rejects_empty_records():
    with pytest.raises(ValueError, match="at least one record"):
        LinkPredictionBatch.from_records([])


def test_link_prediction_batch_batches_mixed_graphs_into_a_disjoint_union():
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=_graph(), src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=_graph(), src_index=1, dst_index=2, label=0),
        ]
    )

    assert torch.equal(batch.src_index, torch.tensor([0, 4]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 5]))
    assert torch.equal(batch.labels, torch.tensor([1.0, 0.0]))
    assert batch.graph.x.size(0) == 6
    assert torch.equal(batch.graph.edge_index, torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]))


def test_link_prediction_batch_rejects_out_of_range_indices():
    graph = _graph()

    with pytest.raises(ValueError, match="node range"):
        LinkPredictionBatch.from_records(
            [LinkPredictionRecord(graph=graph, src_index=3, dst_index=1, label=1)]
        )


def test_link_prediction_batch_rejects_non_binary_labels():
    graph = _graph()

    with pytest.raises(ValueError, match="binary 0/1"):
        LinkPredictionBatch.from_records(
            [LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=2)]
        )


def test_link_prediction_batch_can_exclude_seed_edges_from_message_passing_graph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 0]]),
        x=torch.randn(3, 4),
    )
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                metadata={"exclude_seed_edges": True},
            ),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
        ]
    )

    assert batch.graph is not graph
    assert torch.equal(batch.graph.edge_index, torch.tensor([[1, 1], [2, 0]]))
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 0]))


def test_link_prediction_batch_tracks_filter_mask():
    graph = _graph()
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, query_id="q0"),
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=2,
                label=0,
                query_id="q0",
                filter_ranking=True,
            ),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0, query_id="q0"),
        ]
    )

    assert torch.equal(batch.query_index, torch.tensor([0, 0, 0]))
    assert torch.equal(batch.filter_mask, torch.tensor([False, True, False]))


def test_link_prediction_batch_batches_hetero_graphs_for_single_edge_type():
    edge_type = ("author", "writes", "paper")
    g1 = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    g2 = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(1, 4)},
            "paper": {"x": torch.randn(2, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0], [1]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1], [0]])},
        },
    )

    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=g1, src_index=0, dst_index=1, label=1, edge_type=edge_type),
            LinkPredictionRecord(graph=g2, src_index=0, dst_index=1, label=0, edge_type=edge_type),
        ]
    )

    assert batch.edge_type == edge_type
    assert batch.src_node_type == "author"
    assert batch.dst_node_type == "paper"
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 4]))
    assert torch.equal(
        batch.graph.edges[edge_type].edge_index,
        torch.tensor([[0, 1, 2], [1, 2, 4]]),
    )


def test_link_prediction_batch_can_exclude_seed_and_reverse_edges_for_hetero_records():
    edge_type = ("author", "writes", "paper")
    reverse_edge_type = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(3, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            reverse_edge_type: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
        },
    )
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                edge_type=edge_type,
                reverse_edge_type=reverse_edge_type,
                metadata={"exclude_seed_edges": True},
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=1,
                dst_index=2,
                label=0,
                edge_type=edge_type,
                reverse_edge_type=reverse_edge_type,
            ),
        ]
    )

    assert torch.equal(
        batch.graph.edges[edge_type].edge_index,
        torch.tensor([[1], [2]]),
    )
    assert torch.equal(
        batch.graph.edges[reverse_edge_type].edge_index,
        torch.tensor([[2], [1]]),
    )


def test_link_prediction_batch_supports_mixed_hetero_edge_types():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 2], [0, 1]])},
            cites: {"edge_index": torch.tensor([[0, 2], [2, 3]])},
        },
    )

    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1, edge_type=writes),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=3, label=0, edge_type=cites),
        ]
    )

    assert batch.edge_type is None
    assert batch.edge_types == (writes, cites)
    assert torch.equal(batch.edge_type_index, torch.tensor([0, 1]))
    assert torch.equal(batch.src_index, torch.tensor([0, 2]))
    assert torch.equal(batch.dst_index, torch.tensor([1, 3]))


def test_link_prediction_batch_excludes_seed_edges_for_each_relation_in_mixed_hetero_batch():
    writes = ("author", "writes", "paper")
    reverse_writes = ("paper", "written_by", "author")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            reverse_writes: {"edge_index": torch.tensor([[1, 2], [0, 1]])},
            cites: {"edge_index": torch.tensor([[0, 2], [2, 3]])},
        },
    )
    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(
                graph=graph,
                src_index=0,
                dst_index=1,
                label=1,
                edge_type=writes,
                reverse_edge_type=reverse_writes,
                metadata={"exclude_seed_edges": True},
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=2,
                dst_index=3,
                label=1,
                edge_type=cites,
                metadata={"exclude_seed_edges": True},
            ),
            LinkPredictionRecord(
                graph=graph,
                src_index=1,
                dst_index=2,
                label=0,
                edge_type=writes,
            ),
        ]
    )

    assert torch.equal(batch.graph.edges[writes].edge_index, torch.tensor([[1], [2]]))
    assert torch.equal(batch.graph.edges[reverse_writes].edge_index, torch.tensor([[2], [1]]))
    assert torch.equal(batch.graph.edges[cites].edge_index, torch.tensor([[0], [2]]))


def test_link_prediction_batch_batches_block_layers_across_records():
    g1 = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        x=torch.randn(4, 4),
        n_id=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        edge_data={"e_id": torch.tensor([100, 101, 102], dtype=torch.long)},
    )
    g2 = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        x=torch.randn(4, 4),
        n_id=torch.tensor([20, 21, 22, 23], dtype=torch.long),
        edge_data={"e_id": torch.tensor([200, 201, 202], dtype=torch.long)},
    )
    g1_blocks = [g1.to_block(torch.tensor([1, 2, 3], dtype=torch.long)), g1.to_block(torch.tensor([2, 3], dtype=torch.long))]
    g2_blocks = [g2.to_block(torch.tensor([1, 2, 3], dtype=torch.long)), g2.to_block(torch.tensor([2, 3], dtype=torch.long))]

    batch = LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=g1, src_index=1, dst_index=2, label=1, blocks=g1_blocks),
            LinkPredictionRecord(graph=g2, src_index=1, dst_index=2, label=0, blocks=g2_blocks),
        ]
    )

    assert batch.blocks is not None
    assert len(batch.blocks) == 2
    outer_block = batch.blocks[0]
    inner_block = batch.blocks[1]
    assert torch.equal(outer_block.src_n_id, torch.cat([g1_blocks[0].src_n_id, g2_blocks[0].src_n_id], dim=0))
    assert torch.equal(outer_block.dst_n_id, torch.cat([g1_blocks[0].dst_n_id, g2_blocks[0].dst_n_id], dim=0))
    assert torch.equal(inner_block.src_n_id, torch.cat([g1_blocks[1].src_n_id, g2_blocks[1].src_n_id], dim=0))
    assert torch.equal(inner_block.dst_n_id, torch.cat([g1_blocks[1].dst_n_id, g2_blocks[1].dst_n_id], dim=0))
