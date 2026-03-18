import torch

from vgl import Graph
from vgl.data.sample import LinkPredictionRecord
from vgl.transforms import RandomLinkSplit


def _record_edges(records):
    return {(int(record.src_index), int(record.dst_index)) for record in records}


def test_random_link_split_returns_train_val_test_link_prediction_datasets():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(num_val=1, num_test=1, seed=7)(graph)

    train_records = list(train_dataset)
    val_records = list(val_dataset)
    test_records = list(test_dataset)

    assert len(train_records) == 3
    assert len(val_records) == 1
    assert len(test_records) == 1
    assert all(isinstance(record, LinkPredictionRecord) for record in train_records + val_records + test_records)
    assert {record.metadata["split"] for record in train_records} == {"train"}
    assert {record.metadata["split"] for record in val_records} == {"val"}
    assert {record.metadata["split"] for record in test_records} == {"test"}


def test_random_link_split_uses_train_graph_for_validation_and_train_plus_val_for_test():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(num_val=1, num_test=1, seed=0)(graph)

    train_records = list(train_dataset)
    val_records = list(val_dataset)
    test_records = list(test_dataset)
    train_edges = _record_edges(train_records)
    val_edges = _record_edges(val_records)

    assert _record_edges(train_records) == {
        tuple(edge.tolist())
        for edge in train_records[0].graph.edge_index.t()
    }
    assert {
        tuple(edge.tolist())
        for edge in val_records[0].graph.edge_index.t()
    } == train_edges
    assert {
        tuple(edge.tolist())
        for edge in test_records[0].graph.edge_index.t()
    } == train_edges | val_edges


def test_random_link_split_keeps_reverse_edges_in_same_split_when_undirected():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        x=torch.randn(4, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        is_undirected=True,
        seed=3,
    )(graph)

    split_edges = {
        "train": _record_edges(list(train_dataset)),
        "val": _record_edges(list(val_dataset)),
        "test": _record_edges(list(test_dataset)),
    }

    for src_index, dst_index in graph.edge_index.t().tolist():
        reverse = (int(dst_index), int(src_index))
        containing_splits = [
            split_name
            for split_name, edges in split_edges.items()
            if (int(src_index), int(dst_index)) in edges or reverse in edges
        ]
        assert len(set(containing_splits)) == 1


def test_random_link_split_supports_hetero_edge_type_and_reverse_edge_type():
    edge_type = ("author", "writes", "paper")
    reverse_edge_type = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(3, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1, 2, 1], [1, 2, 3, 0]])},
            reverse_edge_type: {"edge_index": torch.tensor([[1, 2, 3, 0], [0, 1, 2, 1]])},
        },
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        edge_type=edge_type,
        rev_edge_type=reverse_edge_type,
        seed=13,
    )(graph)

    train_records = list(train_dataset)
    val_records = list(val_dataset)
    test_records = list(test_dataset)

    assert len(train_records) == 2
    assert len(val_records) == 1
    assert len(test_records) == 1
    assert all(record.edge_type == edge_type for record in train_records + val_records + test_records)
    assert all(record.reverse_edge_type == reverse_edge_type for record in train_records + val_records + test_records)
    assert {
        tuple(edge.tolist())
        for edge in val_records[0].graph.edges[edge_type].edge_index.t()
    } == _record_edges(train_records)

    train_edges = _record_edges(train_records)
    val_edges = _record_edges(val_records)
    expected_test_graph_edges = train_edges | val_edges
    actual_test_graph_edges = {
        tuple(edge.tolist())
        for edge in test_records[0].graph.edges[edge_type].edge_index.t()
    }
    assert actual_test_graph_edges == expected_test_graph_edges

    reverse_graph_edges = {
        tuple(edge.tolist())
        for edge in test_records[0].graph.edges[reverse_edge_type].edge_index.t()
    }
    assert reverse_graph_edges == {(dst, src) for src, dst in expected_test_graph_edges}


def test_random_link_split_supports_disjoint_train_ratio():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
        x=torch.randn(3, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        disjoint_train_ratio=0.5,
        seed=5,
    )(graph)

    train_records = list(train_dataset)
    val_records = list(val_dataset)
    test_records = list(test_dataset)

    train_supervision_edges = _record_edges(train_records)
    train_message_passing_edges = {
        tuple(edge.tolist())
        for edge in train_records[0].graph.edge_index.t()
    }
    val_message_passing_edges = {
        tuple(edge.tolist())
        for edge in val_records[0].graph.edge_index.t()
    }
    test_message_passing_edges = {
        tuple(edge.tolist())
        for edge in test_records[0].graph.edge_index.t()
    }

    assert len(train_records) == 2
    assert len(train_message_passing_edges) == 2
    assert train_supervision_edges.isdisjoint(train_message_passing_edges)
    assert val_message_passing_edges == train_message_passing_edges
    assert test_message_passing_edges == train_message_passing_edges | _record_edges(val_records)


def test_random_link_split_can_add_negative_samples():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 3),
    )
    positive_edges = {
        (int(src_index), int(dst_index))
        for src_index, dst_index in graph.edge_index.t().tolist()
    }

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        seed=9,
    )(graph)

    for dataset in (train_dataset, val_dataset, test_dataset):
        records = list(dataset)
        labels = [int(record.label) for record in records]
        negatives = {
            (int(record.src_index), int(record.dst_index))
            for record in records
            if int(record.label) == 0
        }
        positives = {
            (int(record.src_index), int(record.dst_index))
            for record in records
            if int(record.label) == 1
        }
        assert 0 in labels
        assert 1 in labels
        assert positives.isdisjoint(negatives)
        assert negatives.isdisjoint(positive_edges)
