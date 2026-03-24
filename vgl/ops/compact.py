import torch

from vgl.graph.graph import Graph
from vgl.ops.subgraph import _ordered_unique


def _slice_node_data(graph: Graph, node_ids: torch.Tensor, *, node_type: str) -> dict[str, torch.Tensor]:
    store = graph.nodes[node_type]
    node_count = graph._node_count(node_type)
    data = {}
    for key, value in store.data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == node_count:
            data[key] = value[node_ids]
        else:
            data[key] = value
    return data


def compact_nodes(graph: Graph, node_ids, *, edge_type=None):
    if edge_type is None:
        edge_type = graph._default_edge_type()
    edge_type = tuple(edge_type)
    src_type, _, dst_type = edge_type
    store = graph.edges[edge_type]
    edge_index = store.edge_index

    if src_type == dst_type and src_type == "node" and len(graph.nodes) == 1 and len(graph.edges) == 1:
        node_ids = _ordered_unique(node_ids)
        mapping = {int(node_id): index for index, node_id in enumerate(node_ids.tolist())}
        relabelled = torch.tensor(
            [[mapping[int(src)], mapping[int(dst)]] for src, dst in edge_index.t().tolist()],
            dtype=edge_index.dtype,
            device=edge_index.device,
        ).t().contiguous()
        node_data = _slice_node_data(graph, node_ids, node_type="node")
        edge_data = {}
        edge_count = int(edge_index.size(1))
        for key, value in store.data.items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_data[key] = value
            else:
                edge_data[key] = value
        return Graph.homo(edge_index=relabelled, edge_data=edge_data, **node_data), mapping

    if not isinstance(node_ids, dict):
        raise ValueError("heterogeneous compact_nodes requires node_ids keyed by node type")
    if src_type not in node_ids or dst_type not in node_ids:
        raise ValueError("heterogeneous compact_nodes requires selections for both source and destination node types")

    selected_node_ids = {}
    mappings = {}
    for node_type in dict.fromkeys((src_type, dst_type)):
        ids = _ordered_unique(node_ids[node_type])
        selected_node_ids[node_type] = ids
        mappings[node_type] = {int(node_id): index for index, node_id in enumerate(ids.tolist())}

    relabelled = torch.tensor(
        [[mappings[src_type][int(src)], mappings[dst_type][int(dst)]] for src, dst in edge_index.t().tolist()],
        dtype=edge_index.dtype,
        device=edge_index.device,
    ).t().contiguous()

    edge_data = {}
    edge_count = int(edge_index.size(1))
    for key, value in store.data.items():
        if key == "edge_index":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value
        else:
            edge_data[key] = value

    nodes = {
        node_type: _slice_node_data(graph, ids, node_type=node_type)
        for node_type, ids in selected_node_ids.items()
    }
    edges = {edge_type: {"edge_index": relabelled, **edge_data}}
    return Graph.hetero(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr), mappings
