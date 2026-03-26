import torch

from vgl.graph.graph import Graph
from vgl.graph.schema import GraphSchema
from vgl.graph.stores import EdgeStore, NodeStore


def _ordered_unique(ids: torch.Tensor | list[int] | tuple[int, ...]) -> torch.Tensor:
    ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
    unique = []
    seen = set()
    for value in ids.tolist():
        value = int(value)
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return torch.tensor(unique, dtype=torch.long, device=ids.device)


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


def _resolve_edge_type(graph: Graph, edge_type=None) -> tuple[str, str, str]:
    if edge_type is None:
        return graph._default_edge_type()
    return tuple(edge_type)


def _slice_edge_store(store, edge_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    edge_data = {"edge_index": store.edge_index[:, edge_ids]}
    edge_count = int(store.edge_index.size(1))
    for key, value in store.data.items():
        if key == "edge_index":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[edge_ids]
        else:
            edge_data[key] = value
    return edge_data


def _relabel_bipartite_edge_index(edge_index, src_mapping: dict[int, int], dst_mapping: dict[int, int]) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    return torch.tensor(
        [[src_mapping[int(src)], dst_mapping[int(dst)]] for src, dst in edge_index.t().tolist()],
        dtype=edge_index.dtype,
        device=edge_index.device,
    ).t().contiguous()


def _normalize_frontier_nodes(graph: Graph, nodes) -> dict[str, torch.Tensor]:
    node_types = graph.schema.node_types
    if isinstance(nodes, dict):
        frontiers = {}
        valid = set(node_types)
        for node_type, node_ids in nodes.items():
            if node_type not in valid:
                raise ValueError(f"unknown node type {node_type!r}")
            frontiers[node_type] = _ordered_unique(node_ids)
        return frontiers
    if len(node_types) != 1:
        raise ValueError("heterogeneous frontier subgraph requires node ids keyed by node type")
    return {node_types[0]: _ordered_unique(nodes)}


def _public_edge_ids(store, edge_ids: torch.Tensor) -> torch.Tensor:
    edge_ids = torch.as_tensor(edge_ids, dtype=torch.long).view(-1)
    public_ids = store.data.get("e_id")
    if public_ids is None:
        return edge_ids
    public_ids = torch.as_tensor(public_ids, dtype=torch.long, device=edge_ids.device)
    return public_ids[edge_ids]


def _preserved_node_stores(graph: Graph) -> dict[str, NodeStore]:
    return {
        node_type: NodeStore(node_type, graph.nodes[node_type].data)
        for node_type in graph.schema.node_types
    }


def _frontier_edge_ids(store, frontier: torch.Tensor | None, *, endpoint: int) -> torch.Tensor:
    device_frontier = None if frontier is None else frontier.to(device=store.edge_index.device)
    if device_frontier is None or int(device_frontier.numel()) == 0:
        return torch.empty(0, dtype=torch.long, device=store.edge_index.device)
    mask = torch.isin(store.edge_index[endpoint], device_frontier)
    return torch.nonzero(mask, as_tuple=False).view(-1)


def _frontier_subgraph(graph: Graph, nodes, *, endpoint: int) -> Graph:
    frontiers = _normalize_frontier_nodes(graph, nodes)
    node_stores = _preserved_node_stores(graph)
    edge_stores = {}
    edge_features = {}
    for edge_type in graph.schema.edge_types:
        store = graph.edges[edge_type]
        node_type = edge_type[2] if endpoint == 1 else edge_type[0]
        edge_ids = _frontier_edge_ids(store, frontiers.get(node_type), endpoint=endpoint)
        edge_data = _slice_edge_store(store, edge_ids)
        edge_data["e_id"] = _public_edge_ids(store, edge_ids)
        edge_stores[edge_type] = EdgeStore(edge_type, edge_data)
        edge_features[edge_type] = tuple(edge_data.keys())
    schema = GraphSchema(
        node_types=graph.schema.node_types,
        edge_types=graph.schema.edge_types,
        node_features={
            node_type: tuple(graph.nodes[node_type].data.keys())
            for node_type in graph.schema.node_types
        },
        edge_features=edge_features,
        time_attr=graph.schema.time_attr,
    )
    return Graph(
        schema=schema,
        nodes=node_stores,
        edges=edge_stores,
        feature_store=graph.feature_store,
        graph_store=graph.graph_store,
    )


def node_subgraph(graph: Graph, node_ids, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    if src_type == dst_type:
        node_ids = _ordered_unique(node_ids)
        node_set = set(node_ids.tolist())
        mapping = {node_id: index for index, node_id in enumerate(node_ids.tolist())}
        store = graph.edges[edge_type]
        selected_edges = [
            index
            for index, (src, dst) in enumerate(store.edge_index.t().tolist())
            if int(src) in node_set and int(dst) in node_set
        ]
        edge_ids = torch.tensor(selected_edges, dtype=torch.long, device=store.edge_index.device)
        edge_index = store.edge_index[:, edge_ids] if edge_ids.numel() > 0 else store.edge_index[:, :0]
        if edge_index.numel() > 0:
            relabelled = torch.tensor(
                [[mapping[int(src)], mapping[int(dst)]] for src, dst in edge_index.t().tolist()],
                dtype=edge_index.dtype,
                device=edge_index.device,
            ).t().contiguous()
        else:
            relabelled = edge_index
        edge_data = {"edge_index": relabelled}
        edge_count = int(store.edge_index.size(1))
        for key, value in store.data.items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_data[key] = value[edge_ids]
            else:
                edge_data[key] = value
        node_data = _slice_node_data(graph, node_ids, node_type=src_type)
        if len(graph.nodes) == 1 and len(graph.edges) == 1 and src_type == "node":
            return Graph.homo(edge_index=relabelled, edge_data={k: v for k, v in edge_data.items() if k != "edge_index"}, **node_data)
        nodes = {src_type: node_data}
        return Graph.hetero(nodes=nodes, edges={edge_type: edge_data}, time_attr=graph.schema.time_attr)

    if not isinstance(node_ids, dict):
        raise ValueError("heterogeneous node_subgraph requires node_ids keyed by node type")
    if src_type not in node_ids or dst_type not in node_ids:
        raise ValueError("heterogeneous node_subgraph requires selections for both source and destination node types")

    src_node_ids = _ordered_unique(node_ids[src_type])
    dst_node_ids = _ordered_unique(node_ids[dst_type])
    src_set = set(int(node_id) for node_id in src_node_ids.tolist())
    dst_set = set(int(node_id) for node_id in dst_node_ids.tolist())
    src_mapping = {int(node_id): index for index, node_id in enumerate(src_node_ids.tolist())}
    dst_mapping = {int(node_id): index for index, node_id in enumerate(dst_node_ids.tolist())}

    store = graph.edges[edge_type]
    selected_edges = [
        index
        for index, (src, dst) in enumerate(store.edge_index.t().tolist())
        if int(src) in src_set and int(dst) in dst_set
    ]
    edge_ids = torch.tensor(selected_edges, dtype=torch.long, device=store.edge_index.device)
    edge_data = _slice_edge_store(store, edge_ids)
    edge_data["edge_index"] = _relabel_bipartite_edge_index(edge_data["edge_index"], src_mapping, dst_mapping)

    nodes = {
        src_type: _slice_node_data(graph, src_node_ids, node_type=src_type),
        dst_type: _slice_node_data(graph, dst_node_ids, node_type=dst_type),
    }
    return Graph.hetero(nodes=nodes, edges={edge_type: edge_data}, time_attr=graph.schema.time_attr)


def edge_subgraph(graph: Graph, edge_ids, *, edge_type=None) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    edge_ids = torch.as_tensor(edge_ids, dtype=torch.long).view(-1)
    store = graph.edges[edge_type]
    edge_data = _slice_edge_store(store, edge_ids)
    if len(graph.nodes) == 1 and len(graph.edges) == 1 and edge_type == ("node", "to", "node"):
        return Graph.homo(
            edge_index=edge_data["edge_index"],
            edge_data={k: v for k, v in edge_data.items() if k != "edge_index"},
            **dict(graph.nodes["node"].data),
        )

    src_type, _, dst_type = edge_type
    nodes = {src_type: dict(graph.nodes[src_type].data)}
    if dst_type != src_type:
        nodes[dst_type] = dict(graph.nodes[dst_type].data)
    return Graph.hetero(nodes=nodes, edges={edge_type: edge_data}, time_attr=graph.schema.time_attr)


def in_subgraph(graph: Graph, nodes) -> Graph:
    return _frontier_subgraph(graph, nodes, endpoint=1)


def out_subgraph(graph: Graph, nodes) -> Graph:
    return _frontier_subgraph(graph, nodes, endpoint=0)
