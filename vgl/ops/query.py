import torch

from vgl.graph.graph import Graph


def _resolve_edge_type(graph: Graph, edge_type=None) -> tuple[str, str, str]:
    if edge_type is None:
        return graph._default_edge_type()
    return tuple(edge_type)


def _public_edge_ids(store) -> torch.Tensor:
    public_ids = store.data.get("e_id")
    if public_ids is None:
        return torch.arange(store.edge_index.size(1), dtype=torch.long, device=store.edge_index.device)
    return torch.as_tensor(public_ids, dtype=torch.long, device=store.edge_index.device).view(-1)


def _edge_positions_by_public_id(store) -> dict[int, int]:
    return {int(edge_id): index for index, edge_id in enumerate(_public_edge_ids(store).tolist())}


def _pair_positions(store) -> dict[tuple[int, int], list[int]]:
    positions: dict[tuple[int, int], list[int]] = {}
    for index, (src, dst) in enumerate(store.edge_index.t().tolist()):
        positions.setdefault((int(src), int(dst)), []).append(index)
    return positions


def _normalize_edge_ids(edge_ids) -> torch.Tensor:
    return torch.as_tensor(edge_ids, dtype=torch.long).view(-1)


def _normalize_node_pairs(u, v) -> tuple[torch.Tensor, torch.Tensor, bool]:
    u_tensor = torch.as_tensor(u, dtype=torch.long)
    v_tensor = torch.as_tensor(v, dtype=torch.long)
    scalar_input = u_tensor.ndim == 0 and v_tensor.ndim == 0
    u_ids = u_tensor.view(-1)
    v_ids = v_tensor.view(-1)
    if u_ids.numel() != v_ids.numel():
        raise ValueError("u and v must describe the same number of node pairs")
    return u_ids, v_ids, scalar_input


def _validate_node_pairs(graph: Graph, edge_type, u_ids: torch.Tensor, v_ids: torch.Tensor) -> None:
    src_type, _, dst_type = edge_type
    src_count = graph._node_count(src_type)
    dst_count = graph._node_count(dst_type)
    if torch.any((u_ids < 0) | (u_ids >= src_count)):
        raise ValueError("source node ids are out of range")
    if torch.any((v_ids < 0) | (v_ids >= dst_count)):
        raise ValueError("destination node ids are out of range")


def find_edges(graph: Graph, eids, *, edge_type=None) -> tuple[torch.Tensor, torch.Tensor]:
    edge_type = _resolve_edge_type(graph, edge_type)
    store = graph.edges[edge_type]
    requested = _normalize_edge_ids(eids)
    positions_by_id = _edge_positions_by_public_id(store)
    positions: list[int] = []
    for edge_id in requested.tolist():
        try:
            positions.append(positions_by_id[int(edge_id)])
        except KeyError as exc:
            raise ValueError(f"unknown edge id {edge_id}") from exc
    if not positions:
        empty = torch.empty(0, dtype=store.edge_index.dtype, device=store.edge_index.device)
        return empty, empty
    position_tensor = torch.tensor(positions, dtype=torch.long, device=store.edge_index.device)
    edge_index = store.edge_index[:, position_tensor]
    return edge_index[0], edge_index[1]


def edge_ids(graph: Graph, u, v, *, return_uv: bool = False, edge_type=None):
    edge_type = _resolve_edge_type(graph, edge_type)
    store = graph.edges[edge_type]
    u_ids, v_ids, _ = _normalize_node_pairs(u, v)
    _validate_node_pairs(graph, edge_type, u_ids, v_ids)
    positions = _pair_positions(store)
    public_ids = _public_edge_ids(store).tolist()
    device = store.edge_index.device

    if return_uv:
        matched_src: list[int] = []
        matched_dst: list[int] = []
        matched_eids: list[int] = []
        for src, dst in zip(u_ids.tolist(), v_ids.tolist()):
            matches = positions.get((int(src), int(dst)))
            if not matches:
                raise ValueError(f"no edge exists between {src} and {dst}")
            for index in matches:
                matched_src.append(int(src))
                matched_dst.append(int(dst))
                matched_eids.append(int(public_ids[index]))
        return (
            torch.tensor(matched_src, dtype=torch.long, device=device),
            torch.tensor(matched_dst, dtype=torch.long, device=device),
            torch.tensor(matched_eids, dtype=torch.long, device=device),
        )

    matched: list[int] = []
    for src, dst in zip(u_ids.tolist(), v_ids.tolist()):
        matches = positions.get((int(src), int(dst)))
        if not matches:
            raise ValueError(f"no edge exists between {src} and {dst}")
        matched.append(int(public_ids[matches[0]]))
    return torch.tensor(matched, dtype=torch.long, device=device)


def has_edges_between(graph: Graph, u, v, *, edge_type=None):
    edge_type = _resolve_edge_type(graph, edge_type)
    store = graph.edges[edge_type]
    u_ids, v_ids, scalar_input = _normalize_node_pairs(u, v)
    _validate_node_pairs(graph, edge_type, u_ids, v_ids)
    positions = _pair_positions(store)
    exists = [((int(src), int(dst)) in positions) for src, dst in zip(u_ids.tolist(), v_ids.tolist())]
    if scalar_input:
        return bool(exists[0])
    return torch.tensor(exists, dtype=torch.bool, device=store.edge_index.device)
