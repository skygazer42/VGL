from __future__ import annotations

from collections import deque

import torch

from vgl.ops.subgraph import node_subgraph
from vgl.transforms.base import BaseTransform
from vgl.transforms._utils import clone_graph, is_edge_aligned, is_homo_graph


class ToUndirected(BaseTransform):
    def __call__(self, graph):
        if not is_homo_graph(graph):
            raise ValueError("ToUndirected currently supports homogeneous graphs only")
        edge_type = graph._default_edge_type()
        store = graph.edges[edge_type]
        edge_index = store.edge_index
        existing = {tuple(edge.tolist()) for edge in edge_index.t()}
        added_edges = []
        mirror_indices = []
        for edge_id, (src_index, dst_index) in enumerate(edge_index.t().tolist()):
            reverse = (int(dst_index), int(src_index))
            if reverse in existing:
                continue
            existing.add(reverse)
            added_edges.append(reverse)
            mirror_indices.append(edge_id)

        if not added_edges:
            return graph

        added_edge_index = torch.tensor(added_edges, dtype=edge_index.dtype, device=edge_index.device).t().contiguous()
        new_edge_index = torch.cat([edge_index, added_edge_index], dim=1)
        edge_count = int(edge_index.size(1))
        edges = {edge_type: {"edge_index": new_edge_index}}
        for key, value in store.data.items():
            if key == "edge_index":
                continue
            if is_edge_aligned(value, edge_count):
                index = torch.tensor(mirror_indices, dtype=torch.long, device=edge_index.device)
                edges[edge_type][key] = torch.cat([value, value[index]], dim=0)
            else:
                edges[edge_type][key] = value
        return clone_graph(graph, edges=edges)


class AddSelfLoops(BaseTransform):
    def __init__(self, *, fill_value: float = 1.0):
        self.fill_value = fill_value

    def __call__(self, graph):
        if not is_homo_graph(graph):
            raise ValueError("AddSelfLoops currently supports homogeneous graphs only")
        edge_type = graph._default_edge_type()
        store = graph.edges[edge_type]
        edge_index = store.edge_index
        num_nodes = graph.x.size(0)
        loop_mask = edge_index[0] == edge_index[1]
        loop_nodes = set(edge_index[0, loop_mask].tolist())
        missing_nodes = [node_id for node_id in range(num_nodes) if node_id not in loop_nodes]
        if not missing_nodes:
            return graph

        added_edge_index = torch.tensor(
            [[node_id, node_id] for node_id in missing_nodes],
            dtype=edge_index.dtype,
            device=edge_index.device,
        ).t().contiguous()
        new_edge_index = torch.cat([edge_index, added_edge_index], dim=1)
        edge_count = int(edge_index.size(1))
        add_count = len(missing_nodes)
        edges = {edge_type: {"edge_index": new_edge_index}}
        for key, value in store.data.items():
            if key == "edge_index":
                continue
            if is_edge_aligned(value, edge_count):
                fill_shape = (add_count,) + tuple(value.shape[1:])
                fill_tensor = value.new_full(fill_shape, self.fill_value)
                edges[edge_type][key] = torch.cat([value, fill_tensor], dim=0)
            else:
                edges[edge_type][key] = value
        return clone_graph(graph, edges=edges)


class RemoveSelfLoops(BaseTransform):
    def __call__(self, graph):
        if not is_homo_graph(graph):
            raise ValueError("RemoveSelfLoops currently supports homogeneous graphs only")
        edge_type = graph._default_edge_type()
        store = graph.edges[edge_type]
        keep_mask = store.edge_index[0] != store.edge_index[1]
        edge_count = int(store.edge_index.size(1))
        edges = {edge_type: {"edge_index": store.edge_index[:, keep_mask]}}
        for key, value in store.data.items():
            if key == "edge_index":
                continue
            if is_edge_aligned(value, edge_count):
                edges[edge_type][key] = value[keep_mask]
            else:
                edges[edge_type][key] = value
        return clone_graph(graph, edges=edges)


class LargestConnectedComponents(BaseTransform):
    def __init__(self, *, num_components: int = 1):
        self.num_components = int(num_components)

    def __call__(self, graph):
        if not is_homo_graph(graph):
            raise ValueError("LargestConnectedComponents currently supports homogeneous graphs only")
        if self.num_components < 1:
            raise ValueError("num_components must be >= 1")

        num_nodes = int(graph.x.size(0))
        adjacency = {node_id: set() for node_id in range(num_nodes)}
        for src_index, dst_index in graph.edge_index.t().tolist():
            src_index = int(src_index)
            dst_index = int(dst_index)
            adjacency[src_index].add(dst_index)
            adjacency[dst_index].add(src_index)

        seen = set()
        components = []
        for node_id in range(num_nodes):
            if node_id in seen:
                continue
            queue = deque([node_id])
            component = []
            seen.add(node_id)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    queue.append(neighbor)
            components.append(sorted(component))

        components.sort(key=len, reverse=True)
        kept_nodes = []
        for component in components[: self.num_components]:
            kept_nodes.extend(component)
        return node_subgraph(graph, torch.tensor(sorted(kept_nodes), dtype=torch.long))
