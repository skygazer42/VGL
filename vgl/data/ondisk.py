import json
from pathlib import Path

import torch

from vgl.data.catalog import DatasetManifest, DatasetSplit
from vgl.dataloading.dataset import ListDataset
from vgl.graph.graph import Graph


def serialize_graph(graph: Graph) -> dict:
    return {
        "nodes": {
            node_type: dict(store.data)
            for node_type, store in graph.nodes.items()
        },
        "edges": {
            tuple(edge_type): dict(store.data)
            for edge_type, store in graph.edges.items()
        },
        "time_attr": graph.schema.time_attr,
    }


def deserialize_graph(payload: dict) -> Graph:
    if "nodes" not in payload or "edges" not in payload:
        return Graph.homo(
            edge_index=payload["edge_index"],
            edge_data=dict(payload.get("edge_data", {})),
            **dict(payload["node_data"]),
        )

    nodes = {
        node_type: dict(node_data)
        for node_type, node_data in payload["nodes"].items()
    }
    edges = {
        tuple(edge_type): dict(edge_data)
        for edge_type, edge_data in payload["edges"].items()
    }
    time_attr = payload.get("time_attr")
    if time_attr is not None:
        return Graph.temporal(nodes=nodes, edges=edges, time_attr=time_attr)
    return Graph.hetero(nodes=nodes, edges=edges)


def manifest_from_dict(payload: dict) -> DatasetManifest:
    return DatasetManifest(
        name=payload["name"],
        version=payload.get("version", "0"),
        description=payload.get("description"),
        metadata=payload.get("metadata", {}),
        splits=tuple(
            DatasetSplit(
                name=split["name"],
                size=split["size"],
                metadata=split.get("metadata", {}),
            )
            for split in payload.get("splits", ())
        ),
    )


class OnDiskGraphDataset(ListDataset):
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = manifest_from_dict(json.loads((self.root / "manifest.json").read_text()))
        payload = torch.load(self.root / "graphs.pt", weights_only=True)
        super().__init__([deserialize_graph(graph_payload) for graph_payload in payload])

    @classmethod
    def write(cls, root, manifest: DatasetManifest, graphs) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        graph_payload = [serialize_graph(graph) for graph in graphs]
        (root / "manifest.json").write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))
        torch.save(graph_payload, root / "graphs.pt")
        return root


__all__ = ["OnDiskGraphDataset"]
