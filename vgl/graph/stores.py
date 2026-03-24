from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class NodeStore:
    type_name: str
    data: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_feature_store(cls, type_name, feature_names, feature_store):
        data = {}
        for feature_name in feature_names:
            key = ("node", type_name, feature_name)
            index = torch.arange(feature_store.shape(key)[0], dtype=torch.long)
            data[feature_name] = feature_store.fetch(key, index).values
        return cls(type_name, data)

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(slots=True)
class EdgeStore:
    type_name: tuple[str, str, str]
    data: dict[str, torch.Tensor] = field(default_factory=dict)
    adjacency_cache: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_storage(cls, type_name, feature_names, feature_store, graph_store):
        edge_count = graph_store.edge_count(type_name)
        data = {"edge_index": graph_store.edge_index(type_name)}
        index = torch.arange(edge_count, dtype=torch.long)
        for feature_name in feature_names:
            if feature_name == "edge_index":
                continue
            key = ("edge", type_name, feature_name)
            data[feature_name] = feature_store.fetch(key, index).values
        return cls(type_name, data)

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
