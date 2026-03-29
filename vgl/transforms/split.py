from __future__ import annotations

from collections import defaultdict

import torch

from vgl.dataloading.dataset import ListDataset
from vgl.transforms.base import BaseTransform
from vgl.transforms._utils import clone_graph


class RandomNodeSplit(BaseTransform):
    def __init__(
        self,
        num_val=0.1,
        num_test=0.2,
        *,
        num_train=None,
        num_train_per_class=None,
        target="y",
        node_type: str | None = None,
        seed: int | None = None,
    ):
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.num_train_per_class = None if num_train_per_class is None else int(num_train_per_class)
        self.target = str(target)
        self.node_type = node_type
        self.seed = seed

    @staticmethod
    def _resolve_count(value, total: int, *, name: str) -> int:
        if value is None:
            return -1
        if isinstance(value, float):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0)")
            return int(total * value)
        count = int(value)
        if count < 0:
            raise ValueError(f"{name} must be >= 0")
        return count

    def _generator(self):
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed))
        return generator

    def __call__(self, graph):
        if self.node_type is None:
            if len(graph.nodes) == 1:
                node_type = graph.schema.node_types[0]
            else:
                raise ValueError("RandomNodeSplit requires node_type for heterogeneous graphs")
        else:
            node_type = self.node_type
        if node_type not in graph.nodes:
            raise ValueError(f"RandomNodeSplit unknown node_type: {node_type!r}")

        node_store = graph.nodes[node_type]
        labels = node_store.data.get(self.target)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"RandomNodeSplit requires tensor node labels at {self.target!r}")
        num_nodes = int(labels.size(0))
        generator = self._generator()

        remaining = torch.ones(num_nodes, dtype=torch.bool, device=labels.device)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=labels.device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=labels.device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=labels.device)

        if self.num_train_per_class is not None:
            if self.num_train_per_class < 0:
                raise ValueError("num_train_per_class must be >= 0")
            per_class = defaultdict(list)
            for index, label in enumerate(labels.tolist()):
                per_class[int(label)].append(index)
            for indices in per_class.values():
                order = torch.randperm(len(indices), generator=generator).tolist()
                for position in order[: self.num_train_per_class]:
                    train_mask[int(indices[position])] = True
            remaining &= ~train_mask

        num_val = self._resolve_count(self.num_val, num_nodes, name="num_val")
        num_test = self._resolve_count(self.num_test, num_nodes, name="num_test")
        remaining_ids = torch.nonzero(remaining, as_tuple=False).view(-1)

        if self.num_train is None:
            num_train = max(int(remaining_ids.numel()) - max(num_val, 0) - max(num_test, 0), 0)
        else:
            num_train = self._resolve_count(self.num_train, num_nodes, name="num_train")
        if num_train > int(remaining_ids.numel()):
            raise ValueError("RandomNodeSplit requested more training nodes than remain available")

        permuted_remaining = remaining_ids[torch.randperm(int(remaining_ids.numel()), generator=generator)]
        if self.num_train_per_class is None and num_train > 0:
            train_ids = permuted_remaining[:num_train]
            train_mask[train_ids] = True
            permuted_remaining = permuted_remaining[num_train:]

        if num_val > int(permuted_remaining.numel()):
            raise ValueError("RandomNodeSplit requested more validation nodes than remain available")
        val_ids = permuted_remaining[:num_val]
        val_mask[val_ids] = True
        permuted_remaining = permuted_remaining[num_val:]

        if num_test > int(permuted_remaining.numel()):
            raise ValueError("RandomNodeSplit requested more test nodes than remain available")
        test_ids = permuted_remaining[:num_test]
        test_mask[test_ids] = True

        nodes = {
            current_node_type: dict(store.data)
            for current_node_type, store in graph.nodes.items()
        }
        nodes[node_type]["train_mask"] = train_mask
        nodes[node_type]["val_mask"] = val_mask
        nodes[node_type]["test_mask"] = test_mask
        return clone_graph(graph, nodes=nodes)


class RandomGraphSplit(BaseTransform):
    def __init__(self, num_val=0.1, num_test=0.2, *, seed: int | None = None):
        self.num_val = num_val
        self.num_test = num_test
        self.seed = seed

    @staticmethod
    def _resolve_count(value, total: int, *, name: str) -> int:
        if isinstance(value, float):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0)")
            return int(total * value)
        count = int(value)
        if count < 0:
            raise ValueError(f"{name} must be >= 0")
        return count

    def __call__(self, dataset):
        items = list(dataset)
        total = len(items)
        if total == 0:
            raise ValueError("RandomGraphSplit requires at least one graph")
        num_val = self._resolve_count(self.num_val, total, name="num_val")
        num_test = self._resolve_count(self.num_test, total, name="num_test")
        if num_val + num_test >= total:
            raise ValueError("RandomGraphSplit requires at least one training graph")
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed))
        permutation = torch.randperm(total, generator=generator).tolist()
        ordered = [items[index] for index in permutation]
        train_count = total - num_val - num_test
        return (
            ListDataset(ordered[:train_count]),
            ListDataset(ordered[train_count:train_count + num_val]),
            ListDataset(ordered[train_count + num_val:]),
        )
