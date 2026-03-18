import torch
from vgl.dataloading.dataset import ListDataset
from vgl.dataloading.records import LinkPredictionRecord
from vgl.graph.stores import EdgeStore
from vgl.graph.view import GraphView


def _slice_edge_store(store, indices):
    edge_count = int(store.edge_index.size(1))
    index = torch.as_tensor(indices, dtype=torch.long)
    edge_data = {}
    for key, value in store.data.items():
        if key == "edge_index":
            edge_data[key] = value[:, index]
        elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[index]
        else:
            edge_data[key] = value
    return EdgeStore(store.type_name, edge_data)


def _edge_subgraph(graph, edge_indices_by_type):
    edges = dict(graph.edges)
    for edge_type, indices in edge_indices_by_type.items():
        edges[edge_type] = _slice_edge_store(graph.edges[edge_type], indices)
    base = getattr(graph, "base", graph)
    return GraphView(base=base, nodes=graph.nodes, edges=edges, schema=graph.schema)


class RandomLinkSplit:
    def __init__(
        self,
        num_val=0.1,
        num_test=0.2,
        *,
        is_undirected=False,
        include_validation_edges_in_test=True,
        edge_type=None,
        rev_edge_type=None,
        disjoint_train_ratio=0.0,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        seed=None,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = bool(is_undirected)
        self.include_validation_edges_in_test = bool(include_validation_edges_in_test)
        self.edge_type = None if edge_type is None else tuple(edge_type)
        self.rev_edge_type = None if rev_edge_type is None else tuple(rev_edge_type)
        self.disjoint_train_ratio = disjoint_train_ratio
        self.neg_sampling_ratio = neg_sampling_ratio
        self.add_negative_train_samples = bool(add_negative_train_samples)
        self.seed = seed

    def _resolve_count(self, value, total, name):
        if isinstance(value, float):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0)")
            return int(total * value)
        count = int(value)
        if count < 0:
            raise ValueError(f"{name} must be >= 0")
        if count >= total:
            raise ValueError(f"{name} must be smaller than the total number of edge groups")
        return count

    def _resolve_negative_count(self, num_positive):
        value = self.neg_sampling_ratio
        if isinstance(value, float):
            if value < 0.0:
                raise ValueError("neg_sampling_ratio must be >= 0")
            return int(num_positive * value)
        ratio = int(value)
        if ratio < 0:
            raise ValueError("neg_sampling_ratio must be >= 0")
        return int(num_positive) * ratio

    def _edge_groups(self, edge_index):
        num_edges = int(edge_index.size(1))
        if not self.is_undirected:
            return [[index] for index in range(num_edges)]
        groups = {}
        for index, (src_index, dst_index) in enumerate(edge_index.t().tolist()):
            key = tuple(sorted((int(src_index), int(dst_index))))
            groups.setdefault(key, []).append(index)
        return list(groups.values())

    def _resolve_edge_type(self, graph):
        if self.edge_type is not None:
            edge_type = self.edge_type
        elif len(graph.edges) == 1:
            edge_type = next(iter(graph.edges))
        else:
            try:
                edge_type = graph._default_edge_type()
            except AttributeError as exc:
                raise ValueError(
                    "RandomLinkSplit requires edge_type for heterogeneous graphs with multiple edge types"
                ) from exc
        if edge_type not in graph.edges:
            raise ValueError("RandomLinkSplit edge_type must exist in the source graph")
        return edge_type

    def _reverse_edge_indices(self, edge_index, reverse_edge_index, indices):
        reverse_lookup = {}
        for reverse_edge_id, (src_index, dst_index) in enumerate(reverse_edge_index.t().tolist()):
            key = (int(src_index), int(dst_index))
            reverse_lookup.setdefault(key, []).append(int(reverse_edge_id))
        selected_reverse_indices = []
        for edge_id in indices:
            src_index = int(edge_index[0, edge_id].item())
            dst_index = int(edge_index[1, edge_id].item())
            selected_reverse_indices.extend(reverse_lookup.get((dst_index, src_index), []))
        return sorted(set(selected_reverse_indices))

    def _records_from_indices(self, graph, edge_index, indices, *, split, edge_type, reverse_edge_type):
        records = []
        for edge_id in indices:
            src_index = int(edge_index[0, edge_id].item())
            dst_index = int(edge_index[1, edge_id].item())
            records.append(
                LinkPredictionRecord(
                    graph=graph,
                    src_index=src_index,
                    dst_index=dst_index,
                    label=1,
                    metadata={
                        "split": split,
                        "edge_id": int(edge_id),
                        "edge_type": edge_type,
                        "reverse_edge_type": reverse_edge_type,
                    },
                    sample_id=f"{split}:{edge_id}",
                    edge_type=edge_type,
                    reverse_edge_type=reverse_edge_type,
                )
            )
        return records

    def _positive_edge_set(self, edge_index, *, src_node_type, dst_node_type):
        edges = {
            (int(src_index), int(dst_index))
            for src_index, dst_index in edge_index.t().tolist()
        }
        if self.is_undirected and src_node_type == dst_node_type:
            edges = edges | {(dst_index, src_index) for src_index, dst_index in edges}
        return edges

    def _sample_negative_edges(self, *, count, num_src_nodes, num_dst_nodes, excluded_edges, generator):
        if count <= 0:
            return []
        all_possible = num_src_nodes * num_dst_nodes
        if all_possible <= len(excluded_edges):
            raise ValueError("RandomLinkSplit could not sample negatives: no valid negative edges remain")

        sampled = set()
        while len(sampled) < count:
            remaining = count - len(sampled)
            # Sample more than needed to reduce rejection loops on dense graphs.
            draw_count = max(remaining * 2, 32)
            src_index = torch.randint(num_src_nodes, (draw_count,), generator=generator)
            dst_index = torch.randint(num_dst_nodes, (draw_count,), generator=generator)
            for src_value, dst_value in zip(src_index.tolist(), dst_index.tolist()):
                pair = (int(src_value), int(dst_value))
                if pair in excluded_edges or pair in sampled:
                    continue
                sampled.add(pair)
                if len(sampled) == count:
                    break
        return sorted(sampled)

    def _attach_negative_records(
        self,
        *,
        graph,
        split,
        records,
        edge_type,
        reverse_edge_type,
        num_src_nodes,
        num_dst_nodes,
        excluded_edges,
        generator,
    ):
        if split == "train" and not self.add_negative_train_samples:
            return records
        negative_count = self._resolve_negative_count(len(records))
        if negative_count <= 0:
            return records

        negative_edges = self._sample_negative_edges(
            count=negative_count,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            excluded_edges=excluded_edges,
            generator=generator,
        )
        augmented = list(records)
        for offset, (src_index, dst_index) in enumerate(negative_edges):
            augmented.append(
                LinkPredictionRecord(
                    graph=graph,
                    src_index=src_index,
                    dst_index=dst_index,
                    label=0,
                    metadata={
                        "split": split,
                        "edge_id": None,
                        "negative_sampled": True,
                        "edge_type": edge_type,
                        "reverse_edge_type": reverse_edge_type,
                    },
                    sample_id=f"{split}:neg:{offset}",
                    edge_type=edge_type,
                    reverse_edge_type=reverse_edge_type,
                )
            )
        return augmented

    def __call__(self, graph):
        edge_type = self._resolve_edge_type(graph)
        edge_index = graph.edges[edge_type].edge_index
        src_node_type, _, dst_node_type = edge_type
        num_src_nodes = int(graph.nodes[src_node_type].x.size(0))
        num_dst_nodes = int(graph.nodes[dst_node_type].x.size(0))
        reverse_edge_index = None
        if self.rev_edge_type is not None:
            if self.rev_edge_type not in graph.edges:
                raise ValueError("RandomLinkSplit rev_edge_type must exist in the source graph")
            reverse_edge_index = graph.edges[self.rev_edge_type].edge_index

        groups = self._edge_groups(edge_index)
        total_groups = len(groups)
        if total_groups < 3:
            raise ValueError("RandomLinkSplit requires at least three edge groups")
        num_val = self._resolve_count(self.num_val, total_groups, "num_val")
        num_test = self._resolve_count(self.num_test, total_groups, "num_test")
        num_train = total_groups - num_val - num_test
        if num_train <= 0:
            raise ValueError("RandomLinkSplit requires at least one training edge group")

        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.seed))
        permutation = torch.randperm(total_groups, generator=generator).tolist()
        shuffled_groups = [groups[index] for index in permutation]

        train_group_indices = shuffled_groups[:num_train]
        val_group_indices = shuffled_groups[num_train:num_train + num_val]
        test_group_indices = shuffled_groups[num_train + num_val:]

        num_disjoint = self._resolve_count(
            self.disjoint_train_ratio,
            len(train_group_indices),
            "disjoint_train_ratio",
        ) if train_group_indices else 0
        if num_disjoint > 0:
            train_supervision_groups = train_group_indices[:num_disjoint]
            train_message_passing_groups = train_group_indices[num_disjoint:]
        else:
            train_supervision_groups = train_group_indices
            train_message_passing_groups = train_group_indices

        train_indices = sorted(index for group in train_supervision_groups for index in group)
        train_graph_indices = sorted(index for group in train_message_passing_groups for index in group)
        val_indices = sorted(index for group in val_group_indices for index in group)
        test_indices = sorted(index for group in test_group_indices for index in group)

        train_edge_indices = {edge_type: train_graph_indices}
        val_edge_indices = {edge_type: train_graph_indices}
        test_target_indices = (
            train_graph_indices + val_indices
            if self.include_validation_edges_in_test
            else train_graph_indices
        )
        test_edge_indices = {edge_type: test_target_indices}
        if reverse_edge_index is not None:
            train_reverse_indices = self._reverse_edge_indices(edge_index, reverse_edge_index, train_graph_indices)
            val_reverse_indices = self._reverse_edge_indices(edge_index, reverse_edge_index, val_indices)
            train_val_reverse_indices = (
                train_reverse_indices + val_reverse_indices
                if self.include_validation_edges_in_test
                else train_reverse_indices
            )
            train_edge_indices[self.rev_edge_type] = train_reverse_indices
            val_edge_indices[self.rev_edge_type] = train_reverse_indices
            test_edge_indices[self.rev_edge_type] = sorted(set(train_val_reverse_indices))

        train_graph = _edge_subgraph(graph, train_edge_indices)
        val_graph = _edge_subgraph(graph, val_edge_indices)
        if self.include_validation_edges_in_test:
            test_graph = _edge_subgraph(graph, test_edge_indices)
        else:
            test_graph = _edge_subgraph(graph, test_edge_indices)

        train_records = self._records_from_indices(
            train_graph,
            edge_index,
            train_indices,
            split="train",
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
        )
        val_records = self._records_from_indices(
            val_graph,
            edge_index,
            val_indices,
            split="val",
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
        )
        test_records = self._records_from_indices(
            test_graph,
            edge_index,
            test_indices,
            split="test",
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
        )

        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.seed) + 1)
        excluded_edges = self._positive_edge_set(
            edge_index,
            src_node_type=src_node_type,
            dst_node_type=dst_node_type,
        )
        train_records = self._attach_negative_records(
            graph=train_graph,
            split="train",
            records=train_records,
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            excluded_edges=excluded_edges,
            generator=generator,
        )
        val_records = self._attach_negative_records(
            graph=val_graph,
            split="val",
            records=val_records,
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            excluded_edges=excluded_edges,
            generator=generator,
        )
        test_records = self._attach_negative_records(
            graph=test_graph,
            split="test",
            records=test_records,
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            excluded_edges=excluded_edges,
            generator=generator,
        )
        return ListDataset(train_records), ListDataset(val_records), ListDataset(test_records)
