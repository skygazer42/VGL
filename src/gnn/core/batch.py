from dataclasses import dataclass

import torch

from gnn.core.graph import Graph


@dataclass(slots=True)
class GraphBatch:
    graphs: list[Graph]
    graph_index: torch.Tensor

    @classmethod
    def from_graphs(cls, graphs: list[Graph]) -> "GraphBatch":
        counts = [graph.x.size(0) for graph in graphs]
        graph_index = torch.repeat_interleave(
            torch.arange(len(graphs)),
            torch.tensor(counts),
        )
        return cls(graphs=graphs, graph_index=graph_index)

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)
