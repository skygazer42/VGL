from dataclasses import dataclass

from gnn.core.schema import GraphSchema
from gnn.core.stores import EdgeStore, NodeStore


@dataclass(slots=True)
class GraphView:
    base: object
    nodes: dict[str, NodeStore]
    edges: dict[tuple[str, str, str], EdgeStore]
    schema: GraphSchema
