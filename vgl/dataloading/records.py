from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SampleRecord:
    graph: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    source_graph_id: str | None = None
    subgraph_seed: Any | None = None


@dataclass(slots=True)
class LinkPredictionRecord:
    graph: Any
    src_index: int
    dst_index: int
    label: int
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    exclude_seed_edge: bool = False
    hard_negative_dst: Any | None = None
    candidate_dst: Any | None = None
    edge_type: Any | None = None
    reverse_edge_type: Any | None = None
    query_id: Any | None = None
    filter_ranking: bool = False


@dataclass(slots=True)
class TemporalEventRecord:
    graph: Any
    src_index: int
    dst_index: int
    timestamp: int
    label: int
    event_features: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
