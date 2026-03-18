# Candidate Link Sampler

## Goal

Add a mainstream link-prediction evaluation path that separates:

- training-time sampled negatives
- evaluation-time candidate ranking

This follows the same high-level split used by PyG/GraphGym, DGL/GraphBolt, and GraphStorm style pipelines, where validation/test ranking is usually computed over a candidate set or all destinations instead of reusing the training negative sampler.

## Scope

- add `CandidateLinkSampler`
- support two candidate modes:
  - all destinations when `candidate_dst` is not provided
  - explicit candidate sets from `LinkPredictionRecord.candidate_dst`
- auto-include the positive destination in the query candidate set
- mark other known positive edges from the same source as `filter_ranking=True`
- preserve existing `exclude_seed_edges` behavior
- expose the sampler through `vgl.dataloading`, `vgl.data`, and `vgl`

## API

```python
CandidateLinkSampler(
    *,
    filter_known_positive_edges: bool = True,
    exclude_seed_edges: bool = False,
)
```

`LinkPredictionRecord` also gains:

```python
candidate_dst: Any | None = None
```

## Tests

- loader expands positive seeds into all-node candidate ranking batches
- explicit `candidate_dst` is supported and deduplicated
- out-of-range candidates are rejected
- trainer can report both raw and filtered ranking metrics from the same candidate batch
