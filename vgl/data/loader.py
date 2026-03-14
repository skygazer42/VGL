from vgl.core.batch import GraphBatch, LinkPredictionBatch, TemporalEventBatch
from vgl.data.sample import LinkPredictionRecord, TemporalEventRecord


class Loader:
    def __init__(self, dataset, sampler, batch_size, label_source=None, label_key=None):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.label_source = label_source
        self.label_key = label_key

    def _build_batch(self, items):
        if items and isinstance(items[0], TemporalEventRecord):
            return TemporalEventBatch.from_records(items)
        if items and isinstance(items[0], LinkPredictionRecord):
            return LinkPredictionBatch.from_records(items)
        if items and hasattr(items[0], "graph") and self.label_source is not None and self.label_key is not None:
            return GraphBatch.from_samples(
                items,
                label_key=self.label_key,
                label_source=self.label_source,
            )
        return GraphBatch.from_graphs(items)

    def __iter__(self):
        batch = []
        for item in self.dataset.graphs:
            batch.append(self.sampler.sample(item))
            if len(batch) == self.batch_size:
                yield self._build_batch(batch)
                batch = []
        if batch:
            yield self._build_batch(batch)

