from collections import deque

from vgl.dataloading.executor import PlanExecutor
from vgl.dataloading.materialize import materialize_batch, materialize_context
from vgl.dataloading.plan import SamplingPlan


class Loader:
    def __init__(
        self,
        dataset,
        sampler,
        batch_size,
        label_source=None,
        label_key=None,
        executor=None,
        prefetch=0,
    ):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.label_source = label_source
        self.label_key = label_key
        self.executor = PlanExecutor() if executor is None else executor
        self.prefetch = int(prefetch)
        if self.prefetch < 0:
            raise ValueError("prefetch must be >= 0")

    def _dataset_iter(self):
        try:
            return iter(self.dataset)
        except TypeError:
            if hasattr(self.dataset, "__len__") and hasattr(self.dataset, "__getitem__"):
                return (self.dataset[index] for index in range(len(self.dataset)))
            raise TypeError("Loader dataset must be iterable or implement __len__ and __getitem__")

    def _resolve_sampled(self, sampled):
        if isinstance(sampled, SamplingPlan):
            context = self.executor.execute(sampled, graph=sampled.graph)
            return materialize_context(context)
        if isinstance(sampled, (list, tuple)):
            resolved = []
            for value in sampled:
                current = self._resolve_sampled(value)
                if isinstance(current, list):
                    resolved.extend(current)
                elif isinstance(current, tuple):
                    resolved.extend(list(current))
                else:
                    resolved.append(current)
            return resolved if isinstance(sampled, list) else tuple(resolved)
        return sampled

    def _sample_item(self, item):
        build_plan = getattr(self.sampler, "build_plan", None)
        if callable(build_plan):
            sampled = build_plan(item)
        else:
            sampled = self.sampler.sample(item)
        return self._resolve_sampled(sampled)

    def _build_batch(self, items):
        return materialize_batch(items, label_source=self.label_source, label_key=self.label_key)

    @staticmethod
    def _append_sampled(batch, sampled):
        if isinstance(sampled, (list, tuple)):
            batch.extend(sampled)
        else:
            batch.append(sampled)

    def __iter__(self):
        dataset_iter = self._dataset_iter()
        pending = deque()

        def fill_pending(limit):
            while len(pending) < limit:
                try:
                    item = next(dataset_iter)
                except StopIteration:
                    break
                pending.append(self._sample_item(item))

        fill_pending(self.batch_size + self.prefetch)
        while pending:
            batch = []
            for _ in range(min(self.batch_size, len(pending))):
                self._append_sampled(batch, pending.popleft())
            yield self._build_batch(batch)
            fill_pending(self.batch_size + self.prefetch)


DataLoader = Loader
