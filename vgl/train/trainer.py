from collections.abc import Iterable


class Trainer:
    def __init__(self, model, task, optimizer, lr, max_epochs):
        self.model = model
        self.task = task
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.max_epochs = max_epochs

    def _batches(self, data):
        if (
            hasattr(data, "nodes")
            or hasattr(data, "graphs")
            or hasattr(data, "labels")
            or hasattr(data, "x")
        ):
            return [data]
        if isinstance(data, Iterable):
            return data
        return [data]

    def fit(self, data):
        for _ in range(self.max_epochs):
            for batch in self._batches(data):
                self.model.train()
                self.optimizer.zero_grad()
                logits = self.model(batch)
                loss = self.task.loss(batch, logits, stage="train")
                loss.backward()
                self.optimizer.step()
        return {"epochs": self.max_epochs}
