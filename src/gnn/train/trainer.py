class Trainer:
    def __init__(self, model, task, optimizer, lr, max_epochs):
        self.model = model
        self.task = task
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.max_epochs = max_epochs

    def fit(self, graph):
        for _ in range(self.max_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(graph)
            loss = self.task.loss(graph, logits, stage="train")
            loss.backward()
            self.optimizer.step()
        return {"epochs": self.max_epochs}
