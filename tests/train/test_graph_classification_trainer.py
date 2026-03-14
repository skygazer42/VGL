import torch
from torch import nn

from vgl.train.tasks import GraphClassificationTask
from vgl.train.trainer import Trainer


class FakeBatch:
    def __init__(self):
        self.labels = torch.tensor([1, 0])
        self.metadata = [{"label": 1}, {"label": 0}]


class TinyGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return self.linear(torch.randn(batch.labels.size(0), 4))


def test_trainer_runs_graph_classification_epoch_over_batches():
    model = TinyGraphClassifier()
    trainer = Trainer(
        model=model,
        task=GraphClassificationTask(target="y", label_source="graph"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit([FakeBatch(), FakeBatch()])

    assert history["epochs"] == 1
    assert model.calls == 2

