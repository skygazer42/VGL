import torch

from examples.homo.link_prediction import TinyLinkPredictor, build_demo_loader, build_split_demo_loaders
from vgl.engine import Trainer
from vgl.tasks import LinkPredictionTask


def test_end_to_end_link_prediction_runs():
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(build_demo_loader())

    assert result["epochs"] == 1


def test_end_to_end_link_prediction_split_pipeline_runs():
    train_loader, val_loader, test_loader = build_split_demo_loaders()
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label", metrics=["mrr", "filtered_mrr", "filtered_hits@1"]),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(train_loader, val_data=val_loader)
    result = trainer.test(test_loader)

    assert history["epochs"] == 1
    assert "mrr" in history["val"][0]
    assert "filtered_mrr" in history["val"][0]
    assert "filtered_hits@1" in result
