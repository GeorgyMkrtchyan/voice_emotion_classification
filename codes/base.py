import typing as tp
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, BackboneFinetuning
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import F1Score, Accuracy  #TODO
from torch.utils.data import Dataset, DataLoader


class BaseClassificationModel(pl.LightningModule):
    """
    Model is trained as classification model.

    The whole piece of data in the input has single label.

    The model is trained using (`sequence`, `game_context`, `label`) representation of a batch.
    `game_context` might have shape (`batch_size`, 12) of (`batch_size`, 0)

    Method `forward` has to be implemented
    """
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()

        #TODO get validation metrics
        self.f1 = F1Score()
        self.val_accuracy = Accuracy()
        self.train_accuracy = Accuracy()

    def training_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        x, ctx, labels = batch
        logits = self(x, ctx)
        loss = self.loss_fn(logits, labels)

        self.train_accuracy.update(logits, labels)
        self.log("train_accuracy", self.train_accuracy, on_epoch=True)

        self.log("train_loss", loss, on_step=True)

        return loss
    
    def validation_step(self, batch:tuple, batch_idx:int) -> torch.Tensor:
        x, ctx, label = batch
        logits = self(x, ctx)
        loss = self.loss_fn(logits, label)

        self.f1.update(logits, label)
        self.val_accuracy.update(logits, label)

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_F1", self.f1, on_epoch=True)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True)


    def predict_step(self, batch:tuple, batch_idx: int) -> torch.Tensor:
        x, ctx = batch
        logits = self(x, ctx)
        preds = torch.argmax(logits, -1)
        return preds

    def configure_optimizers(self):
        # default config, can be reimplemeted
        opt = torch.optim.Adam(self.parameters())
        return opt

def train(model:pl.LightningModule, train_dataloader, val_dataloader, max_epochs=100, **trainer_kwargs):
    log_dir = os.path.join("./logs/", type(model).__name__)
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          monitor="val_F1",
                                          mode="max",
                                          filename="{epoch}-{val_F1}")
    logger = TensorBoardLogger(log_dir)
    trainer = pl.Trainer(logger=logger,
                         log_every_n_steps=5,
                         max_epochs=max_epochs,
                         check_val_every_n_epoch=5,
                         callbacks=[LearningRateMonitor(), checkpoint_callback],
                         **trainer_kwargs)

    trainer.fit(model, train_dataloader, val_dataloader)
    return model, trainer