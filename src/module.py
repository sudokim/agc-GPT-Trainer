from pytorch_lightning import LightningModule
from torch.optim import Adam

from transformers import AutoModelForCausalLM


class GPTModule(LightningModule):
    def __init__(self, model_path: str, lr: float, **kwargs):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.lr = lr

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        return optimizer

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss
