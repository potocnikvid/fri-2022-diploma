import numpy as np
import wandb

import sys

import torch, torchvision
from torch import nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils.core import assert_batch_shape
from utils.viz import image_add_label
from utils.stylegan import StyleGAN2
from dataset.nok_aug import NokAugDataModule
from dataset.nok_mean import NokMeanDataModule

from pprint import pprint

class LinearRegression(LightningModule):

    def __init__(self, **kwargs):
        super().__init__()  
        self.save_hyperparameters()

        self.stylegan = StyleGAN2()

        self.ns = nn.ModuleList([
            nn.Linear(1024, 512)
            for _ in range(18)
        ])

    def forward(self, father, mother):
        assert_batch_shape(father, (18, 512))
        assert_batch_shape(mother, (18, 512))

        child_hat = []
        for i, n in enumerate(self.ns):
            father_slice = father[:,i]
            mather_slice = mother[:,i]
            input_slice = torch.cat([father_slice, mather_slice], dim=1)
            slice_hat = n(input_slice)
            child_hat += [slice_hat]

        child_hat =  torch.stack(child_hat, dim=1)
        return child_hat

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 0, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 1, "validation")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 6, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _step(self, batch, batch_idx, log_results, label):
        father, mother, child, child_gender, labels = batch

        if log_results > 0 and batch_idx == 0:
            self.logger.experiment.log({f"results/{label}": wandb.Image(self._sample_results(batch, n=log_results))})

        child_hat = self.forward(father, mother)

        child_mse = F.mse_loss(child_hat, child)
        loss = child_mse

        self.log('child_mse', child_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def _sample_results(self, batch, n=4):
        father, mother, child, child_gender, labels = batch
        batch_size = father.shape[0]
        idxs = np.random.choice(batch_size, n)

        father = father[idxs, :]
        mother = mother[idxs, :]
        child =  child[idxs, :]
        child_gender =  child_gender[idxs]
        labels =  np.array(labels)[idxs]

        child_hat = self.forward(father, mother)

        toTensor = torchvision.transforms.PILToTensor()
        toPIL = torchvision.transforms.ToPILImage()

        images = []
        for idx in range(n):
            ws = torch.stack([
                father[idx],
                mother[idx],
                child[idx],
                child_hat[idx],
            ]).detach().cpu().numpy()
            wsl = [labels[idx], "", "", ""]
            images += [toTensor(image_add_label(img, label, 50).resize((256,256))) for img, label in zip(self.stylegan.generate_from_array(ws), wsl)]
        return toPIL(torchvision.utils.make_grid(images, nrow=int(len(images)/n))).convert("RGB")

def main():
    model_name = "linear-regression"
    example_input = (torch.randn(1, 18, 512), torch.randn(1, 18, 512))
    config = {
        "lr": 1e-4,
        "weight_decay": 1e-6,
    }
    max_epochs = 100

    with wandb.init(name=model_name, config=config, save_code=True):
        logger = WandbLogger(log_model=False)
        config = wandb.config
        
        data = NokAugDataModule(batch_size=128, inflate=10, num_workers=4)

        model = LinearRegression(**config)
        logger.watch(model, log=None)

        onnx_file = f"{logger.save_dir}/model.onnx"
        torch.onnx.export(
            model,
            example_input,
            onnx_file,
            opset_version=9,
            export_params=False,
            do_constant_folding=False,
            input_names=["father", "mother"],
            output_names=["child"],
            dynamic_axes={
                "father": {0: "batch_size"},
                "mother": {0: "batch_size"},
                "child": {0: "batch_size"},
            }
        )

        trainer = Trainer(
            logger=logger,
            gpus=-1,
            precision=16,
            callbacks= [
                ModelCheckpoint(monitor="loss", save_top_k=1, mode="min", filename="best-{epoch}-{loss:.2f}"),
                # EarlyStopping(monitor="loss", min_delta=0.001, patience=5)
            ],
            max_epochs=max_epochs,
            check_val_every_n_epoch=10
        )

        trainer.fit(model, datamodule=data)

        # test on the best model checkpoint
        model = LinearRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()