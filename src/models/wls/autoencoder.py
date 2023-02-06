from logging import log
from torch.nn.modules.activation import ReLU
from models.v1.base_model import BaseModel
import utils.core
import utils.viz
from pprint import pprint

import wandb
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint

from utils.stylegan import StyleGAN2

from PIL import Image
from pprint import pprint

import numpy as np
import wandb
import torch, torchvision
from torch import nn
from pytorch_lightning import LightningModule

from dataset.wls import WLSDataModule

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


class AutoEncoder(LightningModule):

    def __init__(self, 
        input_dim = 512,
        hidden_dims = [],
        latent_dim = 256,
        lr = 1e-4,
        verbose = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.lr = lr
        self.verbose = verbose

        self.save_hyperparameters()

        dims = [input_dim] + hidden_dims + [latent_dim]
        self.encoder = self._build_encoder(dims)
        self.decoder = self._build_decoder(dims)

        self.stylegan = StyleGAN2(verbose=self.verbose)

    def _build_encoder(self, dims):
        seq = nn.Sequential()
        dims = list(zip(dims[:-1], dims[1:]))
        for i, (dim_in, dim_out) in enumerate(dims):
            seq.add_module(f"fc_{i}", nn.Linear(dim_in, dim_out))
            if i < len(dims) - 1:
                seq.add_module(f"bn_{i}", nn.BatchNorm1d(dim_out))
                seq.add_module(f"relu_{i}", nn.ReLU())
        return seq

    def _build_decoder(self, dims):
        seq = nn.Sequential()
        dims = list(zip(dims[::-1][:-1], dims[::-1][1:]))
        for i, (dim_in, dim_out) in enumerate(dims):
            seq.add_module(f"fc_{i}", nn.Linear(dim_in, dim_out))
            if i < len(dims) - 1:
                seq.add_module(f"bn_{i}", nn.BatchNorm1d(dim_out))
                seq.add_module(f"relu_{i}", nn.ReLU())
        return seq

    def encode(self, input):
        input = input.view(-1, self.input_dim)
        return self.encoder(input)

    def decode(self, z):
        z = z.view(-1, self.latent_dim)
        return self.decoder(z)

    def forward(self, input):
        z = self.encode(input)
        input_hat = self.decode(z)
        return z, input_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 0)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 1)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 4)

    def _step(self, batch, batch_idx, log_results=False):
        ws, raw_image, *other = batch
        ws = ws.view(-1, self.input_dim)

        z, ws_hat = self.forward(ws)

        loss = F.mse_loss(ws_hat, ws)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if log_results > 0 and batch_idx == 0:
            toTensor = torchvision.transforms.PILToTensor()
            toPIL = torchvision.transforms.ToPILImage()
            images = []
            for i in range(log_results):
                label = "|".join(raw_image[i].split("/")[-2:])[:-4]
                images += [
                    toTensor(utils.viz.image_add_label(Image.open(raw_image[i]), label, 50).convert("RGB").resize((265,265))),
                    toTensor(self.stylegan.generate_from_array(ws[i].cpu().numpy()).resize((265,265))),
                    toTensor(self.stylegan.generate_from_array(ws_hat[i].cpu().numpy()).resize((265,265))),
                ]
            self.logger.experiment.log({"results": wandb.Image(toPIL(torchvision.utils.make_grid(images, nrow=3)).convert("RGB"))})

        return loss

def main():
    config = {
        "input_dim": 18*512,
        "hidden_dims": [18*256],
        "latent_dim": 18*128,
        "lr": 1e-4,
        "verbose": False,
    }
    model = AutoEncoder(**config)

    datamodule = WLSDataModule(batch_size=64, num_workers=4)

    model_name = model.__class__.__name__.lower()  
    logger = WandbLogger(name=model_name, log_model=False)
    logger.watch(model, log=None)

    trainer = Trainer(
        logger=logger,
        gpus=-1,
        precision=16,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=0),
        ],
        max_epochs=100
    )
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()