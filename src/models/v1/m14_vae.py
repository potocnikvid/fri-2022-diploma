from argparse import ArgumentError
from torch.nn.modules.activation import ReLU, SELU
from torch.nn.modules.container import Sequential
from models.v1.base_model import BaseModel
from models.wls.autoencoder import AutoEncoder
import utils.core
import utils.viz
from pprint import pprint

from PIL import Image
import numpy as np
import wandb
import torch, torchvision
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint
from pytorch_lightning import LightningModule

from utils.stylegan import StyleGAN2


class M14_VAE(LightningModule):

    def __init__(self,
        lr=1e-4,
        weight_decay=0.0,
        dropout_p=0.0,
        verbose=False,
        **kwargs
    ):
        super().__init__()  
        self.save_hyperparameters()

        self.example_input_array = (torch.rand(128,1,18,512), torch.rand(128,1,18,512))

        self.stylegan = StyleGAN2(self.hparams.verbose)
        
        self._encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 768),
                nn.Linear(768, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
            )
            for _ in range(18)
        ])

        self._mu = nn.ModuleList([
            nn.Linear(64, 2)
            for _ in range(18)
        ])

        self._log_var = nn.ModuleList([
            nn.Linear(64, 2)
            for _ in range(18)
        ])

        self._decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 8),
                nn.Linear(8, 32),
                nn.Linear(32, 128),
                nn.Linear(128, 512),
            )
            for _ in range(18)
        ])

        # self._discriminator = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(3*512, 3*128),
        #         nn.Linear(3*128, 3*32),
        #         nn.Linear(3*32, 3*8),
        #         nn.Linear(3*8, 1),
        #         nn.Sigmoid()
        #     )
        #     for _ in range(18)
        # ])
        # self._discriminator_decision = nn.Linear(18, 1)

    def encode(self, father, mother):
        batch_size = father.shape[0]
        assert father.shape == (batch_size, 1, 18, 512)
        assert mother.shape == (batch_size, 1, 18, 512)

        z = []
        mu = []
        log_var = []
        for i in range(18):
            father_slice = father[:,:,i].flatten(1)
            mather_slice = mother[:,:,i].flatten(1)
            input_slice = torch.cat([father_slice, mather_slice], dim=1)

            z_slice = self._encoder[i](input_slice)
            mu_slice = self._mu[i](z_slice)
            log_var_slice = self._log_var[i](z_slice)

            z += [z_slice]
            mu += [mu_slice]
            log_var += [log_var_slice]
        
        z =  torch.stack(z, dim=1)
        mu =  torch.stack(mu, dim=1)
        log_var =  torch.stack(log_var, dim=1)

        assert mu.shape == (batch_size, 18, 2)
        assert log_var.shape == (batch_size, 18, 2)

        return mu, log_var

    def decoder(self, z):
        batch_size = z.shape[0]
        assert z.shape == (batch_size, 18, 2)

        child_hat = []
        for i in range(18):
            z_slice = z[:,i].flatten(1)
            child_hat_slice = self._decoder[i](z_slice)
            child_hat += [child_hat_slice]
        
        child_hat =  torch.stack(child_hat, dim=1).unsqueeze(1)
        
        assert child_hat.shape == (batch_size, 1, 18, 512)

        return child_hat

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, father, mother):
        mu, log_var = self.encode(father, mother)
        z = self.reparameterize(mu, log_var)
        child_hat = self.decoder(z)

        return child_hat, mu, log_var
        
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 1, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 1, "validation")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 6, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _step(self, batch, batch_idx, log_results=0, label=None):
        father, mother, child, f_label, m_label, c_label, f_image, m_image, c_image = batch

        child_hat, mu, log_var = self.forward(father, mother)

        rec_loss = F.mse_loss(child_hat, child)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 2))
        loss = rec_loss + kld_loss

        self.log('rec_loss', rec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kld_loss', kld_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if log_results > 0 and batch_idx == 0:
            self.logger.experiment.log({f"results/{label}": wandb.Image(self._construct_image_grid(batch, n=log_results))})

        return loss

    def _construct_image_grid(self, batch, n=4):
        father, mother, child, f_label, m_label, c_label, f_image, m_image, c_image = batch
        batch_size = father.shape[0]
        idxs = np.random.choice(batch_size, n)

        father = father[idxs, :]
        mother = mother[idxs, :]
        child =  child[idxs, :]
        f_image = np.array(f_image)[idxs]
        m_image = np.array(m_image)[idxs]
        c_image = np.array(c_image)[idxs]

        child_hat_0, mu, log_var = self.forward(father, mother)
        child_hat_1 = self.decoder(self.reparameterize(mu, log_var))
        child_hat_2 = self.decoder(self.reparameterize(mu, log_var))

        toTensor = torchvision.transforms.PILToTensor()
        toPIL = torchvision.transforms.ToPILImage()

        images = []
        for idx in range(n):
            images += [
                toTensor(Image.open(f_image[idx]).convert("RGB").resize((256,256))),
                toTensor(Image.open(m_image[idx]).convert("RGB").resize((256,256))),
                toTensor(Image.open(c_image[idx]).convert("RGB").resize((256,256))),
            ]
            ws = torch.stack([
                # father[idx],
                # mother[idx],
                child[idx],
                child_hat_0[idx],
                child_hat_1[idx],
                child_hat_2[idx],
            ]).view(-1, 18, 512).detach().cpu().numpy()
            images += [toTensor(x.resize((256,256))) for x in self.stylegan.generate_from_array(ws)]
        return toPIL(torchvision.utils.make_grid(images, nrow=int(len(images)/n))).convert("RGB")

def main():
    config = {
        "lr": 1e-5,
        "weight_decay": 1e-9,
        "verbose": False,
        "batch_size": 512,
    }
    utils.core.train(M14_VAE, config=config, early_stopping_delta=0.00001, max_epochs=100, test=True)


if __name__ == "__main__":
    main()