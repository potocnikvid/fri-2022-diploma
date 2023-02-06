from argparse import ArgumentError
from torch.nn.modules.activation import ReLU, SELU
from torch.nn.modules.batchnorm import BatchNorm1d
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


class M20_CGAN(LightningModule):

    def __init__(self,
        gen_lr=1e-4,
        dis_lr=1e-4,
        gen_weight_decay=0.0,
        dis_weight_decay=0.0,
        rw=0.0,
        verbose=False,
        **kwargs
    ):
        super().__init__()  
        self.save_hyperparameters()

        self.example_input_array = (torch.rand(1,1,18,512), torch.rand(1,1,18,512), torch.randn(1,1,18,64))

        self.stylegan = StyleGAN2(self.hparams.verbose)

        self._generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*512+64, 2*512),
                nn.Linear(2*512, 512),
            )
            for _ in range(18)
        ])

        self._discriminator = nn.ModuleList([
            nn.Sequential(
                self._get_discriminator_block(3*512, 3*256),
                self._get_discriminator_block(3*256, 3*128),
                self._get_discriminator_block(3*128, 3*64),
                self._get_discriminator_block(3*64, 3*32),
                nn.Linear(3*32, 1),
                nn.Sigmoid(),
            )
            for _ in range(18)
        ])

        self._discriminator_output = nn.Sequential(
            nn.Linear(18, 1),
        )

    def _get_discriminator_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def generator(self, father, mother, noise):
        batch_size = father.shape[0]
        assert father.shape == (batch_size, 1, 18, 512)
        assert mother.shape == (batch_size, 1, 18, 512)
        assert noise.shape == (batch_size, 1, 18, 64)

        child_hat = []
        for i, g in enumerate(self._generator):
            father_slice = father[:,:,i].flatten(1)
            mather_slice = mother[:,:,i].flatten(1)
            noise_slice = noise[:,:,i].flatten(1)
            input_slice = torch.cat([father_slice, mather_slice, noise_slice], dim=1)
            slice_hat = g(input_slice)
            child_hat += [slice_hat]
        
        child_hat =  torch.stack(child_hat, dim=1).unsqueeze(1)
        assert child_hat.shape == (batch_size, 1, 18, 512)

        return child_hat

    def discriminator(self, father, mother, child):
        batch_size = child.shape[0]
        assert father.shape == (batch_size, 1, 18, 512)
        assert mother.shape == (batch_size, 1, 18, 512)
        assert child.shape  == (batch_size, 1, 18, 512)

        real_hat = []
        for i, d in enumerate(self._discriminator):
            father_slice = father[:,:,i].flatten(1)
            mather_slice = mother[:,:,i].flatten(1)
            child_slice = child[:,:,i].flatten(1)
            input_slice = torch.cat([father_slice, mather_slice, child_slice], dim=1)
            slice_hat = d(input_slice)
            real_hat += [slice_hat]
        
        real_hat =  torch.cat(real_hat, dim=1)
        assert real_hat.shape == (batch_size, 18)
        real_hat = self._discriminator_output(real_hat)
        assert real_hat.shape == (batch_size, 1)

        return real_hat

    def forward(self, father, mother, noise):
        return self.generator(father, mother, noise)

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._step(batch, batch_idx, optimizer_idx, 1, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, None, 1, "validation")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, None, 6, "test")

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.parameters(), lr=self.hparams.gen_lr, weight_decay=self.hparams.gen_weight_decay)
        dis_opt = torch.optim.Adam(self.parameters(), lr=self.hparams.dis_lr, weight_decay=self.hparams.dis_weight_decay)
        return [gen_opt, dis_opt], []

    def _step(self, batch, batch_idx, optimizer_idx, log_results=0, label=None):
        father, mother, child, *_ = batch

        batch_size = father.shape[0]
        noise = torch.randn(batch_size, 1, 18, 64).type_as(father)

        if log_results > 0 and batch_idx == 0:
            self.logger.experiment.log({f"results/{label}": wandb.Image(self._construct_image_grid(batch, n=log_results))})

        # generator training step
        if optimizer_idx == 0:
            child_hat = self.generator(father, mother, noise)

            fake_hat = self.discriminator(father, mother, child_hat)

            child_hat_mse = F.mse_loss(child_hat, child)
            gen_loss = F.binary_cross_entropy_with_logits(fake_hat, torch.ones_like(fake_hat))
            gen_loss = self.hparams.rw * child_hat_mse + (1-self.hparams.rw) * gen_loss

            self.log('child_hat_mse', child_hat_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return gen_loss

        # discriminator training step
        elif optimizer_idx == 1:
            child_hat = self.generator(father, mother, noise).detach()

            real_hat = self.discriminator(father, mother, child)
            fake_hat = self.discriminator(father, mother, child_hat)

            dis_loss_real = F.binary_cross_entropy_with_logits(real_hat, torch.ones_like(real_hat))
            dis_loss_fake = F.binary_cross_entropy_with_logits(fake_hat, torch.zeros_like(real_hat))
            dis_loss = (dis_loss_real + dis_loss_fake) / 2

            self.log('dis_loss', dis_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return dis_loss

        # validation/test step
        else:
            child_hat = self.generator(father, mother, noise)

            real_hat = self.discriminator(father, mother, child)
            fake_hat = self.discriminator(father, mother, child_hat)

            child_hat_mse = F.mse_loss(child_hat, child)
            gen_loss = F.binary_cross_entropy_with_logits(fake_hat, torch.ones_like(fake_hat))
            gen_loss = self.hparams.rw * child_hat_mse + (1-self.hparams.rw) * gen_loss

            dis_loss_real = F.binary_cross_entropy_with_logits(real_hat, torch.ones_like(real_hat))
            dis_loss_fake = F.binary_cross_entropy_with_logits(fake_hat, torch.zeros_like(real_hat))
            dis_loss = (dis_loss_real + dis_loss_fake) / 2

            loss = (gen_loss + dis_loss) / 2

            self.log('child_hat_mse', child_hat_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('gen_loss',      gen_loss,      on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('dis_loss',      dis_loss,      on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('loss',          loss,          on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return loss

    def _construct_image_grid(self, batch, n=4):
        father, mother, child, f_label, m_label, c_label, f_image, m_image, c_image = batch
        batch_size = father.shape[0]
        idxs = np.random.choice(batch_size, n)

        father = father[idxs, :]
        mother = mother[idxs, :]
        child =  child[idxs, :]
        # child_gender =  child_gender[idxs]
        f_image =  np.array(f_image)[idxs]
        m_image =  np.array(m_image)[idxs]
        c_image =  np.array(c_image)[idxs]

        noise = torch.randn(n, 1, 18, 64).type_as(father)
        child_hat = self.generator(father, mother, noise)

        toTensor = torchvision.transforms.PILToTensor()
        toPIL = torchvision.transforms.ToPILImage()

        get_label = lambda x: "|".join(x.split("/")[-2:])[:-4]
        
        images = []
        for idx in range(n):
            images += [
                toTensor(utils.viz.image_add_label(Image.open(f_image[idx]), get_label(f_image[idx]), 50).convert("RGB").resize((256,256))),
                toTensor(utils.viz.image_add_label(Image.open(m_image[idx]), get_label(m_image[idx]), 50).convert("RGB").resize((256,256))),
                toTensor(utils.viz.image_add_label(Image.open(c_image[idx]), get_label(c_image[idx]), 50).convert("RGB").resize((256,256))),
                toTensor(self.stylegan.generate_from_array(child_hat[idx].detach().cpu().numpy())[0].resize((256,256))),
            ]
        return toPIL(torchvision.utils.make_grid(images, nrow=int(len(images)/n))).convert("RGB")

def main():
    config = {
        "gen_lr": 1e-5,
        "dis_lr": 1e-5,
        "gen_weight_decay": 1e-9,
        "dis_weight_decay": 1e-9,
        "rw": 0.8,
        "verbose": False,
        "batch_size": 512,
    }
    utils.core.train(M13_GAN, config=config, early_stopping_delta=None, max_epochs=100, test=True)


if __name__ == "__main__":
    main()