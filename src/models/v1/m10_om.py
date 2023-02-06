from argparse import ArgumentError
from torch.nn.modules.activation import ReLU
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


class M10_Om(LightningModule):

    def __init__(self,
        ae_checkpoint,
        lr=1e-4,
        weight_decay=0.5,
        verbose=False,
        rand="max",
        **kwargs
    ):
        super().__init__(**kwargs)  

        self.ae_checkpoint = ae_checkpoint
        self.lr = lr
        self.weight_decay = weight_decay
        self.rand = rand
        self.verbose = verbose

        self.save_hyperparameters()

        self.stylegan = StyleGAN2(self.verbose)
        self.ae = AutoEncoder.load_from_checkpoint(ae_checkpoint).eval()
        self.ae.freeze()

        self.encoder = nn.Sequential(
            nn.Linear(self.ae.latent_dim+1, self.ae.latent_dim),
            nn.BatchNorm1d(self.ae.latent_dim),
            nn.ReLU(),
            nn.Linear(self.ae.latent_dim, 2*self.ae.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2*self.ae.latent_dim+1, 2*self.ae.latent_dim),
            nn.BatchNorm1d(2*self.ae.latent_dim),
            nn.ReLU(),
            nn.Linear(2*self.ae.latent_dim, self.ae.latent_dim),
        )

    def encode(self, father, mother, gender, rand=None):
        batch_size = father.shape[0]
        father = father.view(batch_size, 18*512)
        mother = mother.view(batch_size, 18*512)
        gender = gender.view(batch_size, 1)

        father_zip = self.ae.encode(father)
        mother_zip = self.ae.encode(mother)
        father_enc = self.encoder(torch.cat([father_zip, torch.ones(batch_size, 1, device=self.device) ], dim=1))
        mother_enc = self.encoder(torch.cat([mother_zip, torch.zeros(batch_size, 1, device=self.device)], dim=1))
        child_enc = self.random_sample(father_enc, mother_enc, rand)

        return child_enc

    def random_sample(self, father, mother, rand="max"):
        if rand == "max" or rand is None:
            return torch.max(father, mother)
        elif rand == "randint":
            r = torch.randint_like(father, 2)
            return r * father + (1 - r) * mother
        elif rand == "rand":
            r = torch.rand_like(father)
            return r * father + (1 - r) * mother
        # elif torch.is_tensor(rand) and rand.shape == father.shape:
        #     r = rand
        #     return r * father + (1 - r) * mother
        else:
            raise ArgumentError()

    def decode(self, child_enc, gender):
        gender = gender.view(-1, 1)
        child_zip = self.decoder(torch.cat([child_enc, gender], dim=1))
        child_hat = self.ae.decode(child_zip)
        return child_hat.view(-1, 1, 18, 512)

    def forward(self, father, mother, gender, rand=None):
        child_enc = self.encode(father, mother, gender, rand)
        child_hat = self.decode(child_enc, gender)
        return child_hat

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 0)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 1)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 6)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _step(self, batch, batch_idx, log_results=0):
        father, mother, child, child_gender, *other = batch

        child_hat = self.forward(father, mother, child_gender, rand="max")

        child_hat_mse = F.mse_loss(child_hat, child)
        loss = child_hat_mse

        self.log('child_hat_mse', child_hat_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if log_results > 0 and batch_idx == 0:
            self.logger.experiment.log({"results": wandb.Image(self._construct_image_grid(batch, n=log_results))})

        return loss

    def _construct_image_grid(self, batch, n=4):
        father, mother, child, child_gender, f_image, m_image, c_image = batch
        batch_size = father.shape[0]
        idxs = np.random.choice(batch_size, n)

        pprint(idxs.shape)

        father = father[idxs, :]
        mother = mother[idxs, :]
        child =  child[idxs, :]
        child_gender =  child_gender[idxs]
        f_image =  np.array(f_image)[idxs]
        m_image =  np.array(m_image)[idxs]
        c_image =  np.array(c_image)[idxs]

        child_gender_inverse = torch.abs(child_gender-1)

        child_hat_0 = self.forward(father, mother, child_gender, "max")
        child_hat_1 = self.forward(father, mother, child_gender, "randint")
        child_hat_gender_inverse_0 = self.forward(father, mother, child_gender_inverse, "max")
        child_hat_gender_inverse_1 = self.forward(father, mother, child_gender_inverse, "randint")

        toTensor = torchvision.transforms.PILToTensor()
        toPIL = torchvision.transforms.ToPILImage()

        get_label = lambda x: "|".join(x.split("/")[-2:])[:-4]

        images = []
        for idx in range(n):
            images += [
                toTensor(utils.viz.image_add_label(Image.open(f_image[idx]), get_label(f_image[idx]), 50).convert("RGB").resize((265,265))),
                toTensor(utils.viz.image_add_label(Image.open(m_image[idx]), get_label(m_image[idx]), 50).convert("RGB").resize((265,265))),
                toTensor(utils.viz.image_add_label(Image.open(c_image[idx]), get_label(c_image[idx]), 50).convert("RGB").resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat_0[idx].cpu().numpy()).resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat_1[idx].cpu().numpy()).resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat_gender_inverse_0[idx].cpu().numpy()).resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat_gender_inverse_1[idx].cpu().numpy()).resize((265,265))),
            ]
        return toPIL(torchvision.utils.make_grid(images, nrow=7)).convert("RGB")

def main():
    config = {
        "lr": 1e-4,
        "weight_decay": 0.5,
        "verbose": False,
        "rand": "max",
        "ae_checkpoint": ".cache/wandb/run-20210411_194322-3tx0oob8/files/fri-2021-masters-main-src_models_wls/3tx0oob8/checkpoints/epoch=25-step=1090.ckpt"
    }
    utils.core.train(M10_Om(**config), batch_size=64, early_stopping_delta=0.0, max_epochs=10)


if __name__ == "__main__":
    main()