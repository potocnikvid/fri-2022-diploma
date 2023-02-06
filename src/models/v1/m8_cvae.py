from torch.nn.modules.activation import ReLU
from models.v1.base_model import BaseModel
from models.wls.autoencoder import AutoEncoder
import utils.core
import utils.viz
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


class M8_CVAE(LightningModule):

    def __init__(self,
        ae_checkpoint,
        lr=1e-4,
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)  

        self.ae_checkpoint = ae_checkpoint
        self.lr = lr
        self.verbose = verbose

        self.save_hyperparameters()

        self.stylegan = StyleGAN2(self.verbose)
        self.ae = AutoEncoder.load_from_checkpoint(ae_checkpoint).eval()
        self.ae.freeze()

        self.encoder = nn.Sequential(
            nn.Linear(2*self.ae.latent_dim+1, 2*self.ae.latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(2*self.ae.latent_dim),
            nn.Linear(2*self.ae.latent_dim, self.ae.latent_dim),
        )

        self.fc_mu = nn.Linear(self.ae.latent_dim, 128)
        self.fc_var = nn.Linear(self.ae.latent_dim, 128)

        self.decoder = nn.Sequential(
            nn.Linear(128+1, 18*128),
            nn.ReLU(),
            nn.BatchNorm1d(18*128),
            nn.Linear(18*128, 18*256),
            nn.ReLU(),
            nn.BatchNorm1d(18*256),
            nn.Linear(18*256, 18*512),
        )

    def encode(self, father, mother, gender):
        father = father.view(-1, 18*512)
        mother = mother.view(-1, 18*512)
        gender = gender.view(-1, 1)

        father_zip = self.ae.encode(father)
        mother_zip = self.ae.encode(mother)

        child_enc = self.encoder(torch.cat([father_zip, mother_zip, gender], dim=1))
        mu = self.fc_mu(child_enc)
        log_var = self.fc_var(child_enc)

        return (child_enc, mu, log_var)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, gender):
        gender = gender.view(-1, 1)
        child_hat = self.decoder(torch.cat([z, gender], dim=1))
        return child_hat.view(-1, 1, 18, 512)

    def forward(self, father, mother, gender):
        child_enc, mu, log_var = self.encode(father, mother, gender)
        z = self.reparameterize(mu, log_var)
        child_hat = self.decode(z, gender)

        return (
            child_hat,
            mu,
            log_var
        )

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 0)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 1)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 6)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, batch, batch_idx, log_results=0):
        father, mother, child, child_gender, *other = batch

        child_hat, mu, log_var = self.forward(father, mother, child_gender)

        child_hat_mse = F.mse_loss(child_hat, child)
        kld = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = child_hat_mse + kld

        self.log('child_hat_mse', child_hat_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kld', kld, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if log_results > 0 and batch_idx == 0:
            try:
                self.logger.experiment.log({"results": wandb.Image(self._construct_image_grid(batch, n=log_results))})
            except Exception as ex:
                print(f"[WARN] {ex}")

        return loss

    def _construct_image_grid(self, batch, n=4):
        father, mother, child, child_gender, f_image, m_image, c_image = batch
        batch_size = father.shape[0]
        idxs = np.random.choice(batch_size, n)

        father = father[idxs, :]
        mother = mother[idxs, :]
        child =  child[idxs, :]
        child_gender =  child_gender[idxs]
        f_image =  np.array(f_image)[idxs]
        m_image =  np.array(m_image)[idxs]
        c_image =  np.array(c_image)[idxs]

        child_gender_inverse = torch.abs(child_gender-1)

        child_hat, *_ = self.forward(father, mother, child_gender)
        child_hat_gender_inverse, *_ = self.forward(father, mother, child_gender_inverse)

        toTensor = torchvision.transforms.PILToTensor()
        toPIL = torchvision.transforms.ToPILImage()

        get_label = lambda x: "|".join(x.split("/")[-2:])[:-4]

        images = []
        for idx in range(n):
            images += [
                toTensor(utils.viz.image_add_label(Image.open(f_image[idx]), get_label(f_image[idx]), 50).convert("RGB").resize((265,265))),
                toTensor(utils.viz.image_add_label(Image.open(m_image[idx]), get_label(m_image[idx]), 50).convert("RGB").resize((265,265))),
                toTensor(utils.viz.image_add_label(Image.open(c_image[idx]), get_label(c_image[idx]), 50).convert("RGB").resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat[idx].cpu().numpy()).resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat_gender_inverse[idx].cpu().numpy()).resize((265,265))),
            ]
        return toPIL(torchvision.utils.make_grid(images, nrow=5)).convert("RGB")

def main():
    config = {
        "lr": 1e-4,
        "verbose": False,
        "ae_checkpoint": ".cache/wandb/run-20210411_194322-3tx0oob8/files/fri-2021-masters-main-src_models_wls/3tx0oob8/checkpoints/epoch=25-step=1090.ckpt"
    }
    utils.core.train(M8_CVAE(**config), batch_size=64, early_stopping_delta=0.0)


if __name__ == "__main__":
    main()