from torch.nn.modules.activation import ReLU
from models.v1.base_model import BaseModel
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


class M7_CVAE_V3(BaseModel):

    def __init__(self,
        beta=1e-6,
        **kwargs
    ):
        self.beta = beta

        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(2*self.latent_vector_size+1, self.latent_vector_size, bias=False),
            nn.BatchNorm1d(self.latent_vector_size),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(self.latent_vector_size, 9*512),
            nn.BatchNorm1d(9*512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc_mu = nn.Linear(9*512, 512)
        self.fc_var = nn.Linear(9*512, 512)

        self.decoder = nn.Sequential(
            nn.Linear(512+1, 9*512),
            nn.BatchNorm1d(9*512),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(9*512, 18*512),
        )

    def encode(self, father, mother, gender):
        father = father.view(-1, self.latent_vector_size)
        mother = mother.view(-1, self.latent_vector_size)
        gender = gender.view(-1, 1)

        child_enc = torch.utils.checkpoint.checkpoint_sequential(self.encoder, 2, torch.cat([father, mother, gender], dim=1))
        mu = self.fc_mu(child_enc)
        log_var = self.fc_var(child_enc)

        return (mu, log_var)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, gender):
        gender = gender.view(-1, 1)
        child_hat = torch.utils.checkpoint.checkpoint_sequential(self.decoder, 2, torch.cat([z, gender], dim=1))
        return child_hat.view(-1, *self.latent_vector_shape)

    def forward(self, father, mother, gender):
        mu, log_var = self.encode(father, mother, gender)
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
        return self._step(batch, batch_idx, 0)

    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 6)

    def _step(self, batch, batch_idx, log_results=0):
        father, mother, child, child_gender, *other = batch

        child_hat, mu, log_var = self.forward(father, mother, child_gender)

        child_hat_mse = F.mse_loss(child_hat, child)
        kld = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = child_hat_mse + kld * self.beta

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

        images = []
        for idx in range(n):
            images += [
                toTensor(Image.open(f_image[idx]).convert("RGB").resize((265,265))),
                toTensor(Image.open(m_image[idx]).convert("RGB").resize((265,265))),
                toTensor(Image.open(c_image[idx]).convert("RGB").resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat[idx].cpu().numpy()).resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat_gender_inverse[idx].cpu().numpy()).resize((265,265))),
            ]
        return toPIL(torchvision.utils.make_grid(images, nrow=5)).convert("RGB")

def main():
    config = {
        "lr": 1e-4,
        "verbose": True,
        "beta": 1,
    }
    utils.core.train(M7_CVAE_V3(**config), batch_size=128)


if __name__ == "__main__":
    main()