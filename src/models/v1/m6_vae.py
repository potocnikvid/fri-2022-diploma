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


class M6_VAE(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(2*self.latent_vector_size, self.latent_vector_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.latent_vector_size, self.latent_vector_size),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(self.latent_vector_size, self.latent_vector_size)
        self.fc_var = nn.Linear(self.latent_vector_size, self.latent_vector_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_vector_size, self.latent_vector_size)
        )

    def encode(self, father, mother):
        father = father.view(-1, self.latent_vector_size)
        mother = mother.view(-1, self.latent_vector_size)

        enc = self.encoder(torch.cat([father, mother], dim=1))
        mu = self.fc_mu(enc)
        log_var = self.fc_var(enc)

        return (mu, log_var)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        dec = self.decoder(z)
        return dec.view(-1, *self.latent_vector_shape)

    def forward(self, father, mother):
        mu, log_var = self.encode(father, mother)
        z = self.reparameterize(mu, log_var)

        return (self.decode(z), mu, log_var)

    def _step(self, batch, batch_idx, log_results=False):
        father, mother, child, child_gender, *other = batch

        child_hat, mu, log_var = self.forward(father, mother)

        child_hat_mse = F.mse_loss(child_hat, child)
        kld = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = child_hat_mse + kld * 0.000001

        self.log('child_hat_mse', child_hat_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kld', kld, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if log_results and batch_idx == 0:
            self.logger.experiment.log({"results": wandb.Image(self._construct_image_grid(batch))})

        return loss

def main():
    config = {
        "lr": 1e-5,
        "verbose": True,
    }
    utils.core.train(M6_VAE(**config))


if __name__ == "__main__":
    main()
