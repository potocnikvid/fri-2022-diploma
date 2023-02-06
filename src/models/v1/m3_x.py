from models.v1.base_model import BaseModel
import utils.core
import utils.viz
from pprint import pprint

import wandb
import torch
from torch import nn


class M3_X(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(self.latent_vector_size, int(3*self.latent_vector_size/4)),
            nn.ReLU(),
            nn.Linear(int(3*self.latent_vector_size/4), int(self.latent_vector_size/2)),
        )

        self.decoder = nn.Sequential(
            nn.Linear(int(self.latent_vector_size/2), self.latent_vector_size)
        )


    def forward(self, father, mother):
        father = father.view(-1, self.latent_vector_size)
        mother = mother.view(-1, self.latent_vector_size)

        father_enc = self.encoder(father)
        mother_enc = self.encoder(mother)
        random = torch.randn(father_enc.shape, device=self.device)

        return self.decoder(father_enc + mother_enc + random).view(-1, *self.latent_vector_shape)


def main():
    config = {
        "lr": 1e-5
    }
    utils.core.train(M3_X(**config))


if __name__ == "__main__":
    main()
