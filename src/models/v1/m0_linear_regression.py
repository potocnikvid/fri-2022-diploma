from models.v1.base_model import BaseModel
import utils.core
import utils.viz
from pprint import pprint

import wandb
import torch
from torch import nn


class M0_LinearRegression(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Linear(2*self.latent_vector_size, self.latent_vector_size)
        )


    def forward(self, father, mother):
        father = father.view(-1, self.latent_vector_size)
        mother = mother.view(-1, self.latent_vector_size)
        return self.net(torch.cat([father, mother], 1)).view(-1, *self.latent_vector_shape)


def main():
    config = {
        "lr": 1e-4,
        "verbose": True,
    }
    utils.core.train(M0_LinearRegression(**config))


if __name__ == "__main__":
    main()
