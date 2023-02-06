from models.v1.base_model import BaseModel
from models.wls.autoencoder import AutoEncoder
import utils.core
import utils.viz
from pprint import pprint

import wandb
import torch
from torch import nn


class M0_LinearRegression_Zip(BaseModel):

    def __init__(self,
        ae_checkpoint,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ae_checkpoint = ae_checkpoint

        self.zipper = AutoEncoder.load_from_checkpoint(ae_checkpoint).eval()
        self.zipper.freeze()

        self.net = nn.Sequential(
            nn.Linear(2*self.zipper.latent_dim, self.zipper.latent_dim)
        )


    def forward(self, father, mother):
        father = father.view(-1, self.latent_vector_size)
        mother = mother.view(-1, self.latent_vector_size)
        father = self.zipper.encode(father)
        mother = self.zipper.encode(mother)
        return self.zipper.decode(self.net(torch.cat([father, mother], 1))).view(-1, *self.latent_vector_shape)


def main():
    config = {
        "lr": 1e-3,
        "ae_checkpoint": ".cache/wandb/run-20210411_194322-3tx0oob8/files/fri-2021-masters-main-src_models_wls/3tx0oob8/checkpoints/epoch=25-step=1090.ckpt"
    }
    utils.core.train(M0_LinearRegression_Zip(**config))


if __name__ == "__main__":
    main()
