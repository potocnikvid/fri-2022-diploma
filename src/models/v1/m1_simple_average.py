import utils.core
import utils.viz
from models.v1.base_model import BaseModel

from pprint import pprint
import wandb
import torch
from torch import nn


class M1_SimpleAverage(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.father_translation = nn.Parameter(torch.randn(self.latent_vector_size, device=self.device))
        self.mother_translation = nn.Parameter(torch.randn(self.latent_vector_size, device=self.device))
        self.register_parameter('father_translation', self.father_translation)
        self.register_parameter('mother_translation', self.mother_translation)
    

    def forward(self, father, mother):
        father = father.view(-1, self.latent_vector_size)
        mother = mother.view(-1, self.latent_vector_size)
        return torch.mean(torch.stack([
            father + self.father_translation,
            mother + self.mother_translation
        ]), 0).view(-1, *self.latent_vector_shape)


def main():
    config = {
        "lr": 1e-2
    }
    utils.core.train(M1_SimpleAverage(**config), batch_size=1024)


if __name__ == "__main__":
    main()
