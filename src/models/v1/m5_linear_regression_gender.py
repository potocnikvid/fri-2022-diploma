from models.v1.base_model import BaseModel
import utils.core
import utils.viz
from pprint import pprint


from PIL import Image

import numpy as np
import wandb
import torch, torchvision
import torch.nn.functional as F
from torch import nn


class M5_LinearRegression_Gender(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Linear(2*self.latent_vector_size, self.latent_vector_size)
        )

        self.one = nn.Parameter(torch.ones(1))
        self.gender_avg_diff_size = nn.Parameter(torch.zeros(1))
        self.gender_avg_diff = nn.Parameter(torch.zeros(*self.latent_vector_shape))


    def forward(self, father, mother):
        father = father.view(-1, self.latent_vector_size)
        mother = mother.view(-1, self.latent_vector_size)
        return self.net(torch.cat([father, mother], dim=1)).view(-1, *self.latent_vector_shape)

    def _step(self, batch, batch_idx, log_results=False):
        loss = super()._step(batch, batch_idx, log_results)

        if self.training:
            self._add_to_gender_avg_diff(batch)

        return loss

    def _add_to_gender_avg_diff(self, batch):
        father, mother, *other = batch

        with torch.no_grad():
            batch_mean = torch.mean(father - mother, dim=0)

            self.gender_avg_diff = nn.Parameter((self.gender_avg_diff_size * self.gender_avg_diff + batch_mean) / (self.gender_avg_diff_size + self.one))
            self.gender_avg_diff_size = nn.Parameter(self.gender_avg_diff_size + self.one)

            self.logger.experiment.log({"gender_avg_diff": wandb.Histogram(self.gender_avg_diff.cpu())}, step=self.global_step)

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

        child_hat = self.forward(father, mother)
        child_hat_add = child_hat + self.gender_avg_diff
        child_hat_diff = child_hat - self.gender_avg_diff

        toTensor = torchvision.transforms.PILToTensor()
        toPIL = torchvision.transforms.ToPILImage()

        images = []
        for idx in range(4):
            images += [
                toTensor(Image.open(f_image[idx]).convert("RGB").resize((265,265))),
                toTensor(Image.open(m_image[idx]).convert("RGB").resize((265,265))),
                toTensor(Image.open(c_image[idx]).convert("RGB").resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat[idx].cpu().numpy()).resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat_add[idx].cpu().numpy()).resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat_diff[idx].cpu().numpy()).resize((265,265))),
            ]
        return toPIL(torchvision.utils.make_grid(images, nrow=6)).convert("RGB")


def main():
    config = {
        "lr": 1e-5,
        "verbose": True,
    }
    utils.core.train(M5_LinearRegression_Gender(**config))


if __name__ == "__main__":
    main()
