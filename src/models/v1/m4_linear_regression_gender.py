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


class M4_LinearRegression_Gender(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Linear(2*self.latent_vector_size+1, self.latent_vector_size+1)
        )

        # male         female
        # 0                 1
        # male -> female: male + gender
        # female -> male: female - gender

        # self.gender_translation = nn.Parameter(torch.randn(self.latent_vector_size, device=self.device))
        # self.register_parameter('gender_translation', self.gender_translation)


    def forward(self, father, mother, child_gender):
        father = father.view(-1, self.latent_vector_size)
        mother = mother.view(-1, self.latent_vector_size)
        child_gender = child_gender.view(-1, 1)
        pred = self.net(torch.cat([father, mother, child_gender], dim=1))

        child_hat = pred[:,:-1].view(-1, *self.latent_vector_shape)
        child_gender_hat = torch.sigmoid(pred[:,-1])

        return child_hat, child_gender_hat

    def _step(self, batch, batch_idx, log_results=False):
        father, mother, child, child_gender, *other = batch

        child_hat, child_gender_hat = self.forward(father, mother, child_gender)

        child_hat_mse = F.mse_loss(child_hat, child)
        child_gender_hat_bce = F.binary_cross_entropy_with_logits(child_gender_hat, child_gender.type_as(child_gender_hat))
        loss = child_hat_mse + child_gender_hat_bce

        self.log('child_hat_mse', child_hat_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('child_gender_hat_bce', child_gender_hat_bce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if log_results and batch_idx == 0:
            self.logger.experiment.log({"results": wandb.Image(self._construct_image_grid(batch))})

        return loss

    def _construct_image_grid(self, batch):
        father, mother, child, child_gender, f_image, m_image, c_image = batch
        batch_size = father.shape[0]
        idxs = np.random.choice(batch_size, 4)

        father = father[idxs, :]
        mother = mother[idxs, :]
        child =  child[idxs, :]
        child_gender =  child_gender[idxs]
        f_image =  np.array(f_image)[idxs]
        m_image =  np.array(m_image)[idxs]
        c_image =  np.array(c_image)[idxs]

        child_gender_inverse = torch.abs(child_gender - 1)

        child_hat, _ = self.forward(father, mother, child_gender)
        child_hat_inverse, _ = self.forward(father, mother, child_gender_inverse)

        toTensor = torchvision.transforms.PILToTensor()
        toPIL = torchvision.transforms.ToPILImage()

        images = []
        for idx in range(4):
            images += [
                toTensor(Image.open(f_image[idx]).convert("RGB").resize((265,265))),
                toTensor(Image.open(m_image[idx]).convert("RGB").resize((265,265))),
                toTensor(Image.open(c_image[idx]).convert("RGB").resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat[idx].cpu().numpy()).resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat_inverse[idx].cpu().numpy()).resize((265,265)))
            ]
        return toPIL(torchvision.utils.make_grid(images, nrow=len(idxs))).convert("RGB")


def main():
    config = {
        "lr": 1e-5,
        "verbose": True,
    }
    utils.core.train(M4_LinearRegression_Gender(**config))


if __name__ == "__main__":
    main()
