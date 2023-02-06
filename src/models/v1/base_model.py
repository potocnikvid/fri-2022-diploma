import utils.core
import utils.viz
from utils.stylegan import StyleGAN2

from PIL import Image
from pprint import pprint

import numpy as np
import wandb
import torch, torchvision
from torch.nn import functional as F
from torch import nn
from pytorch_lightning import LightningModule


class BaseModel(LightningModule):

    def __init__(self,
        lr: float = 1e-4,
        weight_decay: float = 0.5,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.save_hyperparameters()

        self.stylegan = StyleGAN2(verbose=self.verbose)
        self.latent_vector_size = 1*18*512
        self.latent_vector_shape = (1, 18, 512)
    

    def forward(self, father, mother):
        pass


    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 0)


    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 1)

    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 4)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def _step(self, batch, batch_idx, log_results=0):
        father, mother, child, child_gender, *other = batch

        child_hat = self.forward(father, mother)

        child_hat_mse = F.mse_loss(child_hat, child)
        loss = child_hat_mse

        self.log('child_hat_mse', child_hat_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if log_results and batch_idx == 0:
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

        child_hat = self.forward(father, mother)

        toTensor = torchvision.transforms.PILToTensor()
        toPIL = torchvision.transforms.ToPILImage()

        images = []
        for idx in range(n):
            images += [
                toTensor(Image.open(f_image[idx]).convert("RGB").resize((265,265))),
                toTensor(Image.open(m_image[idx]).convert("RGB").resize((265,265))),
                toTensor(Image.open(c_image[idx]).convert("RGB").resize((265,265))),
                toTensor(self.stylegan.generate_from_array(child_hat[idx].cpu().numpy()).resize((265,265))),
            ]
        return toPIL(torchvision.utils.make_grid(images, nrow=4)).convert("RGB")
