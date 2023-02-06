from argparse import ArgumentError
from torch.nn.modules.activation import ReLU, SELU
from torch.nn.modules.container import Sequential
from models.v1.base_model import BaseModel
from models.wls.autoencoder import AutoEncoder
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
from pytorch_lightning import LightningModule

from utils.stylegan import StyleGAN2


class M11_x18(LightningModule):

    def __init__(self,
        lr=1e-4,
        weight_decay=0.0,
        dropout_p=0.0,
        verbose=False,
        **kwargs
    ):
        super().__init__()  
        self.save_hyperparameters()

        self.example_input_array = (torch.rand(1,1,18,512), torch.rand(1,1,18,512))

        self.stylegan = StyleGAN2(self.hparams.verbose)
        
        self.bn = nn.ModuleList([
            nn.BatchNorm1d(1024)
            for _ in range(18)
        ])

        self.ns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 768),
                # nn.ReLU(),
                nn.Linear(768, 512),
            )
            for _ in range(18)
        ])

    def forward(self, father, mother):
        batch_size = father.shape[0]
        child_hat = []
        for i, n in enumerate(self.ns):
            father_slice = father[:,:,i].flatten(1)
            mather_slice = mother[:,:,i].flatten(1)
            input_slice = torch.cat([father_slice, mather_slice], dim=1)
            input_bn = self.bn[i](input_slice) if batch_size > 1 or not self.training else input_slice
            slice_hat = n(input_bn)
            child_hat += [slice_hat]
        
        child_hat =  torch.stack(child_hat, dim=1).unsqueeze(1)
        return child_hat

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 1, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 1, "validation")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 6, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _step(self, batch, batch_idx, log_results=0, label=None):
        father, mother, child, child_gender, *other = batch

        child_hat = self.forward(father, mother)

        child_hat_mse = F.mse_loss(child_hat, child)
        loss = child_hat_mse

        self.log('child_hat_mse', child_hat_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if log_results > 0 and batch_idx == 0:
            self.logger.experiment.log({f"results/{label}": wandb.Image(self._construct_image_grid(batch, n=log_results))})

        return loss

    def _construct_image_grid(self, batch, n=4):
        # father, mother, child, child_gender, f_image, m_image, c_image = batch
        father, mother, child, child_gender, f_pid, m_pid, c_pid = batch
        batch_size = father.shape[0]
        # idxs = np.random.choice(batch_size, n)
        idxs = list(range(n))

        father = father[idxs, :]
        mother = mother[idxs, :]
        child =  child[idxs, :]
        child_gender =  child_gender[idxs]
        # f_image =  np.array(f_image)[idxs]
        # m_image =  np.array(m_image)[idxs]
        # c_image =  np.array(c_image)[idxs]
        f_pid = f_pid[idxs]
        m_pid = m_pid[idxs]
        c_pid = c_pid[idxs]

        child_hat = self.forward(father, mother)

        toTensor = torchvision.transforms.PILToTensor()
        toPIL = torchvision.transforms.ToPILImage()

        get_label = lambda x: "|".join(x.split("/")[-2:])[:-4]
        
        label = " | ".join([f"({f_pid[i]}, {m_pid[i]}, {c_pid[i]})" for i in range(n)])

        images = []
        for idx in range(n):
            # images += [
            #     toTensor(utils.viz.image_add_label(Image.open(f_image[idx]), get_label(f_image[idx]), 50).convert("RGB").resize((265,265))),
            #     toTensor(utils.viz.image_add_label(Image.open(m_image[idx]), get_label(m_image[idx]), 50).convert("RGB").resize((265,265))),
            #     toTensor(utils.viz.image_add_label(Image.open(c_image[idx]), get_label(c_image[idx]), 50).convert("RGB").resize((265,265))),
            #     toTensor(self.stylegan.generate_from_array(child_hat[idx].detach().cpu().numpy())[0].resize((265,265))),
            # ]
            ws = torch.stack([
                father[idx],
                mother[idx],
                child[idx],
                child_hat[idx],
            ]).view(-1, 18, 512).detach().cpu().numpy()
            images += [toTensor(x.resize((256,256))) for x in self.stylegan.generate_from_array(ws)]
        return utils.viz.image_add_label(toPIL(torchvision.utils.make_grid(images, nrow=int(len(images)/n))).convert("RGB"), label, 50)

def main():
    config = {
        "lr": 1e-3, #0.0005
        "weight_decay": 1e-9,
        "dropout_p": 1e-1,
        "verbose": False,
        "batch_size": 16,
    }
    utils.core.train(M11_x18, config=config, early_stopping_delta=0.00001, max_epochs=100, test=True)


if __name__ == "__main__":
    main()