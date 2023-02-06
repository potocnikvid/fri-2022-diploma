import os
from pprint import pprint
from glob import glob
from github.GitRelease import GitRelease

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from utils.download import download_github_asset, get_github_asset_url, unzip
import shutil
from pytorch_lightning.core import LightningDataModule
import torch.nn.functional as F
import torch, torchvision

class WLSDataset(Dataset):
    def __init__(
            self,
            root: str = "dataset/.cache/nokdb",
            split: str = "train", # train | validation | test
            download: bool = False,
            normalize: bool = True,
        ) -> None:

        self.root = root
        self.split = split
        self.download = download

        if not os.path.exists(self.root):
            self._download(False)
        elif self.download:
            self._download(True)

        self.normalize = None
        if normalize:
            npz = np.load(f"{root}/nokdb/nokdb-normalization.npz")
            self.normalize = torchvision.transforms.Normalize(torch.from_numpy(npz["mean"]), torch.from_numpy(npz["std"]))

        self._load_latent_vectors()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.normalize(self.data[idx]) if self.normalize is not None else self.data[idx], *self.metadata[idx])

    def _load_latent_vectors(self):
        persons_df = pd.read_csv(f"{self.root}/nokdb-persons.csv")
        images_df = pd.read_csv(f"{self.root}/nokdb-images.csv")
        samples_df = pd.read_csv(f"{self.root}/nokdb-samples-{self.split}.csv")
        pids = np.unique(np.concatenate((
            samples_df["f_pid"].unique(),
            samples_df["m_pid"].unique(),
            samples_df["c_pid"].unique(),
        )))

        self.metadata = []
        self.data = []

        for pid in pids:
            for npz in glob(f"{self.root}/{pid}/*.npz"):
                iid = int(npz.split("/")[-1][:-4])

                w = np.load(npz)['w']
                if np.isnan(w).any() or np.isinf(w).any():
                    # w = np.nan_to_num(w)
                    raise Exception("Latent vector contain NaN or Inf.")
                
                image   = npz.replace(".npz", ".png")
                gender  = persons_df[(persons_df.pid == pid)].sex_code.item()
                race    = persons_df[(persons_df.pid == pid)].race_code.item()
                age     = images_df[(images_df.pid == pid) & (images_df.iid == iid)].age.item()
                emotion = images_df[(images_df.pid == pid) & (images_df.iid == iid)].emotion_code.item()

                self.metadata.append((image, gender, race, age, emotion))
                self.data.append(torch.from_numpy(w))
        
        self.data = torch.stack(self.data, dim=0)

    def _download(self, force=False):
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=False)
        asset_url = get_github_asset_url()
        local_file = f"{self.root}/nokdb.zip"
        download_github_asset(asset_url, local_file)
        unzip(local_file, self.root)
        os.remove(local_file)

    def _to_one_hot(self, code, num_classes):
        return F.one_hot(torch.tensor(code), num_classes)


class WLSDataModule(LightningDataModule):

    def __init__(self,
        data_dir: str = "dataset/.cache/nokdb",
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if os.path.exists(self.data_dir): return
        WLSDataset(root=self.data_dir, download=True)

    def setup(self, stage=None):
        ds = WLSDataset(root=self.data_dir)
        p80 = int(len(ds) * 0.8)
        p10_1 = int((len(ds) - p80) * 0.5)
        p10_2 = len(ds) - p80 - p10_1
        train, validation, test = random_split(ds, [p80, p10_1, p10_2])

        if stage == 'fit' or stage is None:
            self.data_train         = WLSDataset(root=self.data_dir, split="train")
            self.data_validation    = WLSDataset(root=self.data_dir, split="validation")
        elif stage == 'test':
            self.data_test          = WLSDataset(root=self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.data_train,      batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.data_test,       batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
