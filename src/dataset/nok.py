import os
from dotenv import load_dotenv
load_dotenv()
from pprint import pprint
from glob import glob

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.download import download_github_asset, get_github_asset_url, unzip
import shutil
from pytorch_lightning.core import LightningDataModule
import torch, torchvision
import torch.nn.functional as F

class NokDataset(Dataset):
    def __init__(
            self,
            root: str = os.getenv('ROOT') + "/src/dataset/nokdb",
            download: bool = False,
            split: str = "train", # train | validation | test
            normalize: bool = False,
        ) -> None:

        self.root = root
        self.split = split
        self.download = download

        if not os.path.exists(self.root):
            self._download(False)
        elif self.download:
            self._download(True)

        self.samples_df = pd.read_csv(f"{root}/nokdb-samples-{self.split}.csv")
        self.persons_df = pd.read_csv(f"{root}/nokdb-persons.csv")
        self.images_df = pd.read_csv(f"{root}/nokdb-images.csv")
        
        self.normalize = None
        if normalize:
            npz = np.load(f"{root}/norm/nokdb-normalization.npz")
            self.normalize = torchvision.transforms.Normalize(torch.from_numpy(npz["mean"]), torch.from_numpy(npz["std"]))

        self._load_latent_vectors()

    def __len__(self):
        return len(self.samples_df.index)

    def __getitem__(self, idx):
        ids = self.samples_df.iloc[idx]
        return (
            self._get_latent_vector(ids["f_pid"], ids["f_iid"], self.normalize),
            self._get_latent_vector(ids["m_pid"], ids["m_iid"], self.normalize),
            self._get_latent_vector(ids["c_pid"], ids["c_iid"], None),
            self._get_labels(ids["f_pid"], ids["f_iid"]),
            self._get_labels(ids["m_pid"], ids["m_iid"]),
            self._get_labels(ids["c_pid"], ids["c_iid"]),
            self._get_image(ids["f_pid"], ids["f_iid"]),
            self._get_image(ids["m_pid"], ids["m_iid"]),
            self._get_image(ids["c_pid"], ids["c_iid"]),
        )

    def _get_latent_vector(self, pid, iid, normalize):
        idx = self.piids_map[f"{pid}-{iid}"]
        return self.normalize(self.data[idx]) if normalize is not None else self.data[idx]

    def _get_labels(self, pid, iid):
        p_data = self.persons_df[self.persons_df.pid == pid]
        i_data = self.images_df[(self.images_df.pid == pid) & (self.images_df.iid == iid)]

        # total 2: M, F
        gender_code = p_data.sex_code.item()
        # total 4: 20-29, 30-39, 40-49, 50-59
        age_code = max(min(int((i_data.age.item() / 10) - 2), 3), 0)

        gender_one_hot = F.one_hot(torch.tensor(gender_code), 2)
        age_one_hot = F.one_hot(torch.tensor(age_code), 4)

        labels_one_hot = torch.cat([gender_one_hot, age_one_hot], dim=0)
        return labels_one_hot

    def _get_image(self, pid, iid):
        return f"{self.root}/{pid}/{iid}.png"

    def _load_latent_vectors(self):
        pids = pd.concat([self.samples_df["f_pid"], self.samples_df["m_pid"], self.samples_df["c_pid"]], ignore_index=True).unique()
        self.piids_map = dict()
        self.data = []
        print(pids)
        i = 0
        for pid in pids:
            pid = int(pid)
            for npz in glob(f"{self.root}/{pid}/*.npz"):
                iid = int(npz[:-4].split("/")[-1])
                w = np.load(npz)['w']
                if np.isnan(w).any() or np.isinf(w).any():
                    w = np.nan_to_num(w)
                    # raise Exception("Latent vector contain NaN or Inf.")
                self.piids_map[f"{pid}-{iid}"] = i
                self.data.append(torch.from_numpy(w))
                i += 1
        print(i)
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


class NokDataModule(LightningDataModule):

    def __init__(self,
        data_dir: str = os.getenv('ROOT') + "/src/dataset/nokdb",
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if os.path.exists(self.data_dir): return
        NokDataset(root=self.data_dir, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train         = NokDataset(root=self.data_dir, split="train")
            self.data_validation    = NokDataset(root=self.data_dir, split="validation")
        elif stage == 'test':
            self.data_test          = NokDataset(root=self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.data_train,      batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.data_test,       batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)
