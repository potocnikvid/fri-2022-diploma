import os
from dotenv import load_dotenv
load_dotenv()
import random
from glob import glob

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.download import download_github_asset, get_github_asset_url, unzip
import shutil
from pytorch_lightning.core import LightningDataModule
import torch
import torch.nn.functional as F


class NokAugDataset(Dataset):
    def __init__(
            self,
            root: str = os.getenv('ROOT') + "/src/dataset/nokdb",
            split: str = "train", # train | validation | test
            inflate: int = 4,
        ) -> None:

        self.root = root
        self.split = split
        self.inflate = inflate

        if not os.path.exists(self.root):
            self._download()

        self.samples_df = pd.read_csv(f"{root}/nokdb-samples-{self.split}.csv")
        del self.samples_df["f_iid"]
        del self.samples_df["m_iid"]
        del self.samples_df["c_iid"]
        self.samples_df = self.samples_df.drop_duplicates()
        self.persons_df = pd.read_csv(f"{root}/nokdb-persons.csv")
        self.images_df = pd.read_csv(f"{root}/nokdb-images.csv")

        self.idx = 0
        self.len = len(self.samples_df.index)
        self._load_latent_vectors()

    def __len__(self):
        return self.len * self.inflate

    def __getitem__(self, idx):
        idx = idx % self.len
        ids = self.samples_df.iloc[idx]
        f_pid, father, f_age, f_gender = self._get_entity_data(ids["f_pid"])
        m_pid, mother, m_age, m_gender = self._get_entity_data(ids["m_pid"])
        c_pid, child,  c_age, c_gender = self._get_entity_data(ids["c_pid"])

        return (
            father,
            mother,
            child,
            c_gender,
            f"{f_pid}|{m_pid}|{c_pid}"   # labels
        )

    def _get_entity_data(self, pid: int):
        idx = self.pids_map[pid]
        iids = self.pids_iids_map[pid]
        person_ws = self.data[idx]

        m = random.random()
        i0, i1 = random.choices(range(person_ws.shape[0]), k=2)
        w = m * person_ws[i0] + (1-m) * person_ws[i1]
        
        gender_code = self.persons_df[self.persons_df.pid == pid].sex_code.item()
        age_0 = self.images_df[(self.images_df.pid == pid) & (self.images_df.iid == iids[i0])].age.item()
        age_1 = self.images_df[(self.images_df.pid == pid) & (self.images_df.iid == iids[i1])].age.item()

        age = m * age_0 + (1-m) * age_1

        assert w.shape == (18, 512)
        return pid, w, age, gender_code


    def _load_latent_vectors(self):
        pids = pd.concat([self.samples_df["f_pid"], self.samples_df["m_pid"], self.samples_df["c_pid"]], ignore_index=True).unique()

        self.pids_map = dict()
        self.pids_iids_map = dict()
        self.data = []

        for i, pid in enumerate(pids):
            pid = int(pid)
            self.pids_iids_map[pid] = []

            ws = []
            for npz in glob(f"{self.root}/{pid}/*.npz"):
                iid = int(npz[:-4].split("/")[-1])

                w = np.load(npz)['w']
                if np.isnan(w).any() or np.isinf(w).any():
                    w = np.nan_to_num(w)
                    # raise Exception("Latent vector contain NaN or Inf.")
                w = torch.from_numpy(w).squeeze()

                ws += [w]
                self.pids_iids_map[pid] += [iid]
            
            ws = torch.stack(ws, dim=0)

            self.pids_map[pid] = i
            self.data += [ws]


    def _download(self):
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=False)
        asset_url = get_github_asset_url()
        local_file = f"{self.root}/nokdb.zip"
        download_github_asset(asset_url, local_file)
        unzip(local_file, self.root)
        os.remove(local_file)


class NokAugDataModule(LightningDataModule):

    def __init__(self,
        data_dir: str = os.getenv('ROOT') + "/src/dataset/nokdb",
        batch_size: int =128,
        inflate: int = 4,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.inflate = inflate
        self.num_workers = num_workers

    def prepare_data(self):
        if os.path.exists(self.data_dir): return
        NokAugDataset(root=self.data_dir)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train         = NokAugDataset(root=self.data_dir, split="train",      inflate=self.inflate)
            self.data_validation    = NokAugDataset(root=self.data_dir, split="validation", inflate=self.inflate)
        elif stage == 'test':
            self.data_test          = NokAugDataset(root=self.data_dir, split="test",       inflate=self.inflate)

    def train_dataloader(self):
        return DataLoader(self.data_train,      batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.data_test,       batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)
