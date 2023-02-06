import os
from dotenv import load_dotenv
load_dotenv()
import random
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


class NokAugParDataset(Dataset):
    def __init__(
            self,
            root: str = os.getenv('ROOT') + "/src/dataset/nokdb",
            split: str = "train", # train | validation | test
        ) -> None:

        self.root = root
        self.split = split

        if not os.path.exists(self.root):
            self._download()

        self.samples_df = pd.read_csv(f"{root}/nokdb-samples-{self.split}.csv")
        # del self.samples_df["f_iid"]
        # del self.samples_df["m_iid"]
        # del self.samples_df["c_iid"]
        # self.samples_df = self.samples_df.drop_duplicates()
        # pprint(self.samples_df.head())
        self.samples_df = self.samples_df.groupby(["f_pid", "m_pid"])["c_pid"].apply(set).apply(list).reset_index(name="c_pids")
        pprint(self.samples_df.head())
        self.persons_df = pd.read_csv(f"{root}/nokdb-persons.csv")
        self.images_df = pd.read_csv(f"{root}/nokdb-images.csv")

        self._load_latent_vectors()


    def __len__(self):
        return len(self.samples_df.index)

    def __getitem__(self, idx):
        ids = self.samples_df.iloc[idx]
        f_pid, father, father_label, f_w0, f_w1, f_m = self._get_parent_entity_data(ids["f_pid"])
        m_pid, mother, mother_label, m_w0, m_w1, m_m = self._get_parent_entity_data(ids["m_pid"])
        c_pid, child,  child_label,  c_w0, c_w1, c_m = self._get_child_entity_data(ids["c_pids"])

        return (
            father,
            mother,
            child,
            father_label,
            mother_label,
            child_label,
            f_pid,
            m_pid,
            c_pid,
            f_w0,
            m_w0,
            c_w0,
            f_w1,
            m_w1,
            c_w1,
            f_m,
            m_m,
            c_m
        )

    def _get_parent_entity_data(self, pid: int):
        idx = self.pids_map[pid]
        iids = self.pids_iids_map[pid]
        person_ws = self.data[idx]

        # morph two random latent vectors for the same entity
        # m = max(0.0, min(1.0, random.gauss(0.5, 0.5/3)))
        m = random.random()
        i0, i1 = random.choices(range(person_ws.shape[0]), k=2)
        w = m * person_ws[i0] + (1-m) * person_ws[i1]
        
        gender_code = self.persons_df[self.persons_df.pid == pid].sex_code.item()
        age_0 = self.images_df[(self.images_df.pid == pid) & (self.images_df.iid == iids[i0])].age.item()
        age_1 = self.images_df[(self.images_df.pid == pid) & (self.images_df.iid == iids[i1])].age.item()

        age = m * age_0 + (1-m) * age_1

        lables = self._get_labels_one_hot(gender_code, age)

        assert w.shape == (18, 512)
        return pid, w, lables, person_ws[i0], person_ws[i1], m

    def _get_child_entity_data(self, pids):
        idx = self.pids_map[pid]
        iids = self.pids_iids_map[pid]
        person_ws = self.data[idx]

        # morph two random latent vectors for the same entity
        # m = max(0.0, min(1.0, random.gauss(0.5, 0.5/3)))
        m = random.random()
        i0, i1 = random.choices(range(person_ws.shape[0]), k=2)
        w = m * person_ws[i0] + (1-m) * person_ws[i1]
        
        gender_code = self.persons_df[self.persons_df.pid == pid].sex_code.item()
        age_0 = self.images_df[(self.images_df.pid == pid) & (self.images_df.iid == iids[i0])].age.item()
        age_1 = self.images_df[(self.images_df.pid == pid) & (self.images_df.iid == iids[i1])].age.item()

        age = m * age_0 + (1-m) * age_1

        lables = self._get_labels_one_hot(gender_code, age)

        assert w.shape == (18, 512)
        return pid, w, lables, person_ws[i0], person_ws[i1], m

    def _get_age_one_hot(self, age):
        # total 4: 20-29, 30-39, 40-49, 50-59
        age_code = max(min(int((age / 10) - 2), 3), 0)
        age_one_hot = F.one_hot(torch.tensor(age_code), 4)
        return age_one_hot

    def _get_gender_one_hot(self, gender_code):
        # total 2: M, F
        gender_one_hot = F.one_hot(torch.tensor(gender_code), 2)
        return gender_one_hot

    def _get_labels_one_hot(self, gender_code, age):
        gender_one_hot = self._get_gender_one_hot(gender_code)
        age_one_hot = self._get_age_one_hot(age)

        labels_one_hot = torch.cat([gender_one_hot, age_one_hot], dim=0)
        return labels_one_hot

    def _load_latent_vectors(self):
        pids = pd.concat([self.samples_df["f_pid"], self.samples_df["m_pid"], self.samples_df["c_pids"].explode()], ignore_index=True).unique()

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
                w = torch.from_numpy(w).squeeze()

                ws += [w]
                self.pids_iids_map[pid] += [iid]
            
            ws = torch.stack(ws, dim=0)

            self.pids_map[pid] = i
            self.data += [ws]

        # self.data = torch.stack(self.data, dim=0)

    def _download(self):
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=False)
        asset_url = get_github_asset_url()
        local_file = f"{self.root}/nokdb.zip"
        download_github_asset(asset_url, local_file)
        unzip(local_file, self.root)
        os.remove(local_file)


class NokAugParDataModule(LightningDataModule):

    def __init__(self,
        data_dir: str = ".cache/nokdb",
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if os.path.exists(self.data_dir): return
        NokAugParDataset(root=self.data_dir)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train         = NokAugParDataset(root=self.data_dir, split="train")
            self.data_validation    = NokAugParDataset(root=self.data_dir, split="validation")
        elif stage == 'test':
            self.data_test          = NokAugParDataset(root=self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.data_train,      batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.data_test,       batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)


if __name__ == "__main__":
    ds = NokAugParDataset()
