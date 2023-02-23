import os
from dotenv import load_dotenv
load_dotenv()
from pprint import pprint
from glob import glob
from github.GitRelease import GitRelease

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.download import download_github_asset, get_github_asset_url, unzip
import shutil
from pytorch_lightning.core.datamodule import LightningDataModule
import torch, torchvision

class NokMeanRealDataset(Dataset):
    def __init__(
            self,
            root: str = os.getenv('ROOT') + "/src/dataset/nokdb",
            download: bool = False,
            split: str = "train", #| "validation" | "test",
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
        del self.samples_df["f_iid"]
        del self.samples_df["m_iid"]
        del self.samples_df["c_iid"]
        del self.samples_df["f_pid"]
        del self.samples_df["m_pid"]
        self.samples_df = self.samples_df.drop_duplicates()
        self.persons_df = pd.read_csv(f"{root}/nokdb-persons.csv")
        
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
            self._get_latent_vector(ids["c_pid"], None),
            self._get_gender(ids["c_pid"]),
            ids["c_pid"],
        )

    def _get_latent_vector(self, pid, normalize):
        idx = self.pid_map[int(pid)]
        return self.normalize(self.data[idx]) if normalize is not None else self.data[idx]

    def _get_gender(self, pid):
        return self.persons_df[self.persons_df.pid == pid].sex_code.item()

    def _load_latent_vectors(self):
        pids = pd.concat([self.samples_df["c_pid"]], ignore_index=True).unique()
        self.pid_map = dict()
        self.data = []
        p_count = 0
        p_img_count = []
        for i, pid in enumerate(pids):
            pid = int(pid)
            ws = []
            for npz in glob(f"{self.root}/{pid}/*.npz"):
                w = np.load(npz)['w']
                if np.isnan(w).any() or np.isinf(w).any():
                    w = np.nan_to_num(w)
                    # raise Exception("Latent vector contain NaN or Inf.")
                ws.append(torch.from_numpy(w))
            p_count += 1
            p_img_count.append(len(ws))
            ws = torch.stack(ws, dim=0)

            self.pid_map[pid] = i
            self.data.append(torch.mean(ws, dim=0))
        
        print(f"Loaded {p_count} persons with {sum(p_img_count)} images.")
        print(f"Average images per person: {sum(p_img_count) / p_count}")
        print(f"Max images per person: {max(p_img_count)}")
        print(f"Min images per person: {min(p_img_count)}")
        print(p_img_count)
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


class NokMeanDataModule(LightningDataModule):

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
        if os.path.exists(self.data_dir): 
            return NokMeanRealDataset(root=self.data_dir, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train         = NokMeanRealDataset(root=self.data_dir, split="train")
            self.data_validation    = NokMeanRealDataset(root=self.data_dir, split="validation")
        elif stage == 'test':
            self.data_test          = NokMeanRealDataset(root=self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.data_train,      batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.data_validation, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.data_test,       batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=False)
