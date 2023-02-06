import torch
import numpy as np

import os
from glob import glob
from pprint import pprint


def main():  
    dataset_root = "./dataset/nokdb"
    wsa = []
    within_means = []

    for pid_root in glob(f"{dataset_root}/*/"):
        # pid = int(pid_root.split("/")[-2])
        # print(pid)
        print(pid_root)
        ws = []
        for npz in glob(f"{pid_root}/*.npz"):
            w = np.load(npz)["w"]
            if np.isnan(w).any() or np.isinf(w).any():
                w = np.nan_to_num(w)
            w = torch.from_numpy(w).squeeze().flatten(0)
            ws += [w]
            wsa += [w]
        ws = torch.stack(ws)

        within_mean = torch.mean(ws, dim=0)
        within_variation = torch.mean((ws - within_mean) ** 2)
        # pprint(within_variation)

        within_means += [within_mean]

    overall_mean = torch.mean(torch.stack(wsa), dim=0)
    within_means = torch.stack(within_means)

    between_variation = torch.mean((within_means - overall_mean) ** 2)

    pprint((between_variation, overall_mean.shape, within_means.shape))
    

if __name__ == "__main__":
    main()
  