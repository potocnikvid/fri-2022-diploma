import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from glob import glob
from pprint import pprint



def main():
    dataset_root = "./dataset/nokdb"
    persons_df = pd.read_csv(f"{dataset_root}/nokdb-persons.csv")
    images_df = pd.read_csv(f"{dataset_root}/nokdb-images.csv")

    mean_age = images_df["age"].mean()

    X = []
    y_age = []
    y_sex = []

    for npz in glob(f"{dataset_root}/**/*.npz", recursive=True):
        # print(npz)
        if npz[:-4].split("/")[3] == 'norm':
            continue
        pid, iid = list(map(int, npz[:-4].split("/")[-2:]))
        w = np.load(npz)['w'].flatten()
        if np.isnan(w).any() or np.isinf(w).any():
            w = np.nan_to_num(w)
            # raise Exception("Latent vector contain NaN or Inf.")
        age = images_df[(images_df.pid == pid) & (images_df.iid == iid)].age.item()
        sex = persons_df[(persons_df.pid == pid)].sex.item()
        age = 0 if age <= mean_age else 1
        sex = 0 if sex == "M" else 1
        X.append(w)
        y_age.append(age)
        y_sex.append(sex)

    X = np.stack(X)
    y_age = np.array(y_age)
    y_sex = np.array(y_sex)

    print(X.shape)

    idx = np.arange(3690)
    np.random.shuffle(idx)

    split_i = int(3690*0.8)

    train_idx = idx[:split_i] # 80%
    test_idx = idx[split_i:]  # 20%

    svm_age = LinearSVC(max_iter=20000)
    svm_age.fit(X[train_idx,:],y_age[train_idx])

    train_age_acc = svm_age.score(X[train_idx,:],y_age[train_idx])
    test_age_acc = svm_age.score(X[test_idx,:],y_age[test_idx])

    print("age acc", train_age_acc, test_age_acc)
    
    
    svm_sex = LinearSVC(max_iter=10000)
    svm_sex.fit(X[train_idx,:],y_sex[train_idx])

    train_sex_acc = svm_sex.score(X[train_idx,:],y_sex[train_idx])
    test_sex_acc = svm_sex.score(X[test_idx,:],y_sex[test_idx])

    print("sex acc", train_sex_acc, test_sex_acc)

if __name__ == "__main__":
    main()