from dataset.nok_mean import NokMeanDataset
from utils.stylegan import StyleGAN2

import numpy as np
import torch, torchvision

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from pprint import pprint


def load_data(split="train"):
    X = []
    y = []
    i=0
    for father, mother, child, *_ in NokMeanDataset(split=split):
        #print(split)
        #print(father)
        #print(len(father), len(father[0]), len(father[0][0]))
        #print(father.flatten(0))
        #np.savez('/home/nejc/fri-2022-diploma-master/src/testfather.npz', w=father)
        #exit()
        #print(len(father.flatten(0)))
        #print("asd")
        #print(mother.flatten(0))
        #print(len(mother.flatten(0)))
        #print("---------------------------------------------------------------")
        input = torch.cat([father.flatten(0), mother.flatten(0)], dim=0)
        output = child.flatten(0)
        #print(input)
        #print(len(input))
        #print("asd")
        #print(output)
        #print(len(output))
        X.append(input)
        y.append(output)
        #print(X)
        #print(len(X))
       # print("--------------------------------------------------------------")
       # print(y)
        #print(len(y))
        i=i+1
        
    #print(i)
    X = torch.stack(X, dim=0)
    y = torch.stack(y, dim=0)
    #print(X)
    #print(len(X[0]))
    #print(len(X))
    #print("asdsdaaaaaaaaaaaaaaaaaaaaaaads")
    #print(y)
    #print(len(y[0]))
    #print("asddddddddddaaaaaasdaaaaaaaaaa")
    #exit()
    return X, y


def main():
    X_train, y_train = load_data(split="train")
    X_test, y_test = load_data(split="test")
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    #print(X_train)
    #print(len(X_train))
    #print(y_train)
    #print(len(y_train))
    #exit()
    regressor = Ridge(normalize=True)
    regressor.fit(X_train, y_train)

    y_train_hat = regressor.predict(X_train)
    y_test_hat = regressor.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_hat)
    mse_test = mean_squared_error(y_test, y_test_hat)
    print(mse_train, mse_test)

    images = []
    for i in range(y_test.shape[0]):
        #print(X_test)
        #print(y_test)
        #print(X_test[i,:18*512])
        #print(y_test[i])
        #print(len(X_test[i,:18*512]))
        #print(len(y_test[i]))
        
        images.append(X_test[i,:18*512].view(18, 512))
        images.append(X_test[i,18*512:].view(18, 512))
        images.append(y_test[i].view(18, 512))
        images.append(torch.from_numpy(y_test_hat[i]).to(torch.float32).view(18, 512))
        #exit()

    images = torch.stack(images, dim=0)
    
    stylegan = StyleGAN2()
    
    
    
    pils = stylegan.generate_from_array(images.detach().cpu().numpy())
    exit()
    toTensor = torchvision.transforms.PILToTensor()
    toPIL = torchvision.transforms.ToPILImage()

    pil = toPIL(torchvision.utils.make_grid([toTensor(pil.resize((128,128))) for pil in pils], nrow=4)).convert("RGB")
    print(pil)

if __name__ == "__main__":
    main()