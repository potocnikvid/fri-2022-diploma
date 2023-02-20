from dataset.nok_mean import NokMeanDataset
from dataset.nok import NokDataset
from dataset.nok_aug import NokAugDataset
from utils.stylegan import StyleGAN2
from utils.eval import BaseEvaluator
from utils.viz import image_add_label
import os
import sys
import numpy as np
import torch, torchvision
from PIL import Image

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


def load_data(split):
    X = []
    y = []
    for father, mother, child, *_ in NokMeanDataset(split=split):
        input = child.flatten(0)
        output = torch.cat([father.flatten(0), mother.flatten(0)], dim=0)
        X.append(input)
        y.append(output)
    X = torch.stack(X, dim=0)
    y = torch.stack(y, dim=0)
    return X, y


def main(output_dir='default'):
    output_path = "./.tmp/stylegan2-ada-pytorch/" + output_dir
    X_train, y_train = load_data(split="train")
    X_validation, y_validation = load_data(split="validation")
    X_test, y_test = load_data(split="test")
    print(X_train.shape, y_train.shape, X_validation.shape, y_validation.shape, X_test.shape, y_test.shape)

    regressor = Ridge()
    regressor.fit(X_train, y_train)

    y_train_hat = regressor.predict(X_train)
    y_validation_hat = regressor.predict(X_validation)
    y_test_hat = regressor.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_hat)
    mse_validation = mean_squared_error(y_validation, y_validation_hat)
    mse_test = mean_squared_error(y_test, y_test_hat)
    print(mse_train, mse_validation, mse_test)

    images = []
    for i in range(y_test.shape[0]):
        images.append(X_test[i].view(18, 512))
        images.append(y_test[i,:18*512].view(18, 512))
        images.append(y_test[i,18*512:].view(18, 512))
        images.append(torch.from_numpy(y_test_hat[i,:18*512]).to(torch.float32).view(18, 512))
        images.append(torch.from_numpy(y_test_hat[i,18*512:]).to(torch.float32).view(18, 512))
    images = torch.stack(images, dim=0)


    toTensor = torchvision.transforms.PILToTensor()
    toPIL = torchvision.transforms.ToPILImage()
    eval = BaseEvaluator()
    stylegan = StyleGAN2(tmp_path=output_path)

    pils = stylegan.generate_from_array(images.detach().cpu().numpy())
    pil = toPIL(torchvision.utils.make_grid([toTensor(pil.resize((128,128))) for pil in pils], nrow=5)).convert("RGB")
    pil.save(output_path + "/result.png")

    pils_reshaped = np.array([np.array(pil).reshape(1024, 1024, 3) for pil in pils])
    pils_reshaped = pils_reshaped.reshape(-1, 5, 1024, 1024, 3)
    images_eval_f = pils_reshaped[:, 1]
    images_eval_m = pils_reshaped[:, 2]
    images_hat_eval_f = pils_reshaped[:, 3]
    images_hat_eval_m = pils_reshaped[:, 4]

    images_eval_arr_f = [Image.fromarray(i).convert("RGB") for i in images_eval_f]
    images_eval_arr_m = [Image.fromarray(i).convert("RGB") for i in images_eval_m]
    images_hat_eval_arr_f = [Image.fromarray(i).convert("RGB") for i in images_hat_eval_f]
    images_hat_eval_arr_m = [Image.fromarray(i).convert("RGB") for i in images_hat_eval_m]

    images_eval_pil_f = toPIL(torchvision.utils.make_grid([toTensor(pil.resize((128, 128))) for pil in images_eval_arr_f], nrow=1)).convert("RGB")
    images_eval_pil_f.save(output_path + "/images_eval_f.png")

    images_eval_pil_m = toPIL(torchvision.utils.make_grid([toTensor(pil.resize((128, 128))) for pil in images_eval_arr_m], nrow=1)).convert("RGB")
    images_eval_pil_m.save(output_path + "/images_eval_m.png")

    eval_res_f_fn = eval.evaluate_batch(images_eval_f, images_hat_eval_f, model_name='Facenet512')
    images_hat_eval_arr_f_labeled_fn = zip(images_hat_eval_arr_f, eval_res_f_fn)
    images_hat_eval_pil_f_fn = toPIL(torchvision.utils.make_grid([toTensor(image_add_label(pil, str(round(label, 3)), 40).resize((256, 256))) for pil, label in images_hat_eval_arr_f_labeled_fn], nrow=1)).convert("RGB")
    images_hat_eval_pil_f_fn.save(output_path + "/images_eval_hat_f_fn.png")

    eval_res_m_fn = eval.evaluate_batch(images_eval_m, images_hat_eval_m, model_name='Facenet512')
    images_hat_eval_arr_m_labeled_fn = zip(images_hat_eval_arr_m, eval_res_m_fn)
    images_hat_eval_pil_m_fn = toPIL(torchvision.utils.make_grid([toTensor(image_add_label(pil, str(round(label, 3)), 40).resize((256, 256))) for pil, label in images_hat_eval_arr_m_labeled_fn], nrow=1)).convert("RGB")
    images_hat_eval_pil_m_fn.save(output_path + "/images_eval_hat_m_fn.png")

    eval_res_f_af = eval.evaluate_batch(images_eval_f, images_hat_eval_f, model_name='ArcFace')
    images_hat_eval_arr_f_labeled_af = zip(images_hat_eval_arr_f, eval_res_f_af)
    images_hat_eval_pil_f_af = toPIL(torchvision.utils.make_grid([toTensor(image_add_label(pil, str(round(label, 3)), 40).resize((256, 256))) for pil, label in images_hat_eval_arr_f_labeled_af], nrow=1)).convert("RGB")
    images_hat_eval_pil_f_af.save(output_path + "/images_eval_hat_f_af.png")

    eval_res_m_af = eval.evaluate_batch(images_eval_m, images_hat_eval_m, model_name='ArcFace')
    images_hat_eval_arr_m_labeled_af = zip(images_hat_eval_arr_m, eval_res_m_af)
    images_hat_eval_pil_m_af = toPIL(torchvision.utils.make_grid([toTensor(image_add_label(pil, str(round(label, 3)), 40).resize((256, 256))) for pil, label in images_hat_eval_arr_m_labeled_af], nrow=1)).convert("RGB")
    images_hat_eval_pil_m_af.save(output_path + "/images_eval_hat_m_af.png")

    eval_res_vgg = eval.evaluate_batch(images_eval_f, images_hat_eval_f, model_name='VGG-Face')
    images_hat_eval_arr_f_labeled_vgg = zip(images_hat_eval_arr_f, eval_res_vgg)
    images_hat_eval_pil_f_vgg = toPIL(torchvision.utils.make_grid([toTensor(image_add_label(pil, str(round(label, 3)), 40).resize((256, 256))) for pil, label in images_hat_eval_arr_f_labeled_vgg], nrow=1)).convert("RGB")
    images_hat_eval_pil_f_vgg.save(output_path + "/images_eval_hat_f_vgg.png")

    eval_res_vgg = eval.evaluate_batch(images_eval_m, images_hat_eval_m, model_name='VGG-Face')
    images_hat_eval_arr_m_labeled_vgg = zip(images_hat_eval_arr_m, eval_res_vgg)
    images_hat_eval_pil_m_vgg = toPIL(torchvision.utils.make_grid([toTensor(image_add_label(pil, str(round(label, 3)), 40).resize((256, 256))) for pil, label in images_hat_eval_arr_m_labeled_vgg], nrow=1)).convert("RGB")
    images_hat_eval_pil_m_vgg.save(output_path + "/images_eval_hat_m_vgg.png")

if __name__ == "__main__":
    main(sys.argv[1])