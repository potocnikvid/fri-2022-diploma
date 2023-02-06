import os
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

sys.path.append(".")
sys.path.append("..")

from submodules.hyperstyle.notebooks.notebook_utils import run_alignment
from submodules.hyperstyle.utils.common import tensor2im
from submodules.hyperstyle.utils.inference_utils import run_inversion
# from submodules.hyperstyle.utils.domain_adaptation_utils import run_domain_adaptation
from submodules.hyperstyle.utils.model_utils import load_model, load_generator


def get_coupled_results(result_batch, transformed_image):
    result_tensors = result_batch[0]  # there's one image in our batch
    final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)
    input_im = tensor2im(transformed_image).resize(resize_amount)
    res = np.concatenate([np.array(input_im), np.array(final_rec)], axis=1)
    res = Image.fromarray(res)
    return res



#@title Select which domain you wish to perform inference on: { display-mode: "form" }
experiment_type = 'faces' #@param ['faces', 'cars', 'afhq_wild']

EXPERIMENT_DATA_ARGS = {
    "faces": {
        "model_path": "./pretrained_models/hyperstyle_ffhq.pt",
        "w_encoder_path": "./pretrained_models/faces_w_encoder.pt",
        "image_path": "./notebooks/images/face_image.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "cars": {
        "model_path": "./pretrained_models/hyperstyle_cars.pt",
        "w_encoder_path": "./pretrained_models/cars_w_encoder.pt",
        "image_path": "./notebooks/images/car_image.jpg",
        "transform": transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "afhq_wild": {
        "model_path": "./pretrained_models/hyperstyle_afhq_wild.pt",
        "w_encoder_path": "./pretrained_models/afhq_wild_w_encoder.pt",
        "image_path": "./notebooks/images/afhq_wild_image.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]



#@title Load HyperStyle Model { display-mode: "form" } 
model_path = EXPERIMENT_ARGS['model_path']
net, opts = load_model(model_path, update_opts={"w_encoder_checkpoint_path": EXPERIMENT_ARGS['w_encoder_path']})
print('Model successfully loaded!')
pprint.pprint(vars(opts))

image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
original_image = Image.open(image_path).convert("RGB")
if experiment_type == 'cars':
    original_image = original_image.resize((192, 256))
else:
    original_image = original_image.resize((256, 256))
original_image


#@title Align Image (If Needed)
input_is_aligned = False #@param {type:"boolean"}
if experiment_type == "faces" and not input_is_aligned:
    input_image = run_alignment(image_path)
else:
    input_image = original_image

input_image.resize((256, 256))



#@title { display-mode: "form" } 
n_iters_per_batch = 5 #@param {type:"integer"}
opts.n_iters_per_batch = n_iters_per_batch
opts.resize_outputs = False  # generate outputs at full resolution


#@title Run Inference! { display-mode: "form" }
img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(input_image) 

with torch.no_grad():
    tic = time.time()
    result_batch, result_latents, _ = run_inversion(transformed_image.unsqueeze(0).cuda(), 
                                                    net, 
                                                    opts,
                                                    return_intermediate_results=True)
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))


if opts.dataset_type == "cars":
    resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
else:
    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

res = get_coupled_results(result_batch, transformed_image)
res


# save image 
outputs_path = "./outputs"
os.makedirs(outputs_path, exist_ok=True)
res.save(os.path.join(outputs_path, os.path.basename(image_path)))