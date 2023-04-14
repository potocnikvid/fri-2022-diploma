import os
import os.path
import io
import IPython.display
import numpy as np
import cv2
import PIL.Image

import torch

from submodules.interfacegan.models.model_settings import MODEL_POOL
from submodules.interfacegan.models.pggan_generator import PGGANGenerator
from submodules.interfacegan.models.stylegan_generator import StyleGANGenerator
from submodules.interfacegan.utils.manipulator import linear_interpolate
from dotenv import load_dotenv
load_dotenv()
root = os.getenv("ROOT")

def build_generator(model_name):
    """Builds the generator by model name."""
    gan_type = MODEL_POOL[model_name]['gan_type']
    if gan_type == 'pggan':
        generator = PGGANGenerator(model_name)
    elif gan_type == 'stylegan':
        generator = StyleGANGenerator(model_name)
    return generator


def sample_codes(generator, num, latent_space_type='Z', seed=0):
    """Samples latent codes randomly."""
    np.random.seed(seed)
    codes = generator.easy_sample(num)
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
        codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
        codes = generator.get_value(generator.model.mapping(codes))
    return codes


def imshow(images, col, viz_size=256):
    """Shows images in one figure."""
    num, height, width, channels = images.shape
    assert num % col == 0
    row = num // col

    fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

    for idx, image in enumerate(images):
        i, j = divmod(idx, col)
        y = i * viz_size
        x = j * viz_size
        if height != viz_size or width != viz_size:
            image = cv2.resize(image, (viz_size, viz_size))
        fused_image[y:y + viz_size, x:x + viz_size] = image

    fused_image = np.asarray(fused_image, dtype=np.uint8)
    data = io.BytesIO()
    PIL.Image.fromarray(fused_image).save(data, 'jpeg')
    im_data = data.getvalue()
    disp = IPython.display.display(IPython.display.Image(im_data))
    return disp




def main():
    #@title { display-mode: "form", run: "auto" }
    submodule_path = f'{root}/submodules/interfacegan/'
    model_name = "stylegan_ffhq" #@param ['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq']
    latent_space_type = "W" #@param ['Z', 'W']

    generator = build_generator(model_name)

    ATTRS = ['age', 'gender', 'eyeglasses']
    boundaries = {}
    for i, attr_name in enumerate(ATTRS):
        boundary_name = f'{model_name}_{attr_name}'
        if generator.gan_type == 'stylegan' and latent_space_type == 'W':
            boundaries[attr_name] = np.load(f'{submodule_path}boundaries/{boundary_name}_w_boundary.npy')
        else:
            boundaries[attr_name] = np.load(f'{submodule_path}boundaries/{boundary_name}_boundary.npy')


    #@title { display-mode: "form", run: "auto" }

    num_samples = 1 #@param {type:"slider", min:1, max:8, step:1}
    noise_seed = 0 #@param {type:"slider", min:0, max:1000, step:1}

    latent_codes = sample_codes(generator, num_samples, latent_space_type, noise_seed)
    print(latent_codes)
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
        synthesis_kwargs = {'latent_space_type': 'W'}
    else:
        synthesis_kwargs = {}

    images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']
    imshow(images, col=num_samples)
    

if __name__ == "__main__":
    main()