from dataset.nok_mean import NokMeanDataset
from utils.stylegan import StyleGAN2

from pprint import pprint


stylegan = StyleGAN2()

images = stylegan._generate_from_file('./.tmp/stylegan2-ada-pytorch/projected_w.npz', 80)