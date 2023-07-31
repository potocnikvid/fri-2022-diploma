import os
import numpy as np
# from onnx import ModelProto
# from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
# from graphviz import render
from PIL import Image, ImageFont, ImageDraw
from matplotlib.font_manager import findfont, FontProperties
import matplotlib.pyplot as plt

# def plot_onnx(onnx_file: str) -> str:
#     dot_file = f"{onnx_file[:-5]}.dot"
#     model = ModelProto()
#     with open(onnx_file, 'rb') as fid:
#         content = fid.read()
#         model.ParseFromString(content)
#     pydot_graph = GetPydotGraph(
#         model.graph,
#         name=model.graph.name,
#         rankdir="LR",
#         node_producer=GetOpNodeProducer(
#             embed_docstring=True,
#         ),
#     )
#     pydot_graph.write_dot(dot_file)
#     viz_file = render("dot", "pdf", dot_file)
#     os.remove(dot_file)
#     return viz_file


def image_add_label(image: Image, label: str, font_size=10) -> Image:
    draw = ImageDraw.Draw(image)
    pyplot_font_file = findfont(FontProperties(family=['sans-serif']))
    font = ImageFont.truetype(pyplot_font_file, size=font_size)
    draw.text((0,0), str(label), (255, 255, 0), font=font)
    return image


def histogram(x, title, x_label, y_label, path, range=(-1, 1), save_name='histogram', save=True, num_bins = 20):
    n, bins, patches = plt.hist(x, num_bins, range=range)
    plt.title(title, fontsize=20)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    if save:
        plt.savefig('{}/{}.png'.format(path, save_name))
    plt.show()
    return n, bins, patches