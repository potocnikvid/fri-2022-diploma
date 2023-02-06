# import os
# from onnx import ModelProto
# from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer
# from graphviz import render
from PIL import Image, ImageFont, ImageDraw
from matplotlib.font_manager import findfont, FontProperties


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
