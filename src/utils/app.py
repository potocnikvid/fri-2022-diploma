import gradio as gr
import numpy as np


def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    image = gr.inputs.File(file_count="multiple")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=sepia, inputs=name, outputs=output, api_name="greet")


demo.launch(share=True)