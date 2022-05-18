import os
import tensorflow as tf
import numpy as np
import PIL.Image
import time
import functools
import gradio as gr
import tensorflow_hub as hub


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def convert_img(img):
  max_dim = 512
  img = tf.image.convert_image_dtype(img, tf.float32)
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def style_transfer(content, style):
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    content = convert_img(content)
    style = convert_img(style)
    stylized_image = hub_model(tf.constant(content), tf.constant(style))[0]
    return tensor_to_image(stylized_image)

content = gr.inputs.Image(shape=None, image_mode="RGB", invert_colors=False, source="upload", tool="editor", type="numpy", label="content", optional=False)
style = gr.inputs.Image(shape=None, image_mode="RGB", invert_colors=False, source="upload", tool="editor", type="numpy", label="Style", optional=False)

ui = gr.Interface(
    style_transfer, title = "artsy",
    description = "The Style Transfer Demo   |   ***Team Aesthetes***   |   Project Sprint TTC",
    article = "Converts your images into artistic paintings. "
    + "Upload your image in content and an artistic image in style"
    + ". Made by Bharath Raj and Raghav Dabral for Project Sprint by TTC",
    theme = "peach",
    inputs=[content, style], outputs=["image"],
    live=False)
ui.launch(debug=True, enable_queue=True)
