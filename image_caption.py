import requests
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_creator(img_path, img_hint_text):
#img_path = img_path#"/Users/akashd/PycharmProjects/ImageCaptioning/sample_images/wp3214348-cow-wallpapers.jpg"
    img_RGB = Image.fromarray(img_path).convert('RGB')
    #img_RGB = Image.open(img_path).convert("RGB")

#img_hint_text = "cow"
    input_img = processor(images = img_RGB, text = img_hint_text, return_tensors="pt")
#readable_input = processor.decode(input_img, skip_special_tokens=True)
#print(f"Readable Input {input_img}")
#**inputs is unpacking the inputs dictionary and passing its items as arguments to the model.
    output = model.generate(**input_img, max_length = 100)
    readable_caption: str = processor.decode(output[0], skip_special_tokens=True)
    return readable_caption
    #print(readable_caption)
'''
the generated output is a sequence of tokens. 
To transform these tokens into human-readable text, you use the decode method provided by the processor. 
The skip_special_tokens argument is set to True to ignore special tokens in the output text.
'''

app = gr.Interface(
    fn = caption_creator,
    inputs = [gr.Image(),"text"],
    outputs = "text",
    title = "Image Caption",
    description="This is a simple web app for generating captions for images using a trained model."
)

app.launch()