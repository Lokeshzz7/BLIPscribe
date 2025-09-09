from PIL import Image
from transformers import BlipProcessor , BlipForConditionalGeneration
import gradio as gr

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") 

def caption_generation(img):
    image = Image.fromarray(img)
    inputs = processor(image , return_tensors = "pt")
    outputs = model.generate(**inputs)
    cap = processor.decode(outputs[0] , skip_special_tokens = True)
    return cap

web = gr.Interface(fn = caption_generation,inputs = [gr.Image(label = "Image")],outputs = [gr.Text(label = "Caption"),],)

web.launch()