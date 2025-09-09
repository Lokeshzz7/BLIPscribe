from PIL import Image
from transformers import BlipProcessor , BlipForConditionalGeneration
import gradio as gr

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") 