import torch
import gradio as gr
from PIL import Image
import scipy.io.wavfile as wavfile

# Use a pipeline as a high-level helper
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# model_path = ("../Models/models--Salesforce--blip-image-captioning-large"
#               "/snapshots/2227ac38c9f16105cb0412e7cab4759978a8fd90")
#
# tts_model_path = ("../Models/models--kakao-enterprise--vits-ljs/snapshots"
#                   "/3bcb8321394f671bd948ebf0d086d694dda95464")

caption_image = pipeline("image-to-text",
                model="Salesforce/blip-image-captioning-large", device=device)

narrator = pipeline("text-to-speech",
                    model="kakao-enterprise/vits-ljs")

# caption_image = pipeline("image-to-text",
#                 model=model_path, device=device)
#
# narrator = pipeline("text-to-speech",
#                     model=tts_model_path)

def generate_audio(text):
    # Generate the narrated text
    narrated_text = narrator(text)

    # Save the audio to a WAV file
    wavfile.write("output.wav", rate=narrated_text["sampling_rate"],
                  data=narrated_text["audio"][0])
    # Return the path to the saved audio file
    return "output.wav"


def caption_my_image(pil_image):
    semantics = caption_image(images=pil_image)[0]['generated_text']
    return generate_audio(semantics)

demo = gr.Interface(fn=caption_my_image,
                    inputs=[gr.Image(label="Select Image",type="pil")],
                    outputs=[gr.Audio(label="Image Caption")],
                    title="@GenAILearniverse Project 8: Image Captioning",
                    description="THIS APPLICATION WILL BE USED TO CAPTION THE IMAGE.")
demo.launch()
