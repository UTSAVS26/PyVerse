from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("Hey Guest! I am Belito, a passionate Engineer with a degree in Information Technology. I am currently working as a Cloud Enginee I am a programmer and more into web development but yeah, can't miss the problem solving. I got selected as Google Summer of Code'22 under Scorelab organization. I also got into LFX'22 Summer mentorship program under Moja Global. I have completed my tenure as an SE Intern at Digital Product School, Germany. prior to that, I worked as Software Engineer Intern at a startup named Summachar.", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
