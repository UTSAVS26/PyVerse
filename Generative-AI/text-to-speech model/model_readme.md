This is a tet to speech model which uses SpeechT5 model,when you run the code a speech.wav file will be created in the folder and the generated speech will be present in it

SpeechT5 (TTS task) SpeechT5 model fine-tuned for speech synthesis (text-to-speech) on LibriTTS.

This model was introduced in SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.

SpeechT5 was first released in this repository, original weights. The license used is MIT.

# Model Description
Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.

Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder.

Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.

Developed by: Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei. Shared by: Matthijs Hollemans Model type: text-to-speech Language(s) (NLP): [More Information Needed] License: MIT Finetuned from model [optional]: [More Information Needed] Model Sources [optional] Repository: [https://github.com/microsoft/SpeechT5/] Paper: [https://arxiv.org/pdf/2110.07205.pdf] Blog Post: [https://huggingface.co/blog/speecht5] Demo: [https://huggingface.co/spaces/Matthijs/speecht5-tts-demo] Uses 🤗 Transformers Usage You can run SpeechT5 TTS locally with the 🤗 Transformers library.

First install the 🤗 Transformers library, sentencepiece, soundfile and datasets(optional):

```python
pip install --upgrade pip
pip install --upgrade transformers sentencepiece datasets[audio]
```
Run inference via the Text-to-Speech (TTS) pipeline. You can access the SpeechT5 model via the TTS pipeline in just a few lines of code!

```python
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
```
Run inference via the Transformers modelling code - You can use the processor + generate code to convert text into a mono 16 kHz speech waveform for more python fine-grained control. 
```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan 
from datasets import load_dataset 
import torch 
import soundfile as sf from datasets 
import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts") model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts") vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")

load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation") speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)
```