# Rick and Morty ChatBot

A conversational AI model based on Microsoft's DialoGPT-small, fine-tuned on Rick and Morty dialogue to create a chatbot that speaks in the style of the show's characters.

## Overview

This project uses the Hugging Face Transformers library to fine-tune Microsoft's DialoGPT model on Rick and Morty script data. The resulting model can engage in conversation while mimicking the show's distinctive dialogue style.

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- Pandas
- NumPy
- Kaggle API
- tqdm
- tensorboardX (if torch.utils.tensorboard is not available)

## Installation

1. Install the required packages:

```bash
pip install torch transformers pandas numpy kaggle tqdm tensorboardX
```

2. Set up Kaggle API credentials:

- Place your `kaggle.json` file in the project directory
- The script will automatically copy it to the correct location

## Dataset

The project uses the "Rick and Morty Scripts" dataset from Kaggle, which contains dialogue from the show. The data is processed to create conversation contexts with:

- Current response
- 7 previous responses for context
- Automatic train/validation split (90/10)

## Model Architecture

- Base model: microsoft/DialoGPT-small
- Architecture: GPT-2 based transformer
- Fine-tuning approach: Causal language modeling (CLM)

## Training

The model is trained with the following parameters:

- Batch size: 4 per GPU
- Learning rate: 5e-5
- Number of epochs: 3
- Max sequence length: 512
- Gradient accumulation steps: 1
- Weight decay: 0.0
- Warm-up steps: 0

Training includes:

- Automatic checkpointing
- Tensorboard logging
- Distributed training support
- Mixed precision training (fp16) support
- Gradient clipping

## Usage

### Training the Model

```python
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

# Train the model
main(trn_df, val_df)
```

### Using the Trained Model

```python
# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained('output-small')

# Chat parameters
max_length = 200
no_repeat_ngram_size = 3
top_k = 100
top_p = 0.7
temperature = 0.8

# Generate responses
chat_history_ids = model.generate(
    input_ids,
    max_length=max_length,
    pad_token_id=tokenizer.eos_token_id,
    no_repeat_ngram_size=no_repeat_ngram_size,
    do_sample=True,
    top_k=top_k,
    top_p=top_p,
    temperature=temperature
)
```

## Generation Parameters

The model uses the following parameters for response generation:

- `max_length`: 200 tokens
- `no_repeat_ngram_size`: 3 (prevents repetition of 3-grams)
- `do_sample`: True (enables sampling-based generation)
- `top_k`: 100 (limits vocabulary to top 100 tokens)
- `top_p`: 0.7 (nucleus sampling parameter)
- `temperature`: 0.8 (controls randomness of sampling)

## Project Structure

```
.
├── output-small/          # Directory for saved model checkpoints
├── cached/               # Cache directory for processed datasets
├── RickAndMortyScripts.csv  # Dataset file
├── kaggle.json          # Kaggle API credentials
└── training_args.bin    # Saved training arguments
```

## Evaluation

The model is evaluated using perplexity on a held-out validation set. Evaluation is performed:

- During training (if enabled)
- After training
- On all checkpoints (if specified)

## Limitations

- The model's responses are probabilistic and may not always be coherent
- Training requires significant computational resources
- The model may generate inappropriate content based on the training data
- Response quality depends heavily on the input context

## License

Please refer to the original licenses for:

- DialoGPT model
- Rick and Morty dataset
- Hugging Face Transformers library

## Acknowledgments

- Microsoft for the DialoGPT model
- Hugging Face for the Transformers library
- Kaggle and the dataset creator
- Rick and Morty creators for the original content
