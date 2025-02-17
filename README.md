# Fine-tuning SpeechT5 for Dutch TTS

This repository demonstrates how to fine-tune the [SpeechT5](https://huggingface.co/microsoft/speecht5_tts) model for text-to-speech on the Dutch subset of the VoxPopuli dataset. It covers data preparation, model training, and inference.

## Repository Structure

```
speecht5-finetune-voxpopuli-nl/
├── README.md
├── requirements.txt
├── config.py
├── data_preparation.py
├── model_training.py
├── inference.py
└── utils.py
```

- **config.py**: Contains configuration parameters and hyperparameters.
- **data_preparation.py**: Loads and preprocesses the VoxPopuli Dutch dataset (including text cleanup, speaker filtering, and embedding extraction).
- **model_training.py**: Fine-tunes the SpeechT5 model on the prepared dataset.
- **inference.py**: Loads the fine-tuned model and generates speech from text.
- **utils.py**: Contains helper functions such as speaker embedding generation and a custom data collator.

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/speecht5-finetune-voxpopuli-nl.git
   cd speecht5-finetune-voxpopuli-nl
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Log in to Hugging Face** (if you wish to push your model to the Hub):

   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```

## Data Preparation

The data preparation script loads the Dutch VoxPopuli dataset, cleans the text, filters speakers, extracts speaker embeddings using SpeechBrain’s X-vector model, and processes the audio into log-mel spectrograms.

Run the script with:
  
```bash
python data_preparation.py
```

## Training

The training script fine-tunes the SpeechT5 model. You can adjust hyperparameters in `config.py` if needed.

To start training run:

```bash
python model_training.py
```

*Note:* Training might take several hours. If you encounter CUDA “out-of-memory” errors, try reducing the batch size and increasing the gradient accumulation steps.

## Inference

Once the model is trained (and pushed to the Hub or saved locally), you can generate speech using the inference script. This script loads the model, tokenizes a sample Dutch text, and synthesizes speech using a vocoder.

Run the inference script with:

```bash
python inference.py
```

The output speech is saved as `output.wav` (and, if you’re in an interactive environment, it will also be played).

## License

This project is licensed under the MIT License.
