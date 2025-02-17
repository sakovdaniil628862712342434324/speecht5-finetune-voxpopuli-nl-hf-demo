import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from config import MODEL_CHECKPOINT, VOCODER_CHECKPOINT

def main():
    # Ask for the model name (this could be your Hugging Face repo, e.g., "YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
    model_name = input("Enter the model name (e.g., YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl): ").strip()
    print("Loading fine-tuned SpeechT5 model...")
    model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
    processor = SpeechT5Processor.from_pretrained(MODEL_CHECKPOINT)

    # Load the vocoder
    print("Loading vocoder...")
    vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_CHECKPOINT)

    # Define a sample Dutch input text
    text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
    print(f"Input text: {text}")

    # Tokenize the input text
    inputs = processor(text=text, return_tensors="pt")

    # For the speaker embedding, we pick a sample from the VoxPopuli test set
    print("Loading a sample speaker embedding from VoxPopuli test set...")
    dataset = load_dataset("facebook/voxpopuli", "nl", split="test")
    sample = dataset[0]
    speaker_waveform = sample["audio"]["array"]

    # Generate a speaker embedding using our helper function
    from utils import create_speaker_embedding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_embeddings = torch.tensor(create_speaker_embedding(speaker_waveform, device=device)).unsqueeze(0)

    # Generate speech from the input text and speaker embedding
    print("Generating speech...")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Save the output speech to a WAV file
    output_path = "output.wav"
    sf.write(output_path, speech.numpy(), 16000)
    print(f"Output speech saved to {output_path}")

    # If you are running this in a Jupyter Notebook, you can play the audio:
    try:
        from IPython.display import Audio, display
        display(Audio(speech.numpy(), rate=16000))
    except Exception:
        print("Audio playback is not supported in this environment.")

if __name__ == "__main__":
    main()
