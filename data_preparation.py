import os
from datasets import load_dataset, Audio
from transformers import SpeechT5Processor
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

from utils import create_speaker_embedding
from config import MODEL_CHECKPOINT, MAX_INPUT_LENGTH

def load_and_prepare_dataset():
    # Load the Dutch subset of VoxPopuli
    print("Loading VoxPopuli Dutch dataset...")
    dataset = load_dataset("facebook/voxpopuli", "nl", split="train")
    print(f"Total examples: {len(dataset)}")

    # Ensure audio is loaded at 16 kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Load the SpeechT5 processor (includes tokenizer and feature extractor)
    processor = SpeechT5Processor.from_pretrained(MODEL_CHECKPOINT)
    tokenizer = processor.tokenizer

    # --- Text Cleanup ---
    # Some special characters (e.g., à, ç, è, ë, í, ï, ö, ü) are not in the tokenizer's vocab.
    replacements = [
        ("à", "a"),
        ("ç", "c"),
        ("è", "e"),
        ("ë", "e"),
        ("í", "i"),
        ("ï", "i"),
        ("ö", "o"),
        ("ü", "u"),
    ]

    def cleanup_text(example):
        for src, dst in replacements:
            example["normalized_text"] = example["normalized_text"].replace(src, dst)
        return example

    dataset = dataset.map(cleanup_text)

    # --- Speaker Filtering ---
    speaker_counts = defaultdict(int)
    for speaker_id in dataset["speaker_id"]:
        speaker_counts[speaker_id] += 1

    def select_speaker(example):
        return 100 <= speaker_counts[example["speaker_id"]] <= 400

    dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
    print(f"After filtering speakers, examples: {len(dataset)}")
    print(f"Unique speakers: {len(set(dataset['speaker_id']))}")

    # --- Processing Examples ---
    def prepare_dataset(example):
        audio = example["audio"]
        # Process text and audio using the processor
        processed = processor(
            text=example["normalized_text"],
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )
        # Remove batch dimension from the spectrogram (labels)
        processed["labels"] = processed["labels"][0]
        # Generate speaker embedding using SpeechBrain
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processed["speaker_embeddings"] = create_speaker_embedding(audio["array"], device=device)
        return processed

    print("Processing dataset examples (this may take several minutes)...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

    # --- Filter Out Long Inputs ---
    def is_not_too_long(example):
        return len(example["input_ids"]) < MAX_INPUT_LENGTH

    dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
    print(f"Examples after filtering long inputs: {len(dataset)}")

    # --- Train/Test Split ---
    dataset = dataset.train_test_split(test_size=0.1)
    print("Dataset split into train and test sets.")

    return dataset, processor

def main():
    dataset, processor = load_and_prepare_dataset()
    # Optionally, you could save the processed dataset:
    # dataset["train"].save_to_disk("processed_train_dataset")
    # dataset["test"].save_to_disk("processed_test_dataset")
    print("Data preparation completed.")

if __name__ == "__main__":
    main()
