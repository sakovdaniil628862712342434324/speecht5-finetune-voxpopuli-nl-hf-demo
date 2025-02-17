import os
import torch
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def create_speaker_embedding(waveform, device="cpu", model_dir="/tmp/speechbrain_spkrec"):
    """
    Generate a 512-dimensional speaker embedding using SpeechBrain's x-vector model.
    """
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": device},
        savedir=os.path.join(model_dir, "speechbrain/spkrec-xvect-voxceleb")
    )
    with torch.no_grad():
        if not torch.is_tensor(waveform):
            waveform = torch.tensor(waveform)
        # Add batch dimension and compute the embedding
        speaker_embeddings = speaker_model.encode_batch(waveform.unsqueeze(0))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

class TTSDataCollatorWithPadding:
    """
    Custom data collator for TTS training.
    Pads input_ids and spectrogram labels; replaces padded label regions with -100.
    Also adds speaker embeddings to the batch.
    """
    def __init__(self, processor, reduction_factor=2):
        self.processor = processor
        self.reduction_factor = reduction_factor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # Use the processor to pad input IDs and spectrograms
        batch = self.processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # Replace padding in labels with -100 so that loss is not computed there
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )
        del batch["decoder_attention_mask"]

        # Adjust target length to be a multiple of the reduction factor
        if self.reduction_factor > 1:
            target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
            target_lengths = target_lengths - (target_lengths % self.reduction_factor)
            max_length = int(target_lengths.max().item())
            batch["labels"] = batch["labels"][:, :max_length]

        # Add speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)
        return batch
