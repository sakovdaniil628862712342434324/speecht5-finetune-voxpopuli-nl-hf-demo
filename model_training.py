import torch
from transformers import SpeechT5ForTextToSpeech, Seq2SeqTrainingArguments, Seq2SeqTrainer, SpeechT5Processor
from functools import partial

from data_preparation import load_and_prepare_dataset
from utils import TTSDataCollatorWithPadding
from config import (
    MODEL_CHECKPOINT,
    TRAINING_OUTPUT_DIR,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WARMUP_STEPS,
    MAX_STEPS,
    GRADIENT_CHECKPOINTING,
    FP16,
    EVAL_STRATEGY,
    PER_DEVICE_EVAL_BATCH_SIZE,
    SAVE_STEPS,
    EVAL_STEPS,
    LOGGING_STEPS,
    REDUCTION_FACTOR,
)

def main():
    # Load and prepare the dataset and processor
    dataset, processor = load_and_prepare_dataset()

    # Load the pre-trained SpeechT5 model
    print("Loading pre-trained SpeechT5 model...")
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_CHECKPOINT)
    # Disable the use of cache during training (incompatible with gradient checkpointing)
    model.config.use_cache = False
    model.generate = partial(model.generate, use_cache=True)

    # Instantiate the custom data collator
    data_collator = TTSDataCollatorWithPadding(processor=processor, reduction_factor=REDUCTION_FACTOR)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=TRAINING_OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        fp16=FP16,
        evaluation_strategy=EVAL_STRATEGY,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        push_to_hub=True,
    )

    # Instantiate the Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Push the final model to the Hugging Face Hub
    trainer.push_to_hub()
    print("Training completed and model pushed to the Hub.")

if __name__ == "__main__":
    main()
