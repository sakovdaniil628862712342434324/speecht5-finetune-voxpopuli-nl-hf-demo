# Model and processor checkpoints
MODEL_CHECKPOINT = "microsoft/speecht5_tts"
VOCODER_CHECKPOINT = "microsoft/speecht5_hifigan"
SPEAKER_RECOGNITION_MODEL = "speechbrain/spkrec-xvect-voxceleb"

# Training configuration
TRAINING_OUTPUT_DIR = "speecht5_finetuned_voxpopuli_nl"
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
MAX_STEPS = 4000
GRADIENT_CHECKPOINTING = True
FP16 = True
EVAL_STRATEGY = "steps"
PER_DEVICE_EVAL_BATCH_SIZE = 2
SAVE_STEPS = 1000
EVAL_STEPS = 1000
LOGGING_STEPS = 25

# Other configurations
MAX_INPUT_LENGTH = 200  # Filter out sequences longer than this
REDUCTION_FACTOR = 2   # Used in the data collator
