import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging

logging.getLogger(
    "transformers"
).setLevel(logging.ERROR)

import numpy as np
import librosa
import torch

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq
)

from config import (
    WHISPER_MODEL_NAME,
    WHISPER_SAMPLING_RATE
)

logging.getLogger("transformers").setLevel(
    logging.ERROR
)

print("Loading Whisper Tiny...")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

processor = AutoProcessor.from_pretrained(
    WHISPER_MODEL_NAME
)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_NAME
)

model.to(device)
model.eval()

print("Whisper Ready")


def load_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if sr != WHISPER_SAMPLING_RATE:

        audio = librosa.resample(
            y=audio,
            orig_sr=sr,
            target_sr=WHISPER_SAMPLING_RATE
        )

    return audio.astype(np.float32)


def extract_voice_embedding(audio_path):

    waveform = load_audio(
        audio_path
    )

    inputs = processor(
        waveform,
        sampling_rate=WHISPER_SAMPLING_RATE,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    ).input_features.to(device)

    with torch.no_grad():

        encoder_outputs = (
            model.model.encoder(inputs)
        )

        embedding = (
            encoder_outputs
            .last_hidden_state
            .mean(dim=1)
            .cpu()
            .numpy()[0]
        )

    return embedding.astype(
        np.float32
    )