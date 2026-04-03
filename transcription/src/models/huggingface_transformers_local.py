from __future__ import annotations

from pathlib import Path

import torch
from transformers import pipeline


CACHE_DIR = Path(__file__).resolve().parents[3] / ".hf-cache"


def load_huggingface_asr(model_id: str):
    return pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=-1,
        model_kwargs={"cache_dir": str(CACHE_DIR)},
    )


def transcribe_huggingface_asr(pipe, wav_path: str) -> str:
    result = pipe(wav_path)
    if isinstance(result, dict):
        return str(result.get("text", ""))
    return str(result)
