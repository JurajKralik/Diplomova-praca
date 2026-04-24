from __future__ import annotations

from pathlib import Path

import torch
from transformers import pipeline


CACHE_DIR = Path(__file__).resolve().parents[3] / ".hf-cache"


def load_huggingface_seq2seq_asr(model_id: str):
    device = 0 if torch.cuda.is_available() else -1
    
    if device == 0:
        try:
            print(f"Loading HuggingFace Seq2Seq ASR {model_id} on CUDA...")
            pipe = pipeline(
                task="automatic-speech-recognition",
                model=model_id,
                device=device,
                model_kwargs={"cache_dir": str(CACHE_DIR)},
            )
            pipe.model.generation_config.forced_decoder_ids = None
            return pipe
        except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"CUDA out of memory for {model_id}, falling back to CPU...")
                torch.cuda.empty_cache()
                device = -1
            else:
                raise
    
    print(f"Loading HuggingFace Seq2Seq ASR {model_id} on CPU...")
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device,
        model_kwargs={"cache_dir": str(CACHE_DIR)},
    )
    pipe.model.generation_config.forced_decoder_ids = None
    return pipe


def transcribe_huggingface_seq2seq_asr(pipe, wav_path: str) -> str:
    forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="czech", task="transcribe")
    try:
        result = pipe(wav_path, generate_kwargs={"forced_decoder_ids": forced_decoder_ids})
        if isinstance(result, dict):
            return str(result.get("text", ""))
        return str(result)
    except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"CUDA out of memory during transcription, moving pipeline to CPU...")
            torch.cuda.empty_cache()
            pipe.model = pipe.model.to("cpu")
            result = pipe(wav_path, generate_kwargs={"forced_decoder_ids": forced_decoder_ids})
            if isinstance(result, dict):
                return str(result.get("text", ""))
            return str(result)
        else:
            raise
