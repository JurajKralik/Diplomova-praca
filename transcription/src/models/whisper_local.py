from __future__ import annotations

import torch
import whisper


def load_whisper(size: str):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	return whisper.load_model(size, device=device)


def transcribe_whisper(model, wav_path: str) -> str:
	result = model.transcribe(
		wav_path,
		language="cs",
		verbose=False,
	)
	return result["text"]
