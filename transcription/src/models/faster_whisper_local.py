from __future__ import annotations

import torch
from faster_whisper import WhisperModel


def load_faster_whisper(size: str, device: str = "cuda", compute_type: str = "float16"):
	if not torch.cuda.is_available():
		device = "cpu"
		if compute_type == "float16":
			compute_type = "int8"
	return WhisperModel(size, device=device, compute_type=compute_type)


def transcribe_faster_whisper(model: WhisperModel, wav_path: str) -> str:
	segments, _info = model.transcribe(wav_path, language="cs")
	return " ".join(seg.text.strip() for seg in segments)
