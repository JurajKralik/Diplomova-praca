from __future__ import annotations

from faster_whisper import WhisperModel


def load_faster_whisper(size: str, device: str = "cpu", compute_type: str = "int8"):
	return WhisperModel(size, device=device, compute_type=compute_type)


def transcribe_faster_whisper(model: WhisperModel, wav_path: str) -> str:
	segments, _info = model.transcribe(wav_path, language="cs")
	return " ".join(seg.text.strip() for seg in segments)
