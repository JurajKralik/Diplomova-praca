from __future__ import annotations

import whisper


def load_whisper(size: str):
	return whisper.load_model(size)


def transcribe_whisper(model, wav_path: str) -> str:
	result = model.transcribe(
		wav_path,
		language="cs",
		verbose=False,
	)
	return result["text"]
