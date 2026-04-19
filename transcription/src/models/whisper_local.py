from __future__ import annotations

import torch
import whisper


def load_whisper(size: str):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
	if device == "cuda":
		try:
			print(f"Loading Whisper {size} on CUDA...")
			model = whisper.load_model(size, device=device)
			return model
		except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
			if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
				print(f"CUDA out of memory for Whisper {size}, falling back to CPU...")
				torch.cuda.empty_cache()
				device = "cpu"
			else:
				raise
	
	print(f"Loading Whisper {size} on CPU...")
	return whisper.load_model(size, device=device)


def transcribe_whisper(model, wav_path: str) -> str:
	try:
		result = model.transcribe(
			wav_path,
			language="cs",
			verbose=False,
		)
		return result["text"]
	except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
		if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
			print(f"CUDA out of memory during transcription, retrying on CPU...")
			torch.cuda.empty_cache()
			# Move model to CPU and retry
			model = model.to("cpu")
			result = model.transcribe(
				wav_path,
				language="cs",
				verbose=False,
			)
			return result["text"]
		else:
			raise
