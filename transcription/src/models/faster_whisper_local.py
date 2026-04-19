from __future__ import annotations

import torch
from faster_whisper import WhisperModel


def load_faster_whisper(size: str, device: str = "cuda", compute_type: str = "float16"):
	if not torch.cuda.is_available():
		device = "cpu"
		if compute_type == "float16":
			compute_type = "int8"
	
	if device == "cuda":
		try:
			print(f"Loading Faster-Whisper {size} on CUDA with {compute_type}...")
			model = WhisperModel(size, device=device, compute_type=compute_type)
			return model
		except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
			if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
				print(f"CUDA out of memory for Faster-Whisper {size}, falling back to CPU...")
				torch.cuda.empty_cache()
				device = "cpu"
				compute_type = "int8" if compute_type == "float16" else compute_type
			else:
				raise
	
	print(f"Loading Faster-Whisper {size} on CPU with {compute_type}...")
	return WhisperModel(size, device=device, compute_type=compute_type)


def transcribe_faster_whisper(model: WhisperModel, wav_path: str) -> str:
	try:
		segments, _info = model.transcribe(wav_path, language="cs")
		return " ".join(seg.text.strip() for seg in segments)
	except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
		if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
			print(f"CUDA out of memory during transcription, recreating model on CPU...")
			torch.cuda.empty_cache()
			# Recreate model on CPU
			model_size = model.model_size_or_path
			cpu_model = WhisperModel(model_size, device="cpu", compute_type="int8")
			segments, _info = cpu_model.transcribe(wav_path, language="cs")
			return " ".join(seg.text.strip() for seg in segments)
		else:
			raise
