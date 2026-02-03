from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Final, Optional, Tuple

from .whisper_local import load_whisper, transcribe_whisper
from .faster_whisper_local import load_faster_whisper, transcribe_faster_whisper
from .speech_recognition_google import load_speech_recognition_google, transcribe_speech_recognition_google
from .speech_recognition_sphinx import load_speech_recognition_sphinx, transcribe_speech_recognition_sphinx
from .speech_recognition_witai import load_speech_recognition_witai, transcribe_speech_recognition_witai


class Model(Enum):
	WHISPER_TINY = "whisper_tiny"
	WHISPER_BASE = "whisper_base"
	WHISPER_SMALL = "whisper_small"
	WHISPER_MEDIUM = "whisper_medium"
	WHISPER_LARGE = "whisper_large"
	FASTER_WHISPER_MEDIUM = "faster_whisper_medium_int8_cpu"
	FASTER_WHISPER_LARGE = "faster_whisper_large_int8_cpu"
	SPEECH_RECOGNITION_GOOGLE = "speech_recognition_google"
	SPEECH_RECOGNITION_SPHINX = "speech_recognition_sphinx"
	SPEECH_RECOGNITION_WITAI = "speech_recognition_witai"


@dataclass(frozen=True)
class ModelSpec:
	model: Model
	label: str
	loader: Callable[[], object]
	transcriber: Callable[[object, str], str]


SPECS: Final[Dict[Model, ModelSpec]] = {
	Model.WHISPER_TINY: ModelSpec(
		model=Model.WHISPER_TINY,
		label="Whisper tiny (local)",
		loader=lambda: load_whisper("tiny"),
		transcriber=transcribe_whisper,
	),
	Model.WHISPER_BASE: ModelSpec(
		model=Model.WHISPER_BASE,
		label="Whisper base (local)",
		loader=lambda: load_whisper("base"),
		transcriber=transcribe_whisper,
	),
	Model.WHISPER_SMALL: ModelSpec(
		model=Model.WHISPER_SMALL,
		label="Whisper small (local)",
		loader=lambda: load_whisper("small"),
		transcriber=transcribe_whisper,
	),
	Model.WHISPER_MEDIUM: ModelSpec(
		model=Model.WHISPER_MEDIUM,
		label="Whisper medium (local)",
		loader=lambda: load_whisper("medium"),
		transcriber=transcribe_whisper,
	),
	Model.WHISPER_LARGE: ModelSpec(
		model=Model.WHISPER_LARGE,
		label="Whisper large (local)",
		loader=lambda: load_whisper("large"),
		transcriber=transcribe_whisper,
	),

	Model.FASTER_WHISPER_MEDIUM: ModelSpec(
		model=Model.FASTER_WHISPER_MEDIUM,
		label="Faster-Whisper medium int8 cpu (local)",
		loader=lambda: load_faster_whisper("medium", device="cpu", compute_type="int8"),
		transcriber=transcribe_faster_whisper,
	),
	Model.FASTER_WHISPER_LARGE: ModelSpec(
		model=Model.FASTER_WHISPER_LARGE,
		label="Faster-Whisper large int8 cpu (local)",
		loader=lambda: load_faster_whisper("large-v2", device="cpu", compute_type="int8"),
		transcriber=transcribe_faster_whisper,
	),

	Model.SPEECH_RECOGNITION_GOOGLE: ModelSpec(
		model=Model.SPEECH_RECOGNITION_GOOGLE,
		label="Google Speech Recognition (online)",
		loader=load_speech_recognition_google,
		transcriber=transcribe_speech_recognition_google,
	),
	Model.SPEECH_RECOGNITION_SPHINX: ModelSpec(
		model=Model.SPEECH_RECOGNITION_SPHINX,
		label="Sphinx Speech Recognition (offline)",
		loader=load_speech_recognition_sphinx,
		transcriber=transcribe_speech_recognition_sphinx,
	),
	Model.SPEECH_RECOGNITION_WITAI: ModelSpec(
		model=Model.SPEECH_RECOGNITION_WITAI,
		label="Wit.ai Speech Recognition (online)",
		loader=load_speech_recognition_witai,
		transcriber=transcribe_speech_recognition_witai,
	),
}

_CURRENT: Optional[Tuple[Model, object]] = None


def _get_loaded_model(spec: ModelSpec) -> object:
	global _CURRENT

	if _CURRENT is not None:
		current_id, current_obj = _CURRENT
		if current_id == spec.model:
			return current_obj

	obj = spec.loader()
	_CURRENT = (spec.model, obj)
	return obj


def transcript(wav_path: str, model: Model) -> str:
	try:
		if model not in SPECS:
			raise ValueError(f"Unknown model: {model}")

		spec = SPECS[model]
		loaded = _get_loaded_model(spec)
		return spec.transcriber(loaded, wav_path).strip()
	except Exception as e:
		print(f"Transcription failed: \n{e}")
		return None