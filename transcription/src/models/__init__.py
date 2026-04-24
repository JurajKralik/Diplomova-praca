from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Final, Optional, Tuple

from .whisper_local import load_whisper, transcribe_whisper
from .faster_whisper_local import load_faster_whisper, transcribe_faster_whisper
from .speech_recognition_google import load_speech_recognition_google, transcribe_speech_recognition_google
from .speech_recognition_sphinx import load_speech_recognition_sphinx, transcribe_speech_recognition_sphinx
from .speech_recognition_witai import load_speech_recognition_witai, transcribe_speech_recognition_witai
from .speech_recognition_openai import load_speech_recognition_openai, transcribe_speech_recognition_openai
from .speech_recognition_groq import load_speech_recognition_groq, transcribe_speech_recognition_groq
from .speech_recognition_vosk import load_speech_recognition_vosk, transcribe_speech_recognition_vosk
from .huggingface_transformers_local import load_huggingface_asr, transcribe_huggingface_asr
from .huggingface_transformers_seq2seq_local import load_huggingface_seq2seq_asr, transcribe_huggingface_seq2seq_asr


class Model(Enum):
	WHISPER_TINY = "whisper_tiny"
	WHISPER_BASE = "whisper_base"
	WHISPER_SMALL = "whisper_small"
	WHISPER_MEDIUM = "whisper_medium"
	WHISPER_LARGE = "whisper_large"
	FASTER_WHISPER_MEDIUM = "faster_whisper_medium_float16_cuda"
	FASTER_WHISPER_LARGE = "faster_whisper_large_float16_cuda"
	SPEECH_RECOGNITION_GOOGLE = "speech_recognition_google"
	SPEECH_RECOGNITION_SPHINX = "speech_recognition_sphinx"
	SPEECH_RECOGNITION_WITAI = "speech_recognition_witai"
	SPEECH_RECOGNITION_OPENAI = "speech_recognition_openai"
	SPEECH_RECOGNITION_GROQ = "speech_recognition_groq"
	SPEECH_RECOGNITION_VOSK = "speech_recognition_vosk"
	HUGGINGFACE_WAV2VEC2_XLSR_53_CZECH = "huggingface_wav2vec2_xlsr_53_czech"
	HUGGINGFACE_WAV2VEC2_XLSR_CZECH = "huggingface_wav2vec2_xlsr_czech"
	HUGGINGFACE_WAV2VEC2_XLSR_CZECH_SAMMY = "huggingface_wav2vec2_xlsr_czech_sammy"
	HUGGINGFACE_WAV2VEC2_XLS_R_300M_CZECH = "huggingface_wav2vec2_xls_r_300m_czech"
	HUGGINGFACE_WHISPER_MEDIUM_CZECH = "huggingface_whisper_medium_czech"
	HUGGINGFACE_WHISPER_LARGE_V3_CZECH = "huggingface_whisper_large_v3_czech"


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
		label="Faster-Whisper medium float16 cuda (local)",
		loader=lambda: load_faster_whisper("medium", device="cuda", compute_type="float16"),
		transcriber=transcribe_faster_whisper,
	),
	Model.FASTER_WHISPER_LARGE: ModelSpec(
		model=Model.FASTER_WHISPER_LARGE,
		label="Faster-Whisper large float16 cuda (local)",
		loader=lambda: load_faster_whisper("large-v2", device="cuda", compute_type="float16"),
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
	Model.SPEECH_RECOGNITION_OPENAI: ModelSpec(
		model=Model.SPEECH_RECOGNITION_OPENAI,
		label="OpenAI Speech Recognition (online)",
		loader=load_speech_recognition_openai,
		transcriber=transcribe_speech_recognition_openai,
	),
	Model.SPEECH_RECOGNITION_GROQ: ModelSpec(
		model=Model.SPEECH_RECOGNITION_GROQ,
		label="Groq Speech Recognition (online)",
		loader=load_speech_recognition_groq,
		transcriber=transcribe_speech_recognition_groq,
	),
	Model.SPEECH_RECOGNITION_VOSK: ModelSpec(
		model=Model.SPEECH_RECOGNITION_VOSK,
		label="Vosk Speech Recognition (offline)",
		loader=load_speech_recognition_vosk,
		transcriber=transcribe_speech_recognition_vosk,
	),
	Model.HUGGINGFACE_WAV2VEC2_XLSR_53_CZECH: ModelSpec(
		model=Model.HUGGINGFACE_WAV2VEC2_XLSR_53_CZECH,
		label="Hugging Face wav2vec2-large-xlsr-53 Czech (local)",
		loader=lambda: load_huggingface_asr("MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-Czech"),
		transcriber=transcribe_huggingface_asr,
	),
	Model.HUGGINGFACE_WAV2VEC2_XLSR_CZECH: ModelSpec(
		model=Model.HUGGINGFACE_WAV2VEC2_XLSR_CZECH,
		label="Hugging Face wav2vec2-large-xlsr Czech (local)",
		loader=lambda: load_huggingface_asr("arampacha/wav2vec2-large-xlsr-czech"),
		transcriber=transcribe_huggingface_asr,
	),
	Model.HUGGINGFACE_WAV2VEC2_XLSR_CZECH_SAMMY: ModelSpec(
		model=Model.HUGGINGFACE_WAV2VEC2_XLSR_CZECH_SAMMY,
		label="Hugging Face wav2vec2-xlsr Czech sammy786 (local)",
		loader=lambda: load_huggingface_asr("sammy786/wav2vec2-xlsr-czech"),
		transcriber=transcribe_huggingface_asr,
	),
	Model.HUGGINGFACE_WAV2VEC2_XLS_R_300M_CZECH: ModelSpec(
		model=Model.HUGGINGFACE_WAV2VEC2_XLS_R_300M_CZECH,
		label="Hugging Face wav2vec2-large-xls-r-300m Czech (local)",
		loader=lambda: load_huggingface_asr("Roxysun/wav2vec2-large-xls-r-300m-czech-colab-finetuned"),
		transcriber=transcribe_huggingface_asr,
	),
	Model.HUGGINGFACE_WHISPER_MEDIUM_CZECH: ModelSpec(
		model=Model.HUGGINGFACE_WHISPER_MEDIUM_CZECH,
		label="Hugging Face Whisper medium Czech (local)",
		loader=lambda: load_huggingface_seq2seq_asr("mikr/whisper-medium-czech-cv11"),
		transcriber=transcribe_huggingface_seq2seq_asr,
	),
	Model.HUGGINGFACE_WHISPER_LARGE_V3_CZECH: ModelSpec(
		model=Model.HUGGINGFACE_WHISPER_LARGE_V3_CZECH,
		label="Hugging Face Whisper large-v3 Czech (local)",
		loader=lambda: load_huggingface_seq2seq_asr("mikr/whisper-large-v3-czech-cv13"),
		transcriber=transcribe_huggingface_seq2seq_asr,
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
	except KeyboardInterrupt:
		print("Transcription interrupted by user.")
		return None
	except Exception as e:
		print(f"Transcription failed: \n{e}")
		return None