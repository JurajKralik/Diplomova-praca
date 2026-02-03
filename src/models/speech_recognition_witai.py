from __future__ import annotations

import speech_recognition as sr


def load_speech_recognition_witai():
	"""Load Wit.ai Speech Recognition (no model loading needed)."""
	return sr.Recognizer()


def transcribe_speech_recognition_witai(recognizer: sr.Recognizer, wav_path: str, api_key: str = None) -> str:
	"""Transcribe audio using Wit.ai Speech Recognition API.
	
	Args:
		recognizer: Speech recognition instance
		wav_path: Path to the WAV file
		api_key: Wit.ai API key (optional, can be set as environment variable)
	"""
	with sr.AudioFile(wav_path) as source:
		audio = recognizer.record(source)
	
	try:
		if api_key:
			return recognizer.recognize_wit(audio, key=api_key)
		else:
			# Will use WIT_AI_KEY environment variable
			return recognizer.recognize_wit(audio)
	except sr.UnknownValueError:
		return ""
	except sr.RequestError as e:
		raise Exception(f"Could not request results from Wit.ai service; {e}")
