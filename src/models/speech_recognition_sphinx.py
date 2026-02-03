from __future__ import annotations

import speech_recognition as sr


def load_speech_recognition_sphinx():
	"""Load Sphinx Speech Recognition (no model loading needed)."""
	return sr.Recognizer()


def transcribe_speech_recognition_sphinx(recognizer: sr.Recognizer, wav_path: str) -> str:
	"""Transcribe audio using Sphinx (offline) Speech Recognition."""
	with sr.AudioFile(wav_path) as source:
		audio = recognizer.record(source)
	
	try:
		return recognizer.recognize_sphinx(audio, language="cs-CZ")
	except sr.UnknownValueError:
		raise Exception("Sphinx could not understand audio")
	except sr.RequestError as e:
		raise Exception(f"Sphinx error; {e}")
