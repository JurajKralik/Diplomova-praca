from __future__ import annotations

import speech_recognition as sr


def load_speech_recognition_google():
	"""Load Google Speech Recognition (no model loading needed)."""
	return sr.Recognizer()


def transcribe_speech_recognition_google(recognizer: sr.Recognizer, wav_path: str) -> str:
	"""Transcribe audio using Google Speech Recognition API."""
	with sr.AudioFile(wav_path) as source:
		audio = recognizer.record(source)
	
	try:
		return recognizer.recognize_google(audio, language="cs-CZ")
	except sr.UnknownValueError:
		raise Exception("Google Speech Recognition could not understand audio")
	except sr.RequestError as e:
		raise Exception(f"Could not request results from Google Speech Recognition service; {e}")
