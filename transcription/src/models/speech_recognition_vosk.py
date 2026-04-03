from __future__ import annotations

import speech_recognition as sr


def load_speech_recognition_vosk():
    return sr.Recognizer()


def transcribe_speech_recognition_vosk(recognizer: sr.Recognizer, wav_path: str) -> str:
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_vosk(audio)
    except sr.UnknownValueError:
        raise Exception("Vosk could not understand audio")
    except Exception as e:
        raise Exception(f"Vosk error; {e}")
