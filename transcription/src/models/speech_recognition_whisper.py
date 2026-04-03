from __future__ import annotations

import speech_recognition as sr


def load_speech_recognition_whisper(model_size: str):
    return (sr.Recognizer(), model_size)


def transcribe_speech_recognition_whisper(loaded: tuple[sr.Recognizer, str], wav_path: str) -> str:
    recognizer, model_size = loaded
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_whisper(audio, model=model_size, language="czech")
    except sr.UnknownValueError:
        raise Exception("SpeechRecognition Whisper could not understand audio")
    except Exception as e:
        raise Exception(f"SpeechRecognition Whisper error; {e}")
