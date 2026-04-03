from __future__ import annotations

import speech_recognition as sr


def load_speech_recognition_groq():
    return sr.Recognizer()


def transcribe_speech_recognition_groq(recognizer: sr.Recognizer, wav_path: str) -> str:
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_groq(audio, model="whisper-large-v3-turbo", language="cs")
    except sr.UnknownValueError:
        raise Exception("Groq Speech Recognition could not understand audio")
    except Exception as e:
        raise Exception(f"Groq Speech Recognition error; {e}")
