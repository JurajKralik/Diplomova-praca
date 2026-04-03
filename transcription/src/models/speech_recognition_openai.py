from __future__ import annotations

import speech_recognition as sr


def load_speech_recognition_openai():
    return sr.Recognizer()


def transcribe_speech_recognition_openai(recognizer: sr.Recognizer, wav_path: str) -> str:
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_openai(audio, model="gpt-4o-mini-transcribe", language="cs")
    except sr.UnknownValueError:
        raise Exception("OpenAI Speech Recognition could not understand audio")
    except Exception as e:
        raise Exception(f"OpenAI Speech Recognition error; {e}")
