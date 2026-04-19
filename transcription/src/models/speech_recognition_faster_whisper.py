from __future__ import annotations

import torch
import speech_recognition as sr


def load_speech_recognition_faster_whisper(model_size: str, device: str = "cuda", compute_type: str = "float16"):
    if not torch.cuda.is_available():
        device = "cpu"
        if compute_type == "float16":
            compute_type = "int8"
    return (sr.Recognizer(), model_size, device, compute_type)


def transcribe_speech_recognition_faster_whisper(loaded: tuple[sr.Recognizer, str, str, str], wav_path: str) -> str:
    recognizer, model_size, device, compute_type = loaded
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_faster_whisper(
            audio,
            model=model_size,
            language="cs",
            init_options={"device": device, "compute_type": compute_type},
        )
    except sr.UnknownValueError:
        raise Exception("SpeechRecognition Faster-Whisper could not understand audio")
    except Exception as e:
        raise Exception(f"SpeechRecognition Faster-Whisper error; {e}")
