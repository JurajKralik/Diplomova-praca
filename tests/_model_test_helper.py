import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTION_DIR = PROJECT_ROOT / "transcription"
SRC_DIR = TRANSCRIPTION_DIR / "src"
sys.path.insert(0, str(TRANSCRIPTION_DIR))
sys.path.insert(0, str(SRC_DIR))

from models import transcript  # noqa: E402


SAMPLE_WAV = PROJECT_ROOT / "data" / "sample_short.wav"


def run_model_test(test_case: unittest.TestCase, model) -> None:
    if not SAMPLE_WAV.exists():
        test_case.skipTest(f"Missing sample wav: {SAMPLE_WAV}")

    text = transcript(str(SAMPLE_WAV), model)
    if text is None:
        test_case.fail(f"Transcription returned None for model: {model.value}")
    else:
        print(text)
    test_case.assertIsInstance(text, str)
