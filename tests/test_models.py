import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from models import Model, transcript


SAMPLE_WAV = PROJECT_ROOT / "data" / "sample.wav"


def _skip_if_missing(test_case: unittest.TestCase) -> None:
    if not SAMPLE_WAV.exists():
        test_case.skipTest(f"Missing sample wav: {SAMPLE_WAV}")


class TestModels(unittest.TestCase):
    def test_whisper_medium_transcript(self) -> None:
        _skip_if_missing(self)
        text = transcript(str(SAMPLE_WAV), Model.WHISPER_MEDIUM)
        self.assertIsNotNone(text)
        self.assertIsInstance(text, str)

    def test_faster_whisper_medium_transcript(self) -> None:
        _skip_if_missing(self)
        text = transcript(str(SAMPLE_WAV), Model.FASTER_WHISPER_MEDIUM)
        self.assertIsNotNone(text)
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
