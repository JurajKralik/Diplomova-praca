import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTION_DIR = PROJECT_ROOT / "transcription"
SRC_DIR = TRANSCRIPTION_DIR / "src"
sys.path.insert(0, str(TRANSCRIPTION_DIR))
sys.path.insert(0, str(SRC_DIR))

from models import Model, transcript


SAMPLE_WAV = PROJECT_ROOT / "data" / "sample_short.wav"


LOCAL_MODELS = [
    Model.WHISPER_TINY,
    Model.WHISPER_BASE,
    Model.FASTER_WHISPER_MEDIUM,
]


ONLINE_MODELS = [
    Model.SPEECH_RECOGNITION_GOOGLE,
]


def _skip_if_missing(test_case: unittest.TestCase) -> None:
    if not SAMPLE_WAV.exists():
        test_case.skipTest(f"Missing sample wav: {SAMPLE_WAV}")


class TestModels(unittest.TestCase):
    def test_model_enum_not_empty(self) -> None:
        self.assertGreater(len(list(Model)), 0)

    def test_local_models_transcript(self) -> None:
        _skip_if_missing(self)
        any_success = False
        for model in LOCAL_MODELS:
            with self.subTest(model=model.value):
                text = transcript(str(SAMPLE_WAV), model)
                if text is not None:
                    any_success = True
                    self.assertIsInstance(text, str)
        self.assertTrue(any_success, "No local model produced a transcript")

    def test_google_transcript(self) -> None:
        _skip_if_missing(self)
        text = transcript(str(SAMPLE_WAV), Model.SPEECH_RECOGNITION_GOOGLE)
        if text is None:
            self.skipTest("Google Speech Recognition unavailable in current environment")
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
