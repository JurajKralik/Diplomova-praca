import unittest

from _model_test_helper import run_model_test
from models import Model


class TestHuggingFaceWhisperLargeV3Czech(unittest.TestCase):
    def test_transcript(self) -> None:
        run_model_test(self, Model.HUGGINGFACE_WHISPER_LARGE_V3_CZECH)


if __name__ == "__main__":
    unittest.main()
