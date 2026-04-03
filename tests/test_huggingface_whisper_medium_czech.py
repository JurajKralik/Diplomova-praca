import unittest

from _model_test_helper import run_model_test
from models import Model


class TestHuggingFaceWhisperMediumCzech(unittest.TestCase):
    def test_transcript(self) -> None:
        run_model_test(self, Model.HUGGINGFACE_WHISPER_MEDIUM_CZECH)


if __name__ == "__main__":
    unittest.main()
