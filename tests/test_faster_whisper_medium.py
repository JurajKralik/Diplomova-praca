import unittest

from _model_test_helper import run_model_test
from models import Model


class TestFasterWhisperMedium(unittest.TestCase):
    def test_transcript(self) -> None:
        run_model_test(self, Model.FASTER_WHISPER_MEDIUM)


if __name__ == "__main__":
    unittest.main()
