import unittest

from _model_test_helper import run_model_test
from models import Model


class TestSpeechRecognitionGroq(unittest.TestCase):
    def test_transcript(self) -> None:
        run_model_test(self, Model.SPEECH_RECOGNITION_GROQ)


if __name__ == "__main__":
    unittest.main()
