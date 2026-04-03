import unittest

from _model_test_helper import run_model_test
from models import Model


class TestSpeechRecognitionFasterWhisperBase(unittest.TestCase):
    def test_transcript(self) -> None:
        run_model_test(self, Model.SPEECH_RECOGNITION_FASTER_WHISPER_BASE)


if __name__ == "__main__":
    unittest.main()
