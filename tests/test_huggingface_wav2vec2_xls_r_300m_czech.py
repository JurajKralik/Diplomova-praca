import unittest

from _model_test_helper import run_model_test
from models import Model


class TestHuggingFaceWav2Vec2XlsR300mCzech(unittest.TestCase):
    def test_transcript(self) -> None:
        run_model_test(self, Model.HUGGINGFACE_WAV2VEC2_XLS_R_300M_CZECH)


if __name__ == "__main__":
    unittest.main()
