setup venv:

python3 -m venv .venv

Activate venv ( from project root ):

source .venv/bin/activate

Update pip:

python -m pip install --upgrade pip

Install dependencies:

pip install -r requirements.txt

Development dependencies:

pip freeze > requirements.txt

See available models:

python transcription/main.py --help

Run transcription with selected model:

python transcription/main.py --model speech_recognition_faster_whisper_small

### Faster Whisper Implementations
- faster_whisper_large_float16_cuda: 🟡 slow
- faster_whisper_medium_float16_cuda: 🟢 3 tests

### HuggingFace (Wav2Vec2/Whisper fine-tuned models)
- huggingface_wav2vec2_xls_r_300m_czech: 🔴 returning "?....??..."
- huggingface_wav2vec2_xlsr_53_czech: 🟢 3 tests
- huggingface_wav2vec2_xlsr_czech: 🟢 3 tests
- huggingface_wav2vec2_xlsr_czech_sammy: 🟢 3 tests
- huggingface_whisper_large_v3_czech: 🟢 3 tests
- huggingface_whisper_medium_czech: 🟡 slow

### Direct Whisper Implementations
- whisper_large: 🟡 3 tests (3 000/15 000) slow
- whisper_medium: 🟢 3 tests
- whisper_small: 🟢 3 tests
- whisper_tiny: 🟢 5 tests
- whisper_base: 🟢 3 tests

### Speech Recognition Wrappers/APIs
- speech_recognition_google: 🟢 3 tests
- speech_recognition_openai: 🔴  "You exceeded your current quota, please check your plan and billing details."
- speech_recognition_groq: 🟡 very slow
- speech_recognition_sphinx: 🔴 EN only
- speech_recognition_vosk: 🟢 TODO
- speech_recognition_witai: 🔴 CZ experimental and not workign
