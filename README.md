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
