setup venv:

python3 -m venv venv

Activate venv:

source venv/bin/activate

Update pip:

python -m pip install --upgrade pip

Install dependencies:

pip install -r requirements.txt

Development dependencies:

pip freeze > requirements.txt