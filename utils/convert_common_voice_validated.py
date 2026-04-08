from pathlib import Path
import os

from dotenv import load_dotenv
from pydub import AudioSegment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

INPUT_DIR = PROJECT_ROOT / os.environ["COMMONVOICE_VALIDATED_DIR"]
OUTPUT_DIR = PROJECT_ROOT / os.environ["COMMONVOICE_VALIDATED_WAV_DIR"]


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input folder does not exist: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in INPUT_DIR.iterdir() if path.suffix.lower() == ".mp3")

    if not files:
        print(f"No MP3 files found in: {INPUT_DIR}")
        return

    total = len(files)
    for index, file_path in enumerate(files, start=1):
        wav_path = OUTPUT_DIR / f"{file_path.stem}.wav"
        if wav_path.exists():
            print(f"({index}/{total}) Skipped existing: {wav_path.name}")
            continue

        audio = AudioSegment.from_mp3(file_path)
        audio.export(wav_path, format="wav")
        print(f"({index}/{total}) Converted: {file_path.name} -> {wav_path.name}")


if __name__ == "__main__":
    main()
