from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.models import transcript, Model


# Validate we're running from the correct directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / ".env").exists():
    print("Error: .env file not found")
    print(f"       Expected at: {PROJECT_ROOT / '.env'}")
    print(f"       Current working directory: {Path.cwd()}")
    print(f"       Please run this script from the Diplomova-praca directory:")
    print(f"       cd Diplomova-praca")
    print(f"       python transcription/main.py --model <model_name>")
    sys.exit(1)

load_dotenv(PROJECT_ROOT / ".env")

try:
    INPUT_DIR = PROJECT_ROOT / os.environ["COMMONVOICE_VALIDATED_WAV_DIR"]
except KeyError:
    print("Error: COMMONVOICE_VALIDATED_WAV_DIR environment variable not set in .env file")
    sys.exit(1)

try:
    OUTPUT_DIR = PROJECT_ROOT / os.environ["TRANSCRIPTION_OUTPUT_DIR"]
except KeyError:
    print("Error: TRANSCRIPTION_OUTPUT_DIR environment variable not set in .env file")
    sys.exit(1)


def get_audio_duration_seconds(file_path: Path) -> float | None:
    try:
        with wave.open(str(file_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate == 0:
                return None
            return frames / float(rate)
    except wave.Error:
        return None


def parse_arguments() -> Model:
    parser = argparse.ArgumentParser(description="Transcribe audio files using a selected model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[m.value for m in Model],
        help="Model to use for transcription"
    )
    args = parser.parse_args()
    
    for model in Model:
        if model.value == args.model:
            return model
    
    print(f"Invalid model: {args.model}")
    sys.exit(1)


def build_output_path(selected_model: Model) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = selected_model.value
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{model_name}_{timestamp}.json"


def save_progress(output_path: Path, items: list[dict], total: int, started_at: datetime, selected_model: Model) -> None:
    payload = {
        "model": selected_model.value,
        "created_at": started_at.isoformat(),
        "updated_at": datetime.now().isoformat(),
        "total_files": total,
        "completed": len(items) == total,
        "items": items,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    selected_model = parse_arguments()
    
    if not INPUT_DIR.exists():
        print(f"Error: Input folder does not exist: {INPUT_DIR}")
        print(f"(configured via COMMONVOICE_VALIDATED_WAV_DIR in .env)")
        return

    files = sorted(path for path in INPUT_DIR.iterdir() if path.suffix.lower() == ".wav")
    if not files:
        print(f"Error: No WAV files found in: {INPUT_DIR}")
        print(f"(configured via COMMONVOICE_VALIDATED_WAV_DIR in .env)")
        return

    started_at = datetime.now()
    output_path = build_output_path(selected_model)
    results: list[dict] = []
    total = len(files)

    save_progress(output_path, results, total, started_at, selected_model)
    print(f"Saving progress to: {output_path}")

    for index, file_path in enumerate(files, start=1):
        started_perf = time.perf_counter()
        item = {
            "file_name": file_path.name,
            "audio_duration_seconds": get_audio_duration_seconds(file_path),
            "transcript_duration_seconds": None,
            "text": None,
        }

        try:
            text = transcript(str(file_path), selected_model)
            transcript_duration = time.perf_counter() - started_perf
            item["transcript_duration_seconds"] = round(transcript_duration, 3)
            item["text"] = text
            print(f"({index}/{total}) Processed: {file_path.name}")
        except Exception as e:
            transcript_duration = time.perf_counter() - started_perf
            item["transcript_duration_seconds"] = round(transcript_duration, 3)
            print(f"({index}/{total}) Failed: {file_path.name} -> {e}")

        results.append(item)
        save_progress(output_path, results, total, started_at, selected_model)

    print(f"Saved final output: {output_path}")


if __name__ == "__main__":
    main()
