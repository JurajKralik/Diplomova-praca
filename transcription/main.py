from __future__ import annotations

import json
import os
import time
import wave
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.models import transcript, Model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

INPUT_DIR = PROJECT_ROOT / os.environ["COMMONVOICE_VALIDATED_WAV_DIR"]
OUTPUT_DIR = PROJECT_ROOT / os.environ["TRANSCRIPTION_OUTPUT_DIR"]
SELECTED_MODEL = Model.SPEECH_RECOGNITION_FASTER_WHISPER_SMALL


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


def build_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = SELECTED_MODEL.value
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{model_name}_{timestamp}.json"


def save_progress(output_path: Path, items: list[dict], total: int, started_at: datetime) -> None:
    payload = {
        "model": SELECTED_MODEL.value,
        "input_dir": str(INPUT_DIR),
        "created_at": started_at.isoformat(),
        "updated_at": datetime.now().isoformat(),
        "total_files": total,
        "processed_files": len(items),
        "remaining_files": total - len(items),
        "completed": len(items) == total,
        "items": items,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Input folder does not exist: {INPUT_DIR}")
        return

    files = sorted(path for path in INPUT_DIR.iterdir() if path.suffix.lower() == ".wav")
    if not files:
        print(f"No WAV files found in: {INPUT_DIR}")
        return

    started_at = datetime.now()
    output_path = build_output_path()
    results: list[dict] = []
    total = len(files)

    save_progress(output_path, results, total, started_at)
    print(f"Saving progress to: {output_path}")

    for index, file_path in enumerate(files, start=1):
        started_perf = time.perf_counter()
        item = {
            "file_name": file_path.name,
            "audio_duration_seconds": get_audio_duration_seconds(file_path),
            "transcript_duration_seconds": None,
            "text": None,
            "error": None,
        }

        try:
            text = transcript(str(file_path), SELECTED_MODEL)
            transcript_duration = time.perf_counter() - started_perf
            item["transcript_duration_seconds"] = round(transcript_duration, 3)
            item["text"] = text
            print(f"({index}/{total}) Processed: {file_path.name}")
        except Exception as e:
            transcript_duration = time.perf_counter() - started_perf
            item["transcript_duration_seconds"] = round(transcript_duration, 3)
            item["error"] = str(e)
            print(f"({index}/{total}) Failed: {file_path.name} -> {e}")

        results.append(item)
        save_progress(output_path, results, total, started_at)

    print(f"Saved final output: {output_path}")


if __name__ == "__main__":
    main()
