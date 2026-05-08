from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from jiwer import cer, wer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_DIR = PROJECT_ROOT / os.environ["TRANSCRIPTION_OUTPUT_DIR"]
VALIDATED_TSV = PROJECT_ROOT / os.environ["VALIDATED_TSV_PATH"]


def build_reference_map(tsv_path: Path) -> dict[str, dict[str, str]]:
    references: dict[str, dict[str, str]] = {}
    with tsv_path.open("r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        for row in reader:
            rel_path = row.get("path", "").strip()
            sentence = row.get("sentence", "").strip()
            if not rel_path:
                continue
            stem = Path(rel_path).stem
            references[stem] = {
                "path": rel_path,
                "reference_text": sentence,
            }
    return references


def get_all_transcription_jsons(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.glob("*.json"))


def build_output_path(source_json: Path, validated_dir: Path) -> Path:
    return validated_dir / f"{source_json.stem}_validated.json"


def validate_json(source_json: Path, references: dict, validated_dir: Path) -> None:
    output_path = build_output_path(source_json, validated_dir)
    if output_path.exists():
        print(f"Skipping (already validated): {source_json.name}")
        return

    source_payload = json.loads(source_json.read_text(encoding="utf-8"))
    source_items = source_payload.get("items", [])

    created_at_str = source_payload.get("created_at")
    updated_at_str = source_payload.get("updated_at")
    elapsed_seconds: float | None = None
    if created_at_str and updated_at_str:
        try:
            elapsed_seconds = (
                datetime.fromisoformat(updated_at_str) - datetime.fromisoformat(created_at_str)
            ).total_seconds()
        except ValueError:
            pass

    validated_items: list[dict] = []
    matched_count = 0
    wer_values: list[float] = []
    cer_values: list[float] = []
    transcript_ratio_values: list[float] = []

    for item in source_items:
        file_name = item.get("file_name")
        stem = Path(file_name).stem if file_name else None
        reference = references.get(stem)
        predicted_text = item.get("text")
        reference_text = reference["reference_text"] if reference else None

        item_result = {
            "file_name": file_name,
            "clip_path": reference["path"] if reference else None,
            "reference_text": reference_text,
            "predicted_text": predicted_text,
            "audio_duration_seconds": item.get("audio_duration_seconds"),
            "transcript_duration_seconds": item.get("transcript_duration_seconds"),
            "error": item.get("error"),
            "matched_reference": reference is not None,
            "wer": None,
            "cer": None,
        }

        if reference_text is not None and predicted_text is not None and item.get("error") is None:
            item_result["wer"] = wer(reference_text, predicted_text)
            item_result["cer"] = cer(reference_text, predicted_text)
            wer_values.append(item_result["wer"])
            cer_values.append(item_result["cer"])
            matched_count += 1

        audio_dur = item.get("audio_duration_seconds")
        transcript_dur = item.get("transcript_duration_seconds")
        if audio_dur and transcript_dur is not None and audio_dur > 0:
            transcript_ratio_values.append(transcript_dur / audio_dur)

        validated_items.append(item_result)

    payload = {
        "source_json": str(source_json.relative_to(PROJECT_ROOT)),
        "validated_tsv": str(VALIDATED_TSV.relative_to(PROJECT_ROOT)),
        "model": source_payload.get("model"),
        "created_at": datetime.now().isoformat(),
        "total_items": len(source_items),
        "matched_items": matched_count,
        "average_wer": (sum(wer_values) / len(wer_values)) if wer_values else None,
        "average_cer": (sum(cer_values) / len(cer_values)) if cer_values else None,
        "elapsed_seconds": elapsed_seconds,
        "average_transcript_ratio": (sum(transcript_ratio_values) / len(transcript_ratio_values)) if transcript_ratio_values else None,
        "items": validated_items,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"Matched items: {matched_count}/{len(source_items)}")
    print(f"Average WER: {payload['average_wer']}")
    print(f"Average CER: {payload['average_cer']}")
    print(f"Elapsed seconds: {payload['elapsed_seconds']}")
    print(f"Average transcript ratio: {payload['average_transcript_ratio']}")


def build_summary(validated_dir: Path) -> None:
    model_runs: dict[str, list[dict]] = {}

    for validated_json in sorted(validated_dir.glob("*_validated.json")):
        payload = json.loads(validated_json.read_text(encoding="utf-8"))
        model = payload.get("model")
        if not model:
            continue
        model_runs.setdefault(model, []).append({
            "source_json": payload.get("source_json"),
            "created_at": payload.get("created_at"),
            "total_items": payload.get("total_items"),
            "matched_items": payload.get("matched_items"),
            "average_wer": payload.get("average_wer"),
            "average_cer": payload.get("average_cer"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "average_transcript_ratio": payload.get("average_transcript_ratio"),
        })

    summary: list[dict] = []
    for model, runs in sorted(model_runs.items()):
        wer_vals = [r["average_wer"] for r in runs if r["average_wer"] is not None]
        cer_vals = [r["average_cer"] for r in runs if r["average_cer"] is not None]
        ratio_vals = [r["average_transcript_ratio"] for r in runs if r["average_transcript_ratio"] is not None]
        summary.append({
            "model": model,
            "test_run_count": len(runs),
            "average_wer": (sum(wer_vals) / len(wer_vals)) if wer_vals else None,
            "average_cer": (sum(cer_vals) / len(cer_vals)) if cer_vals else None,
            "average_transcript_ratio": (sum(ratio_vals) / len(ratio_vals)) if ratio_vals else None,
        })

    summary_path = validated_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved: {summary_path} ({len(summary)} models)")


def main() -> None:
    if not VALIDATED_TSV.exists():
        print(f"Missing validated TSV: {VALIDATED_TSV}")
        return

    validated_dir = OUTPUT_DIR / "validated"
    validated_dir.mkdir(exist_ok=True)

    references = build_reference_map(VALIDATED_TSV)

    if len(sys.argv) > 1:
        source_jsons = [Path(sys.argv[1]).resolve()]
    else:
        source_jsons = get_all_transcription_jsons(OUTPUT_DIR)

    for source_json in source_jsons:
        validate_json(source_json, references, validated_dir)

    build_summary(validated_dir)


if __name__ == "__main__":
    main()
