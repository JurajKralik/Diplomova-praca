from pathlib import Path
from tkinter import filedialog

from pydub import AudioSegment


def convert_mp3_to_wav(input_folder: Path, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(input_folder.iterdir())
    total = len(files)
    current = 0

    for file_path in files:
        current += 1
        if file_path.suffix.lower() != ".mp3":
            continue

        wav_path = output_folder / f"{file_path.stem}.wav"
        audio = AudioSegment.from_mp3(file_path)
        audio.export(wav_path, format="wav")
        print(f"({current}/{total}) Converted: {file_path.name} -> {wav_path.name}")


def main() -> None:
    input_path = filedialog.askdirectory(title="Select input folder with MP3 files")
    if not input_path:
        print("No input directory selected.")
        return

    output_path = filedialog.askdirectory(title="Select output folder for WAV files")
    if not output_path:
        print("No output directory selected.")
        return

    convert_mp3_to_wav(Path(input_path), Path(output_path))


if __name__ == "__main__":
    main()
