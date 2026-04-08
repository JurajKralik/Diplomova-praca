from pathlib import Path
from tkinter import filedialog

from pydub import AudioSegment


def modify_speed(input_folder: Path, output_folder: Path, speed: float = 0.9) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(input_folder.iterdir()):
        if file_path.suffix.lower() != '.wav':
            continue

        modified_path = output_folder / file_path.name
        sound = AudioSegment.from_wav(file_path)
        modified_sound = sound._spawn(
            sound.raw_data,
            overrides={"frame_rate": int(sound.frame_rate * speed)},
        ).set_frame_rate(sound.frame_rate)

        modified_sound.export(modified_path, format='wav')
        print(f"Saved modified file: {modified_path}")


def main() -> None:
    input_path = filedialog.askdirectory(title='Select input folder with WAV files')
    if not input_path:
        print('No input directory selected.')
        return

    output_path = filedialog.askdirectory(title='Select output folder for speed-modified WAV files')
    if not output_path:
        print('No output directory selected.')
        return

    modify_speed(Path(input_path), Path(output_path), speed=0.9)


if __name__ == '__main__':
    main()
