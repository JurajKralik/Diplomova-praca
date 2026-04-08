import os
from pathlib import Path
from tkinter import filedialog

import numpy as np
from pydub import AudioSegment


def add_noise(input_folder: Path, output_folder: Path, noise_level_db: int = -10) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    for filename in sorted(os.listdir(input_folder)):
        if not filename.endswith('.wav'):
            continue

        original_path = input_folder / filename
        modified_path = output_folder / filename

        sound = AudioSegment.from_wav(original_path)
        duration_ms = len(sound)
        noise = generate_white_noise(duration_ms, sample_rate=sound.frame_rate)

        if sound.channels != noise.channels:
            noise = noise.set_channels(sound.channels)

        noise = noise - abs(noise_level_db)
        modified_sound = sound.overlay(noise)
        modified_sound.export(modified_path, format='wav')
        print(f"Saved modified file: {modified_path}")


def generate_white_noise(duration_ms: int, sample_rate: int = 44100, amplitude: float = 0.1) -> AudioSegment:
    samples = np.random.normal(0, amplitude, int(sample_rate * duration_ms / 1000.0))
    samples = np.clip(samples, -1.0, 1.0)
    samples_int16 = (samples * 32767).astype(np.int16)
    raw_noise = samples_int16.tobytes()
    return AudioSegment(data=raw_noise, sample_width=2, frame_rate=sample_rate, channels=1)


def main() -> None:
    input_path = filedialog.askdirectory(title='Select input folder with WAV files')
    if not input_path:
        print('No input directory selected.')
        return

    output_path = filedialog.askdirectory(title='Select output folder for noisy WAV files')
    if not output_path:
        print('No output directory selected.')
        return

    add_noise(Path(input_path), Path(output_path), noise_level_db=-10)


if __name__ == '__main__':
    main()
