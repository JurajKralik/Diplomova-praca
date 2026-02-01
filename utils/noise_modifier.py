from pydub import AudioSegment
import numpy as np
from tkinter import filedialog
import os

def add_noise(input_folder: str, output_folder: str, noise_level_db=-30):
	# Loop through all WAV files
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            original_path = os.path.join(input_folder, filename)
            modified_path = os.path.join(output_folder, filename)

            sound = AudioSegment.from_wav(original_path)

            duration_ms = len(sound)
            noise = generate_white_noise(duration_ms, sample_rate=sound.frame_rate)
            # Match channels
            if sound.channels != noise.channels:
                noise = noise.set_channels(sound.channels)

            # Apply gain to noise
            noise = noise - abs(noise_level_db)
            modified_sound = sound.overlay(noise)

            modified_sound.export(modified_path, format="wav")
            print(f"Saved modified file: {modified_path}")

def generate_white_noise(duration_ms, sample_rate=44100, amplitude=0.1):
	# Generate white noise using numpy
	samples = np.random.normal(0, amplitude, int(sample_rate * duration_ms / 1000.0))
	samples = np.clip(samples, -1.0, 1.0)
	samples_int16 = (samples * 32767).astype(np.int16)

	# Convert to raw audio
	raw_noise = samples_int16.tobytes()
	return AudioSegment(
		data=raw_noise,
		sample_width=2,  # 16-bit audio
		frame_rate=sample_rate,
		channels=1
	)

input_path = filedialog.askdirectory()
if not input_path:
    print("No input directory selected.")
    exit()
output_path = filedialog.askdirectory()
if not output_path:
    print("No output directory selected.")
    exit()
add_noise(input_path, output_path, noise_level_db=-10)
