from pydub import AudioSegment
from tkinter import filedialog
import os

def modify_speed(input_folder: str, output_folder: str, speed: float = 0.5):
	# Loop through all WAV files
	for filename in os.listdir(input_folder):
		if filename.endswith('.wav'):
			original_path = os.path.join(input_folder, filename)
			modified_path = os.path.join(output_folder, filename)

			sound = AudioSegment.from_wav(original_path)

			# Modify speed
			modified_sound = sound._spawn(sound.raw_data, overrides={
				"frame_rate": int(sound.frame_rate * speed)
			}).set_frame_rate(sound.frame_rate)  # Keep original frame rate to save as valid WAV

			modified_sound.export(modified_path, format="wav")
			print(f"Saved modified file: {modified_path}")

input_path = filedialog.askdirectory()
if not input_path:
    print("No input directory selected.")
    exit()
output_path = filedialog.askdirectory()
if not output_path:
    print("No output directory selected.")
    exit()
modify_speed(input_path, output_path, speed=0.9)
