from models import transcript, Model


file_path = "data/sample.wav"

def main():
    text = transcript(file_path, Model.WHISPER_MEDIUM)
    print(text)
    text = transcript(file_path, Model.FASTER_WHISPER_MEDIUM)
    print(text)

if __name__ == "__main__":
    main()