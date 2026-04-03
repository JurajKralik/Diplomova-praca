from src.models import transcript, Model


FILE_PATH = "../data/sample_short.wav"

def main():
    text = transcript(FILE_PATH, Model.SPEECH_RECOGNITION_SPHINX)
    print(f"Result: {text}\n")
    
    text = transcript(FILE_PATH, Model.WHISPER_TINY)
    print(f"Result: {text}\n")
    

if __name__ == "__main__":
    main()