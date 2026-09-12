import os
import sys
import shutil
import whisper

from firebase_service import download_audio_from_firebase


# Load Whisper model
model = whisper.load_model("small.en")


def transcribe_from_firebase(storage_path):
    temp_audio_path = None

    try:
        print("Downloading audio from Firebase...")

        temp_audio_path = download_audio_from_firebase(
            storage_path
        )

        print("Audio downloaded successfully.")

        # Save a copy so we can inspect the audio
        debug_path = r"C:\alzheimer_backend\debug_audio.m4a"

        shutil.copy(
            temp_audio_path,
            debug_path
        )

        print("Debug audio saved to:")
        print(debug_path)

        print(
            "File size:",
            os.path.getsize(debug_path),
            "bytes"
        )

        print("Starting transcription...")

        result = model.transcribe(
            temp_audio_path,
            language="en",
            fp16=False,
        )

        transcript = result["text"].strip()

        return transcript

    finally:
        # Delete only Firebase temporary file
        # debug_audio.m4a will remain
        if (
            temp_audio_path
            and os.path.exists(temp_audio_path)
        ):
            os.remove(temp_audio_path)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python speech_to_text.py "
            "\"Firebase storage path\""
        )
        sys.exit(1)

    storage_path = sys.argv[1]

    print("\n------------------------------")
    print("Speech-to-Text Test")
    print("------------------------------")

    transcript = transcribe_from_firebase(
        storage_path
    )

    print("\n------------------------------")
    print("TRANSCRIPT:")
    print("------------------------------")
    print(transcript)

    print(
        "\nSpeech-to-Text completed successfully "
    )