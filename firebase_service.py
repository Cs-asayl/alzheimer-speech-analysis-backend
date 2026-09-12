import os
import tempfile

import firebase_admin
from firebase_admin import credentials, storage


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVICE_ACCOUNT_PATH = os.path.join(
    BASE_DIR,
    "serviceAccountKey.json",
)

BUCKET_NAME = "alzheimer-mobile-app.firebasestorage.app"


def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(
            SERVICE_ACCOUNT_PATH
        )

        firebase_admin.initialize_app(
            cred,
            {
                "storageBucket": BUCKET_NAME
            },
        )


def download_audio_from_firebase(storage_path):
    initialize_firebase()

    bucket = storage.bucket()

    blob = bucket.blob(storage_path)

    if not blob.exists():
        raise FileNotFoundError(
            f"Audio file not found: {storage_path}"
        )

    suffix = os.path.splitext(storage_path)[1]

    if not suffix:
        suffix = ".m4a"

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temp_path = temp_file.name
    temp_file.close()

    blob.download_to_filename(temp_path)

    return temp_path


if __name__ == "__main__":
    initialize_firebase()

    print("Firebase Admin connected successfully ✅")
    print("Bucket:", storage.bucket().name)