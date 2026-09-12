import os

import opensmile

from firebase_service import download_audio_from_firebase


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def extract_egemaps_from_firebase(storage_path):
    temp_audio_path = None

    try:
        temp_audio_path = download_audio_from_firebase(storage_path)

        features_df = smile.process_file(temp_audio_path)

        row = features_df.iloc[0]

        return row.to_dict()

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


if __name__ == "__main__":
    print("Firebase eGeMAPS extractor is ready ")