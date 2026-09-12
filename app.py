from flask import Flask, request, jsonify

from feature_extractor import extract_egemaps_from_firebase
from linguistic_extractor import extract_linguistic_features
from model_predictor import predict_alzheimer


app = Flask(__name__)


# -------------------------------------------------
# HOME
# -------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Alzheimer Speech Analysis API is running"
    })


# -------------------------------------------------
# ANALYZE AUDIO FROM FIREBASE STORAGE
# -------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze_audio():

    try:

        # Flutter sends:
        #
        # {
        #   "storagePath": "users/.../audio/..."
        # }

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "error": "No JSON data received"
            }), 400


        storage_path = data.get(
            "storagePath"
        )


        if not storage_path:

            return jsonify({
                "error": "storagePath is required"
            }), 400


        print("\n--------------------------------")
        print("NEW ANALYSIS REQUEST")
        print("--------------------------------")

        print("Storage path:")
        print(storage_path)

        print("--------------------------------")


        # =================================================
        # STEP 1
        # EXTRACT 88 eGeMAPS FEATURES FROM FIREBASE AUDIO
        # =================================================

        egemaps_features = (
            extract_egemaps_from_firebase(
                storage_path
            )
        )


        egemaps_count = len(
            egemaps_features
        )


        print(
            "eGeMAPS features extracted:",
            egemaps_count
        )


        if egemaps_count != 88:

            return jsonify({
                "error":
                    "Unexpected number of eGeMAPS features",

                "featureCount":
                    egemaps_count

            }), 500


        print(
            "88 eGeMAPS features "
            "extracted successfully ✅"
        )


        # =================================================
        # STEP 2
        # TEMPORARY MANUAL TRANSCRIPT
        # =================================================
        #
        # IMPORTANT:
        #
        # Whisper is not reliable yet with the emulator.
        #
        # So for now we use a correct manual transcript
        # ONLY to test the complete AI pipeline.
        #
        # Later:
        #
        # Audio
        #   ↓
        # Speech-to-Text
        #   ↓
        # transcript
        #
        # will replace this manual transcript.
        # =================================================

        manual_transcript = (
            "The picture shows a family in a kitchen. "
            "The mother is washing dishes while the "
            "children are taking cookies from a jar. "
            "The water is overflowing from the sink."
        )


        print("\n--------------------------------")
        print("TRANSCRIPT USED FOR TEST")
        print("--------------------------------")

        print(
            manual_transcript
        )


        # =================================================
        # STEP 3
        # EXTRACT 19 LINGUISTIC FEATURES
        # =================================================

        linguistic_features = (
            extract_linguistic_features(
                manual_transcript
            )
        )


        linguistic_count = len(
            linguistic_features
        )


        print(
            "\nLinguistic features extracted:",
            linguistic_count
        )


        if linguistic_count != 19:

            return jsonify({
                "error":
                    "Unexpected number of linguistic features",

                "featureCount":
                    linguistic_count

            }), 500


        print(
            "19 linguistic features "
            "extracted successfully ✅"
        )


        # =================================================
        # STEP 4
        #
        # 88 eGeMAPS
        # +
        # 19 Linguistic
        # =
        # 107 Features
        #
        # Then:
        #
        # StandardScaler
        #       ↓
        # GA Selection
        # 107 → 56
        #       ↓
        # Logistic Regression
        # =================================================

        print("\n--------------------------------")
        print("RUNNING AI MODEL")
        print("--------------------------------")


        prediction_result = (
            predict_alzheimer(
                egemaps_features,
                linguistic_features
            )
        )


        print(
            "\nPrediction result:"
        )

        print(
            prediction_result
        )


        # =================================================
        # STEP 5
        # EXTRACT RESULT
        # =================================================

        prediction = (
            prediction_result[
                "prediction"
            ]
        )


        probabilities = (
            prediction_result[
                "probabilities"
            ]
        )


        selected_feature_count = (
            prediction_result[
                "selectedFeatureCount"
            ]
        )


        healthy_probability = float(
            probabilities.get(
                "healthy",
                0.0
            )
        )


        patient_probability = float(
            probabilities.get(
                "patient",
                0.0
            )
        )


        # =================================================
        # STEP 6
        # CONVERT MODEL OUTPUT TO APP RESULT
        # =================================================

        if prediction == "patient":

            risk = "high"

            score = patient_probability

            message = (
                "The speech analysis model detected "
                "patterns associated with the patient "
                "class. This result is intended for "
                "screening purposes only and is not "
                "a medical diagnosis."
            )

        else:

            risk = "low"

            score = patient_probability

            message = (
                "The speech analysis model classified "
                "the sample as healthy. This result is "
                "intended for screening purposes only "
                "and is not a medical diagnosis."
            )


        print("\n--------------------------------")
        print("FINAL RESULT")
        print("--------------------------------")

        print(
            "Prediction:",
            prediction
        )

        print(
            "Healthy probability:",
            healthy_probability
        )

        print(
            "Patient probability:",
            patient_probability
        )

        print(
            "Risk:",
            risk
        )

        print("--------------------------------")


        # =================================================
        # RETURN RESULT TO FLUTTER
        # =================================================

        return jsonify({

            "status":
                "success",

            "storagePath":
                storage_path,

            "egemapsFeatureCount":
                egemaps_count,

            "linguisticFeatureCount":
                linguistic_count,

            "combinedFeatureCount":
                egemaps_count
                + linguistic_count,

            "selectedFeatureCount":
                selected_feature_count,

            "prediction":
                prediction,

            "probabilities":
                probabilities,

            "healthyProbability":
                healthy_probability,

            "patientProbability":
                patient_probability,

            # Keep these fields temporarily
            # because Flutter currently expects them.

            "risk":
                risk,

            "score":
                score,

            "message":
                message,

            # Important so we remember
            # this prediction used manual text.

            "transcriptMode":
                "manual_test"

        }), 200


    # -------------------------------------------------
    # FIREBASE FILE NOT FOUND
    # -------------------------------------------------

    except FileNotFoundError as e:

        print(
            "Firebase file error:",
            str(e)
        )

        return jsonify({

            "error":
                "Audio file not found in Firebase",

            "details":
                str(e)

        }), 404


    # -------------------------------------------------
    # GENERAL ERROR
    # -------------------------------------------------

    except Exception as e:

        print(
            "Analysis error:",
            repr(e)
        )

        return jsonify({

            "error":
                "Audio analysis failed",

            "details":
                str(e)

        }), 500


# -------------------------------------------------
# START SERVER
# -------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )