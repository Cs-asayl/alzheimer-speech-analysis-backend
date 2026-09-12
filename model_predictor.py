import os
import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


SCALER_PATH = os.path.join(
    MODELS_DIR,
    "eGeMAPS_Linguistic_Scaler.pkl"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "GA_LogisticRegression.pkl"
)

INDICES_PATH = os.path.join(
    MODELS_DIR,
    "GA_Selected_Indices.pkl"
)


# -------------------------------
# LOAD SAVED OBJECTS
# -------------------------------

scaler = joblib.load(
    SCALER_PATH
)

model = joblib.load(
    MODEL_PATH
)

selected_indices = joblib.load(
    INDICES_PATH
)


print(
    "Scaler loaded:",
    scaler.n_features_in_
)

print(
    "Model loaded:",
    model.n_features_in_
)

print(
    "GA selected:",
    len(selected_indices)
)


# -------------------------------
# PREDICTION FUNCTION
# -------------------------------

def predict_alzheimer(
    egemaps_features,
    linguistic_features
):

    # Convert acoustic features to dictionary
    if hasattr(
        egemaps_features,
        "to_dict"
    ):
        acoustic_dict = (
            egemaps_features.to_dict()
        )

    elif isinstance(
        egemaps_features,
        dict
    ):
        acoustic_dict = (
            egemaps_features
        )

    else:
        raise TypeError(
            "eGeMAPS features must contain "
            "feature names."
        )

    # Combine 88 + 19
    combined = {}

    combined.update(
        acoustic_dict
    )

    combined.update(
        linguistic_features
    )

    print(
        "Combined features:",
        len(combined)
    )

    # Use EXACT feature order
    # used when scaler was trained
    feature_names = list(
        scaler.feature_names_in_
    )

    missing = [
        feature
        for feature in feature_names
        if feature not in combined
    ]

    if missing:
        raise ValueError(
            "Missing features:\n"
            + "\n".join(missing)
        )

    # Create input in correct order
    ordered_values = [
        combined[name]
        for name in feature_names
    ]

    X = pd.DataFrame(
        [ordered_values],
        columns=feature_names
    )

    print(
        "Input before scaler:",
        X.shape
    )

    # 107 → standardized 107
    X_scaled = scaler.transform(
        X
    )

    print(
        "After scaler:",
        X_scaled.shape
    )

    # 107 → GA selected 56
    X_selected = X_scaled[
        :,
        selected_indices
    ]

    print(
        "After GA:",
        X_selected.shape
    )

    # Prediction
    prediction = model.predict(
        X_selected
    )[0]

    probabilities = (
        model.predict_proba(
            X_selected
        )[0]
    )

    classes = list(
        model.classes_
    )

    probability_dict = {
        str(label): float(prob)
        for label, prob
        in zip(
            classes,
            probabilities
        )
    }

    return {
        "prediction": str(
            prediction
        ),

        "probabilities":
            probability_dict,

        "selectedFeatureCount":
            int(
                X_selected.shape[1]
            )
    }


if __name__ == "__main__":

    print("\n----------------------")
    print("MODEL PIPELINE CHECK")
    print("----------------------")

    print(
        "Scaler features:",
        scaler.n_features_in_
    )

    print(
        "GA features:",
        len(selected_indices)
    )

    print(
        "Model features:",
        model.n_features_in_
    )

    print(
        "Classes:",
        model.classes_
    )