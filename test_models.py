import joblib
import json

model = joblib.load("models/GA_LogisticRegression.pkl")
scaler = joblib.load("models/eGeMAPS_Linguistic_Scaler.pkl")
indices = joblib.load("models/GA_Selected_Indices.pkl")

with open(
    "models/GA_Selected_Features.json",
    "r",
    encoding="utf-8",
) as f:
    features = json.load(f)

print("Model loaded:", type(model))
print("Scaler loaded:", type(scaler))

print("\n--- SHAPES ---")
print("Model n_features_in_:", model.n_features_in_)
print("Scaler n_features_in_:", scaler.n_features_in_)

print("\nSelected indices:", len(indices))
print("Selected features:", len(features))

print("\nModel classes:", model.classes_)

if hasattr(scaler, "feature_names_in_"):
    print("\nScaler feature names:")
    print(list(scaler.feature_names_in_))

if hasattr(model, "feature_names_in_"):
    print("\nModel feature names:")
    print(list(model.feature_names_in_))

print("\nSelected feature list:")
for i, feature in enumerate(features, start=1):
    print(f"{i}. {feature}")
selected_from_indices = [
    scaler.feature_names_in_[i]
    for i in indices
]

print("\n--- GA CHECK ---")

print(
    "Indices and JSON features match:",
    selected_from_indices == features
)

if selected_from_indices != features:
    print("\nMismatches:")

    for i, (a, b) in enumerate(
        zip(selected_from_indices, features),
        start=1,
    ):
        if a != b:
            print(
                f"{i}: index gives '{a}' "
                f"but JSON gives '{b}'"
            )
else:
    print("Perfect match ✅")