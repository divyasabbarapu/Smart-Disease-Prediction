import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# Kidney Disease input features
KIDNEY_FEATURES = [
    "age", "bp", "sg", "al", "su",
    "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane"
]


# Liver Disease input features
LIVER_FEATURES = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio"
]


def load_model(disease, model_name):

    disease_key = disease.lower().replace(" ", "_")
    model_key = model_name.lower().replace(" ", "_")

    # Kidney and Liver models are stored in separate folders
    if disease_key in ["kidney_disease", "liver_disease"]:

        folder = os.path.join(MODEL_DIR, disease_key)

        prefix = (
            "kidney"
            if disease_key == "kidney_disease"
            else "liver"
        )

        model_path = os.path.join(
            folder,
            f"{prefix}_{model_key}.pkl"
        )

    else:
        # Existing Heart Disease and Diabetes model locations
        model_path = os.path.join(
            MODEL_DIR,
            f"{disease_key}_{model_key}.pkl"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


def load_features(disease):

    disease_key = disease.lower().replace(" ", "_")

    if disease_key == "kidney_disease":
        return KIDNEY_FEATURES

    elif disease_key == "liver_disease":
        return LIVER_FEATURES

    else:
        feature_path = os.path.join(
            MODEL_DIR,
            f"{disease_key}_features.pkl"
        )

        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )

        return joblib.load(feature_path)


def predict_disease(disease, input_data):

    # Load all four models
    svm_model = load_model(disease, "SVM")
    knn_model = load_model(disease, "KNN")
    rf_model = load_model(disease, "Random Forest")
    ensemble_model = load_model(disease, "Ensemble")

    # Get feature order
    features = load_features(disease)

    # Convert input into DataFrame
    input_df = pd.DataFrame(
        [[input_data[feature] for feature in features]],
        columns=features
    )

    # Individual predictions
    svm_prediction = int(svm_model.predict(input_df)[0])
    knn_prediction = int(knn_model.predict(input_df)[0])
    rf_prediction = int(rf_model.predict(input_df)[0])
    ensemble_prediction = int(
        ensemble_model.predict(input_df)[0]
    )

    # Majority voting using SVM, KNN and Random Forest
    predictions = [
        svm_prediction,
        knn_prediction,
        rf_prediction
    ]

    disease_votes = sum(predictions)
    healthy_votes = len(predictions) - disease_votes

    if disease_votes >= 2:
        final_result = 1
        agreement = disease_votes
    else:
        final_result = 0
        agreement = healthy_votes

    return {
        "SVM": svm_prediction,
        "KNN": knn_prediction,
        "Random Forest": rf_prediction,
        "Ensemble": ensemble_prediction,
        "Agreement": agreement,
        "Final": final_result
    }