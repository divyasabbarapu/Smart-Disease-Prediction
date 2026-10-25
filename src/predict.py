import os
import joblib
import pandas as pd


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(disease, model_name):

    filename = (
        disease.lower().replace(" ", "_")
        + "_"
        + model_name.lower().replace(" ", "_")
        + ".pkl"
    )

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    return joblib.load(path)


# =========================================================
# LOAD FEATURES
# =========================================================

def load_features(disease):

    filename = (
        disease.lower().replace(" ", "_")
        + "_features.pkl"
    )

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    return joblib.load(path)


# =========================================================
# MAKE PREDICTION
# =========================================================

def predict_disease(disease, input_data):

    svm_model = load_model(
        disease,
        "SVM"
    )

    knn_model = load_model(
        disease,
        "KNN"
    )

    rf_model = load_model(
        disease,
        "Random Forest"
    )

    ensemble_model = load_model(
        disease,
        "Ensemble"
    )

    features = load_features(
        disease
    )

    # Make sure input follows training order
    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    # Individual predictions
    svm_prediction = int(
        svm_model.predict(input_df)[0]
    )

    knn_prediction = int(
        knn_model.predict(input_df)[0]
    )

    rf_prediction = int(
        rf_model.predict(input_df)[0]
    )

    ensemble_prediction = int(
        ensemble_model.predict(input_df)[0]
    )

    # -----------------------------------------------------
    # Agreement
    # -----------------------------------------------------

    predictions = [
        svm_prediction,
        knn_prediction,
        rf_prediction
    ]

    disease_votes = sum(predictions)

    healthy_votes = 3 - disease_votes

    if disease_votes > healthy_votes:
        agreement = disease_votes
        final_result = 1
    else:
        agreement = healthy_votes
        final_result = 0

    return {
        "SVM": svm_prediction,
        "KNN": knn_prediction,
        "Random Forest": rf_prediction,
        "Ensemble": ensemble_prediction,
        "Agreement": agreement,
        "Final": final_result
    }