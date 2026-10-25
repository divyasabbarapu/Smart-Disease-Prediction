import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "datasets", "indian_liver_patient.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models", "liver_disease"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# Input features
FEATURES = [
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

TARGET = "Dataset"


def main():

    # Load dataset
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    # Select required columns
    data = data[FEATURES + [TARGET]].copy()

    # Clean Gender values
    data["Gender"] = (
        data["Gender"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # Remove duplicate records
    data = data.drop_duplicates()

    # Convert target:
    # 1 = Liver Disease
    # 2 = No Liver Disease
    data[TARGET] = data[TARGET].map({
        1: 1,
        2: 0
    })

    data = data.dropna(subset=[TARGET])

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    # Separate numerical and categorical columns
    numerical_features = [
        feature for feature in FEATURES
        if feature != "Gender"
    ]

    categorical_features = ["Gender"]

    # Numerical preprocessing
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessing = ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # SVM Model
    svm_model = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])

    # KNN Model
    knn_model = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])

    # Random Forest Model
    rf_model = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Train individual models
    models = {
        "SVM": svm_model,
        "KNN": knn_model,
        "Random Forest": rf_model
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    # Ensemble model using soft voting
    ensemble_model = VotingClassifier(
        estimators=[
            ("svm", svm_model),
            ("knn", knn_model),
            ("random_forest", rf_model)
        ],
        voting="soft"
    )

    ensemble_model.fit(X_train, y_train)

    models["Ensemble"] = ensemble_model

    # Evaluate and save models
    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # Save trained model
        filename = (
            "liver_" +
            name.lower().replace(" ", "_") +
            ".pkl"
        )

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, filename)
        )

        print(f"\n{name} Classification Report")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

    # Save feature names
    joblib.dump(
        FEATURES,
        os.path.join(MODEL_DIR, "liver_features.pkl")
    )

    # Save model comparison results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(MODEL_DIR, "liver_model_results.csv"),
        index=False
    )

    print("\n========== LIVER DISEASE MODEL RESULTS ==========")

    print(results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2%}".format,
            "Precision": "{:.2%}".format,
            "Recall": "{:.2%}".format,
            "F1 Score": "{:.2%}".format
        }
    ))

    print("\nAll Liver Disease models trained and saved successfully!")


if __name__ == "__main__":
    main()