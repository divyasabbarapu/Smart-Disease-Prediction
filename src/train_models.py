import os
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# =========================================================
# GENERAL MODEL FUNCTION
# =========================================================

def train_disease_models(
    dataset_path,
    disease_name,
    target_column,
    positive_values=None
):
    print("\n" + "=" * 80)
    print(f"TRAINING MODELS FOR: {disease_name.upper()}")
    print("=" * 80)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    df = pd.read_csv(dataset_path)

    print(f"\nDataset shape: {df.shape}")

    # -----------------------------------------------------
    # Remove duplicate rows
    # -----------------------------------------------------

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate rows...")
        df = df.drop_duplicates()

    # -----------------------------------------------------
    # Handle missing values
    # -----------------------------------------------------

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Replace '?' with NaN
    df = df.replace("?", pd.NA)

    # Convert possible numeric columns
    for column in df.columns:
        if column != target_column:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows where target is missing
    df = df.dropna(subset=[target_column])

    # Fill numeric missing values with median
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].median())

    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -----------------------------------------------------
    # Convert target to binary classification
    # -----------------------------------------------------

    if positive_values is not None:
        y = y.apply(
            lambda value: 1 if value in positive_values else 0
        )

    y = pd.to_numeric(y, errors="coerce")

    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    print("\nTarget distribution:")
    print(y.value_counts())

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # =====================================================
    # SVM
    # =====================================================

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
    ])

    # =====================================================
    # KNN
    # =====================================================

    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=5)
        )
    ])

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    # =====================================================
    # ENSEMBLE
    # =====================================================

    ensemble_model = VotingClassifier(
        estimators=[
            ("svm", svm_model),
            ("knn", knn_model),
            ("random_forest", random_forest_model)
        ],
        voting="soft"
    )

    models = {
        "SVM": svm_model,
        "KNN": knn_model,
        "Random Forest": random_forest_model,
        "Ensemble": ensemble_model
    }

    results = []

    # =====================================================
    # TRAIN AND EVALUATE
    # =====================================================

    for model_name, model in models.items():

        print("\n" + "-" * 60)
        print(f"Training {model_name}...")
        print("-" * 60)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        matrix = confusion_matrix(
            y_test,
            predictions
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nConfusion Matrix:")
        print(matrix)

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        results.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4)
        })

        # -------------------------------------------------
        # Save model
        # -------------------------------------------------

        model_filename = (
            disease_name.lower().replace(" ", "_")
            + "_"
            + model_name.lower().replace(" ", "_")
            + ".pkl"
        )

        model_path = os.path.join(
            MODEL_DIR,
            model_filename
        )

        joblib.dump(model, model_path)

        print(f"Saved model: {model_path}")

    # =====================================================
    # SAVE FEATURE INFORMATION
    # =====================================================

    feature_filename = (
        disease_name.lower().replace(" ", "_")
        + "_features.pkl"
    )

    feature_path = os.path.join(
        MODEL_DIR,
        feature_filename
    )

    joblib.dump(
        list(X.columns),
        feature_path
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    report_filename = (
        disease_name.lower().replace(" ", "_")
        + "_model_results.csv"
    )

    report_path = os.path.join(
        REPORT_DIR,
        report_filename
    )

    results_df.to_csv(
        report_path,
        index=False
    )

    print("\n" + "=" * 80)
    print(f"{disease_name.upper()} MODEL RESULTS")
    print("=" * 80)

    print(results_df.to_string(index=False))

    print(f"\nResults saved to:")
    print(report_path)

    return results_df


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # HEART DISEASE
    # -----------------------------------------------------

    heart_dataset = os.path.join(
        DATASET_DIR,
        "heart_disease.csv"
    )

    if os.path.exists(heart_dataset):

        train_disease_models(
            dataset_path=heart_dataset,
            disease_name="Heart Disease",
            target_column="target",
            positive_values=None
        )

    else:

        print(
            "\nHeart disease dataset not found."
        )
        print(
            f"Expected location: {heart_dataset}"
        )

    # -----------------------------------------------------
    # DIABETES
    # -----------------------------------------------------

    diabetes_dataset = os.path.join(
        DATASET_DIR,
        "diabetes.csv"
    )

    if os.path.exists(diabetes_dataset):

        train_disease_models(
            dataset_path=diabetes_dataset,
            disease_name="Diabetes",
            target_column="Outcome",
            positive_values=None
        )

    else:

        print(
            "\nDiabetes dataset not found."
        )
        print(
            f"Expected location: {diabetes_dataset}"
        )