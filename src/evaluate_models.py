import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    confusion_matrix
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)


# =========================================================
# EVALUATE DISEASE
# =========================================================

def evaluate_disease(
    dataset_file,
    disease_name,
    target_column
):

    print("\n" + "=" * 70)
    print(f"EVALUATING {disease_name.upper()}")
    print("=" * 70)

    dataset_path = os.path.join(
        DATASET_DIR,
        dataset_file
    )

    df = pd.read_csv(dataset_path)

    # =====================================================
    # CLEAN DATA
    # =====================================================

    df = df.replace("?", pd.NA)

    for column in df.columns:

        if column != target_column:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    df = df.dropna(
        subset=[target_column]
    )

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:

        if df[column].isnull().sum() > 0:

            df[column] = df[column].fillna(
                df[column].median()
            )

    # =====================================================
    # PREPARE X AND Y
    # =====================================================

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # Convert Heart Disease target to binary
    if disease_name == "Heart Disease":

        y = y.apply(
            lambda value: 1 if value > 0 else 0
        )

    # =====================================================
    # CLASS DISTRIBUTION GRAPH
    # =====================================================

    class_counts = y.value_counts().sort_index()

    class_labels = []

    if disease_name == "Heart Disease":

        class_labels = [
            "No Heart Disease",
            "Heart Disease"
        ]

    else:

        class_labels = [
            "No Diabetes",
            "Diabetes"
        ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        class_labels,
        class_counts.values
    )

    plt.title(
        f"{disease_name} Class Distribution"
    )

    plt.xlabel("Class")

    plt.ylabel("Number of Records")

    plt.xticks(rotation=0)

    plt.tight_layout()

    class_graph_file = (
        disease_name.lower()
        .replace(" ", "_")
        + "_class_distribution.png"
    )

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            class_graph_file
        ),
        dpi=300
    )

    plt.close()

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # =====================================================
    # MODELS
    # =====================================================

    svm_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
    ])

    knn_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ])

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

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
    # MODEL EVALUATION
    # =====================================================

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        precision = precision_score(
            y_test,
            prediction,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            prediction,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            prediction,
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # =================================================
        # CONFUSION MATRIX
        # =================================================

        cm = confusion_matrix(
            y_test,
            prediction
        )

        plt.figure(
            figsize=(6, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "No Disease",
                "Disease"
            ],
            yticklabels=[
                "No Disease",
                "Disease"
            ]
        )

        plt.title(
            f"{disease_name} - {name} Confusion Matrix"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        filename = (
            disease_name.lower()
            .replace(" ", "_")
            + "_"
            + name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                REPORT_DIR,
                filename
            ),
            dpi=300
        )

        plt.close()

    # =====================================================
    # RESULTS TABLE
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print(
        results_df.to_string(
            index=False
        )
    )

    # =====================================================
    # SAVE RESULTS TO CSV
    # =====================================================

    result_file = (
        disease_name.lower()
        .replace(" ", "_")
        + "_evaluation.csv"
    )

    results_df.to_csv(
        os.path.join(
            REPORT_DIR,
            result_file
        ),
        index=False
    )

    # =====================================================
    # MODEL PERFORMANCE GRAPH
    # =====================================================

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    results_plot = results_df.set_index(
        "Model"
    )[metrics]

    ax = results_plot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        f"{disease_name} Model Performance"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    graph_file = (
        disease_name.lower()
        .replace(" ", "_")
        + "_model_performance.png"
    )

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            graph_file
        ),
        dpi=300
    )

    plt.close()

    # =====================================================
    # ACCURACY COMPARISON GRAPH
    # =====================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    plt.title(
        f"{disease_name} Accuracy Comparison"
    )

    plt.xlabel(
        "Machine Learning Model"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    accuracy_graph_file = (
        disease_name.lower()
        .replace(" ", "_")
        + "_accuracy_comparison.png"
    )

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            accuracy_graph_file
        ),
        dpi=300
    )

    plt.close()

    print(
        f"\nGraphs saved for {disease_name}."
    )

    return results_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    evaluate_disease(
        "heart_disease.csv",
        "Heart Disease",
        "target"
    )

    evaluate_disease(
        "diabetes.csv",
        "Diabetes",
        "Outcome"
    )

    print("\n" + "=" * 70)
    print("EVALUATION AND GRAPH GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 70)