
import os
import joblib
import pandas as pd

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
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR, "datasets", "kidney_disease.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models", "kidney_disease"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load and clean dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

# Remove unnecessary ID column
df = df.drop(columns=["id"], errors="ignore")

# Remove extra spaces and tabs from text values
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Convert target labels into numerical values
df["classification"] = df["classification"].str.lower()

df["classification"] = df["classification"].map({
    "ckd": 1,
    "notckd": 0
})

# Remove rows with invalid target values
df = df.dropna(subset=["classification"])

# Convert target to integer
df["classification"] = df["classification"].astype(int)

# Convert numeric-looking columns into numeric values.
# Invalid entries become missing values and are imputed later.
for column in df.columns:
    if column != "classification" and df[column].dtype == "object":
        converted = pd.to_numeric(df[column], errors="coerce")

        if converted.notna().sum() > 0:
            df[column] = converted

# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["classification"])
y = df["classification"]

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Dataset shape:", df.shape)
print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# --------------------------------------------------
# 5. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 6. Define machine learning models
# --------------------------------------------------

svm_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

knn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Voting Ensemble
ensemble_pipeline = VotingClassifier(
    estimators=[
        ("svm", svm_pipeline),
        ("knn", knn_pipeline),
        ("rf", rf_pipeline)
    ],
    voting="soft"
)

models = {
    "svm": svm_pipeline,
    "knn": knn_pipeline,
    "random_forest": rf_pipeline,
    "ensemble": ensemble_pipeline
}

# --------------------------------------------------
# 7. Train, evaluate and save models
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\nTraining", model_name, "...")

    model.fit(X_train, y_train)

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
        "Model": model_name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    # Save complete pipeline, including preprocessing
    model_path = os.path.join(
        MODEL_DIR, f"kidney_{model_name}.pkl"
    )

    joblib.dump(model, model_path)

    print("Saved:", model_path)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(
        y_test,
        predictions,
        target_names=["Not CKD", "CKD"],
        zero_division=0
    ))

# --------------------------------------------------
# 8. Display comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== KIDNEY DISEASE MODEL RESULTS ==========")
print(results_df.to_string(index=False))

results_path = os.path.join(
    MODEL_DIR, "kidney_model_results.csv"
)

results_df.to_csv(results_path, index=False)

print("\nAll models trained and saved successfully.")
print("Results saved at:", results_path)
