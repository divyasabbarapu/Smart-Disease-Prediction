import pandas as pd
import os


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")


# ---------------------------------------------------------
# DATASET FILES
# ---------------------------------------------------------

HEART_FILE = os.path.join(DATASET_DIR, "heart_disease.csv")
DIABETES_FILE = os.path.join(DATASET_DIR, "diabetes.csv")


# ---------------------------------------------------------
# FUNCTION TO INSPECT A DATASET
# ---------------------------------------------------------

def inspect_dataset(file_path, dataset_name):
    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} DATASET")
    print("=" * 70)

    if not os.path.exists(file_path):
        print(f"\nERROR: File not found:")
        print(file_path)
        return

    try:
        df = pd.read_csv(file_path)

        print("\n1. Dataset shape")
        print("-" * 30)
        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")

        print("\n2. Column names")
        print("-" * 30)
        for column in df.columns:
            print(column)

        print("\n3. First five rows")
        print("-" * 30)
        print(df.head())

        print("\n4. Data types")
        print("-" * 30)
        print(df.dtypes)

        print("\n5. Missing values")
        print("-" * 30)
        print(df.isnull().sum())

        print("\n6. Duplicate rows")
        print("-" * 30)
        print(df.duplicated().sum())

        print("\n7. Statistical summary")
        print("-" * 30)
        print(df.describe())

    except Exception as error:
        print("\nERROR while reading dataset:")
        print(error)


# ---------------------------------------------------------
# INSPECT BOTH DATASETS
# ---------------------------------------------------------

inspect_dataset(
    HEART_FILE,
    "Heart Disease"
)

inspect_dataset(
    DIABETES_FILE,
    "Diabetes"
)