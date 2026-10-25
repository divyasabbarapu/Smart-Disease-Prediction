import os
import sqlite3
import hashlib
import pandas as pd
from datetime import datetime


# =========================================================
# DATABASE PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "medipredict.db"
)


# =========================================================
# CONNECTION
# =========================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False
    )

    connection.row_factory = sqlite3.Row

    return connection


# =========================================================
# INITIALIZE DATABASE
# =========================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    # USERS
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            full_name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            created_at TEXT NOT NULL

        )
        """
    )

    # PATIENTS
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            patient_name TEXT NOT NULL,

            email TEXT,

            age INTEGER,

            gender TEXT,

            phone TEXT,

            created_at TEXT NOT NULL,

            FOREIGN KEY(user_id)
            REFERENCES users(id)

        )
        """
    )

    # PREDICTIONS
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            patient_id INTEGER,

            patient_name TEXT NOT NULL,

            disease TEXT NOT NULL,

            final_result INTEGER NOT NULL,

            svm_prediction INTEGER,

            knn_prediction INTEGER,

            random_forest_prediction INTEGER,

            ensemble_prediction INTEGER,

            agreement INTEGER,

            input_data TEXT,

            prediction_date TEXT NOT NULL,

            FOREIGN KEY(user_id)
            REFERENCES users(id),

            FOREIGN KEY(patient_id)
            REFERENCES patients(id)

        )
        """
    )

    connection.commit()

    connection.close()


# =========================================================
# PASSWORD HASH
# =========================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# =========================================================
# REGISTER USER
# =========================================================

def register_user(
    full_name,
    email,
    password
):

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                full_name,
                email,
                password,
                created_at
            )

            VALUES (?, ?, ?, ?)
            """,
            (
                full_name.strip(),
                email.strip().lower(),
                hash_password(password),
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        )

        connection.commit()

        return True, "Account created successfully."

    except sqlite3.IntegrityError:

        return False, (
            "An account with this email "
            "already exists."
        )

    finally:

        connection.close()


# =========================================================
# LOGIN
# =========================================================

def authenticate_user(
    email,
    password
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users

        WHERE email = ?
        AND password = ?
        """,
        (
            email.strip().lower(),
            hash_password(password)
        )
    )

    user = cursor.fetchone()

    connection.close()

    if user:

        return dict(user)

    return None


# =========================================================
# ADD PATIENT
# =========================================================

def add_patient(
    user_id,
    patient_name,
    age,
    gender,
    phone,
    email=None
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO patients
        (
            user_id,
            patient_name,
            email,
            age,
            gender,
            phone,
            created_at
        )

        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            patient_name,
            email,
            age,
            gender,
            phone,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    connection.commit()

    patient_id = cursor.lastrowid

    connection.close()

    return patient_id


# =========================================================
# GET PATIENTS
# =========================================================

def get_patients(user_id):

    connection = get_connection()

    query = """
        SELECT
            id,
            patient_name,
            email,
            age,
            gender,
            phone,
            created_at

        FROM patients

        WHERE user_id = ?

        ORDER BY id DESC
    """

    dataframe = pd.read_sql_query(
        query,
        connection,
        params=(user_id,)
    )

    connection.close()

    return dataframe


# =========================================================
# SAVE PREDICTION
# =========================================================

def save_prediction(
    user_id,
    patient_id,
    patient_name,
    disease,
    final_result,
    svm_prediction,
    knn_prediction,
    random_forest_prediction,
    ensemble_prediction,
    agreement,
    input_data
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (
            user_id,
            patient_id,
            patient_name,
            disease,
            final_result,
            svm_prediction,
            knn_prediction,
            random_forest_prediction,
            ensemble_prediction,
            agreement,
            input_data,
            prediction_date
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            patient_id,
            patient_name,
            disease,
            final_result,
            svm_prediction,
            knn_prediction,
            random_forest_prediction,
            ensemble_prediction,
            agreement,
            str(input_data),
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    connection.commit()

    connection.close()


# =========================================================
# GET PREDICTIONS
# =========================================================

def get_predictions(user_id):

    connection = get_connection()

    query = """
        SELECT
            id,
            patient_name,
            disease,
            final_result,
            svm_prediction,
            knn_prediction,
            random_forest_prediction,
            ensemble_prediction,
            agreement,
            prediction_date

        FROM predictions

        WHERE user_id = ?

        ORDER BY id DESC
    """

    dataframe = pd.read_sql_query(
        query,
        connection,
        params=(user_id,)
    )

    connection.close()

    return dataframe


# =========================================================
# INITIALIZE
# =========================================================

initialize_database()