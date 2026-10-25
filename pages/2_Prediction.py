import os
import sys
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from database import add_patient, save_prediction
from src.predict import (
    predict_disease,
    KIDNEY_FEATURES,
    LIVER_FEATURES
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="New Prediction | MediPredict",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

css_path = os.path.join(BASE_DIR, "assets", "style.css")

if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# =========================================================
# LOGIN CHECK
# =========================================================

if not st.session_state.get("logged_in", False):
    st.warning("Please login to access this page.")
    st.switch_page("app.py")
    st.stop()

user = st.session_state.user


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        font-size:28px;
        font-weight:800;
        margin-bottom:20px;
    ">
        🏥 MediPredict
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div style="
        padding:15px;
        border-radius:12px;
        background:rgba(255,255,255,0.12);
        margin-bottom:20px;
    ">
        <b>👤 {user['full_name']}</b><br>
        <small>{user['email']}</small>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### Navigation")

if st.sidebar.button("🏠 Dashboard", use_container_width=True):
    st.switch_page("pages/1_Dashboard.py")

if st.sidebar.button("🩺 New Prediction", use_container_width=True):
    st.rerun()

if st.sidebar.button("👥 Patient Records", use_container_width=True):
    st.switch_page("pages/3_Patient_Records.py")

if st.sidebar.button("ℹ️ About", use_container_width=True):
    st.switch_page("pages/4_About.py")

st.sidebar.markdown("---")

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.switch_page("app.py")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="medical-badge">🩺 PATIENT HEALTH ANALYSIS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">New Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Enter patient information and health parameters '
    'to generate an ensemble-based prediction.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# DISEASE SELECTION
# =========================================================

st.markdown(
    '<div class="section-heading">1. Select Disease</div>',
    unsafe_allow_html=True
)

disease = st.selectbox(
    "Prediction Type",
    [
        "Heart Disease",
        "Diabetes",
        "Kidney Disease",
        "Liver Disease"
    ]
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-heading">2. Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input(
        "Patient Name *",
        placeholder="Enter patient name"
    )

with col2:
    patient_age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

with col3:
    gender = st.selectbox(
        "Gender",
        ["Not Specified", "Male", "Female"]
    )

phone = st.text_input(
    "Phone Number",
    placeholder="Optional"
)


# =========================================================
# MODEL FEATURES
# =========================================================

model_name = disease.lower().replace(" ", "_")

if disease == "Kidney Disease":

    features = KIDNEY_FEATURES

elif disease == "Liver Disease":

    features = LIVER_FEATURES

else:

    features_path = os.path.join(
        BASE_DIR,
        "models",
        model_name + "_features.pkl"
    )

    if not os.path.exists(features_path):
        st.error(
            f"Model feature file not found: {features_path}"
        )
        st.stop()

    features = joblib.load(features_path)


# =========================================================
# HEALTH PARAMETER INFORMATION
# =========================================================

st.markdown(
    '<div class="section-heading">3. Health Parameters</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the values required by the selected machine learning model. "
    "For Liver Disease, enter the laboratory values from the patient's "
    "report."
)


# =========================================================
# NUMERICAL FEATURE DETAILS
# =========================================================

feature_info = {

    # ---------------- HEART DISEASE ----------------

    "age": {
        "label": "Age",
        "min": 1.0,
        "max": 120.0,
        "step": 1.0,
        "default": 45.0
    },

    "sex": {
        "label": "Sex",
        "min": 0.0,
        "max": 1.0,
        "step": 1.0,
        "default": 1.0
    },

    "cp": {
        "label": "Chest Pain Type",
        "min": 0.0,
        "max": 3.0,
        "step": 1.0,
        "default": 0.0
    },

    "trestbps": {
        "label": "Resting Blood Pressure",
        "min": 50.0,
        "max": 250.0,
        "step": 1.0,
        "default": 120.0
    },

    "chol": {
        "label": "Serum Cholesterol",
        "min": 50.0,
        "max": 700.0,
        "step": 1.0,
        "default": 200.0
    },

    "fbs": {
        "label": "Fasting Blood Sugar",
        "min": 0.0,
        "max": 1.0,
        "step": 1.0,
        "default": 0.0
    },

    "restecg": {
        "label": "Resting ECG",
        "min": 0.0,
        "max": 2.0,
        "step": 1.0,
        "default": 0.0
    },

    "thalach": {
        "label": "Maximum Heart Rate",
        "min": 50.0,
        "max": 250.0,
        "step": 1.0,
        "default": 150.0
    },

    "exang": {
        "label": "Exercise Induced Angina",
        "min": 0.0,
        "max": 1.0,
        "step": 1.0,
        "default": 0.0
    },

    "oldpeak": {
        "label": "ST Depression",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "default": 1.0
    },

    "slope": {
        "label": "Slope",
        "min": 0.0,
        "max": 2.0,
        "step": 1.0,
        "default": 1.0
    },

    "ca": {
        "label": "Number of Major Vessels",
        "min": 0.0,
        "max": 4.0,
        "step": 1.0,
        "default": 0.0
    },

    "thal": {
        "label": "Thalassemia",
        "min": 0.0,
        "max": 3.0,
        "step": 1.0,
        "default": 2.0
    },


    # ---------------- DIABETES ----------------

    "pregnancies": {
        "label": "Pregnancies",
        "min": 0.0,
        "max": 20.0,
        "step": 1.0,
        "default": 2.0
    },

    "glucose": {
        "label": "Glucose Level",
        "min": 0.0,
        "max": 300.0,
        "step": 1.0,
        "default": 120.0
    },

    "bloodpressure": {
        "label": "Blood Pressure",
        "min": 0.0,
        "max": 200.0,
        "step": 1.0,
        "default": 70.0
    },

    "skinthickness": {
        "label": "Skin Thickness",
        "min": 0.0,
        "max": 100.0,
        "step": 1.0,
        "default": 20.0
    },

    "insulin": {
        "label": "Insulin",
        "min": 0.0,
        "max": 900.0,
        "step": 1.0,
        "default": 80.0
    },

    "bmi": {
        "label": "BMI",
        "min": 0.0,
        "max": 70.0,
        "step": 0.1,
        "default": 25.0
    },

    "diabetespedigreefunction": {
        "label": "Diabetes Pedigree Function",
        "min": 0.0,
        "max": 3.0,
        "step": 0.01,
        "default": 0.5
    },


    # ---------------- KIDNEY DISEASE ----------------

    "bp": {
        "label": "Blood Pressure",
        "min": 0.0,
        "max": 200.0,
        "step": 1.0,
        "default": 80.0
    },

    "sg": {
        "label": "Specific Gravity",
        "min": 1.0,
        "max": 1.03,
        "step": 0.005,
        "default": 1.020
    },

    "al": {
        "label": "Albumin",
        "min": 0.0,
        "max": 5.0,
        "step": 1.0,
        "default": 0.0
    },

    "su": {
        "label": "Sugar",
        "min": 0.0,
        "max": 5.0,
        "step": 1.0,
        "default": 0.0
    },

    "bgr": {
        "label": "Blood Glucose Random",
        "min": 0.0,
        "max": 600.0,
        "step": 1.0,
        "default": 121.0
    },

    "bu": {
        "label": "Blood Urea",
        "min": 0.0,
        "max": 400.0,
        "step": 1.0,
        "default": 36.0
    },

    "sc": {
        "label": "Serum Creatinine",
        "min": 0.0,
        "max": 80.0,
        "step": 0.1,
        "default": 1.2
    },

    "sod": {
        "label": "Sodium",
        "min": 0.0,
        "max": 200.0,
        "step": 1.0,
        "default": 138.0
    },

    "pot": {
        "label": "Potassium",
        "min": 0.0,
        "max": 20.0,
        "step": 0.1,
        "default": 4.4
    },

    "hemo": {
        "label": "Hemoglobin",
        "min": 0.0,
        "max": 25.0,
        "step": 0.1,
        "default": 13.0
    },

    "pcv": {
        "label": "Packed Cell Volume",
        "min": 0.0,
        "max": 60.0,
        "step": 1.0,
        "default": 40.0
    },

    "wc": {
        "label": "White Blood Cell Count",
        "min": 0.0,
        "max": 30000.0,
        "step": 100.0,
        "default": 8000.0
    },

    "rc": {
        "label": "Red Blood Cell Count",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "default": 4.8
    },


    # ---------------- LIVER DISEASE ----------------

    "total_bilirubin": {
        "label": "Total Bilirubin",
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
        "default": 1.0
    },

    "direct_bilirubin": {
        "label": "Direct Bilirubin",
        "min": 0.0,
        "max": 50.0,
        "step": 0.1,
        "default": 0.3
    },

    "alkaline_phosphotase": {
        "label": "Alkaline Phosphotase",
        "min": 0.0,
        "max": 3000.0,
        "step": 1.0,
        "default": 200.0
    },

    "alamine_aminotransferase": {
        "label": "Alamine Aminotransferase (ALT)",
        "min": 0.0,
        "max": 3000.0,
        "step": 1.0,
        "default": 30.0
    },

    "aspartate_aminotransferase": {
        "label": "Aspartate Aminotransferase (AST)",
        "min": 0.0,
        "max": 5000.0,
        "step": 1.0,
        "default": 35.0
    },

    "total_protiens": {
        "label": "Total Proteins",
        "min": 0.0,
        "max": 15.0,
        "step": 0.1,
        "default": 6.5
    },

    "albumin": {
        "label": "Albumin",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "default": 3.5
    },

    "albumin_and_globulin_ratio": {
        "label": "Albumin and Globulin Ratio",
        "min": 0.0,
        "max": 5.0,
        "step": 0.01,
        "default": 1.0
    }
}


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

categorical_options = {

    # Kidney Disease
    "rbc": {
        "label": "Red Blood Cells",
        "options": ["normal", "abnormal"]
    },

    "pc": {
        "label": "Pus Cell",
        "options": ["normal", "abnormal"]
    },

    "pcc": {
        "label": "Pus Cell Clumps",
        "options": ["notpresent", "present"]
    },

    "ba": {
        "label": "Bacteria",
        "options": ["notpresent", "present"]
    },

    "htn": {
        "label": "Hypertension",
        "options": ["no", "yes"]
    },

    "dm": {
        "label": "Diabetes Mellitus",
        "options": ["no", "yes"]
    },

    "cad": {
        "label": "Coronary Artery Disease",
        "options": ["no", "yes"]
    },

    "appet": {
        "label": "Appetite",
        "options": ["good", "poor"]
    },

    "pe": {
        "label": "Pedal Edema",
        "options": ["no", "yes"]
    },

    "ane": {
        "label": "Anemia",
        "options": ["no", "yes"]
    },

    # Liver Disease
    "gender": {
        "label": "Gender",
        "options": ["Female", "Male"]
    }
}


# =========================================================
# CREATE HEALTH PARAMETER INPUTS
# =========================================================

input_data = {}

columns = st.columns(3)

for index, feature in enumerate(features):

    feature_key = str(feature).lower()

    column = columns[index % 3]

    with column:

        # Categorical inputs for Kidney and Liver Disease
        if (
            disease in ["Kidney Disease", "Liver Disease"]
            and feature_key in categorical_options
        ):

            info = categorical_options[feature_key]

            input_data[feature] = st.selectbox(
                info["label"],
                info["options"],
                key=f"input_{feature}"
            )

        # Numerical inputs
        elif feature_key in feature_info:

            info = feature_info[feature_key]

            input_data[feature] = st.number_input(
                info["label"],
                min_value=info["min"],
                max_value=info["max"],
                value=info["default"],
                step=info["step"],
                key=f"input_{feature}"
            )

        else:

            input_data[feature] = st.number_input(
                str(feature),
                value=0.0,
                key=f"input_{feature}"
            )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("")

st.markdown(
    '<div class="section-heading">4. Generate Prediction</div>',
    unsafe_allow_html=True
)

predict_button = st.button(
    "🔬 Analyze Patient & Generate Prediction",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if not patient_name.strip():
        st.error(
            "Please enter the patient name before prediction."
        )
        st.stop()

    try:

        # Generate predictions
        results = predict_disease(
            disease,
            input_data
        )

        svm_prediction = results["SVM"]
        knn_prediction = results["KNN"]
        rf_prediction = results["Random Forest"]
        ensemble_prediction = results["Ensemble"]

        agreement = results["Agreement"]
        final_result = results["Final"]


        # -------------------------------------------------
        # SAVE PATIENT
        # -------------------------------------------------

        patient_id = add_patient(
            user["id"],
            patient_name.strip(),
            patient_age,
            gender,
            phone.strip(),
            email=user["email"]
        )


        # -------------------------------------------------
        # SAVE PREDICTION
        # -------------------------------------------------

        save_prediction(
            user_id=user["id"],
            patient_id=patient_id,
            patient_name=patient_name.strip(),
            disease=disease,
            final_result=final_result,
            svm_prediction=svm_prediction,
            knn_prediction=knn_prediction,
            random_forest_prediction=rf_prediction,
            ensemble_prediction=ensemble_prediction,
            agreement=agreement,
            input_data=input_data
        )


        # -------------------------------------------------
        # SAVE RESULT IN SESSION
        # -------------------------------------------------

        st.session_state["last_prediction"] = {
            "patient_name": patient_name,
            "disease": disease,
            "svm": svm_prediction,
            "knn": knn_prediction,
            "rf": rf_prediction,
            "ensemble": ensemble_prediction,
            "final": final_result,
            "agreement": agreement,
            "input_data": input_data,
            "date": datetime.now().strftime(
                "%d %B %Y, %I:%M %p"
            )
        }


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.markdown("---")

        st.markdown(
            '<div class="section-heading">Prediction Results</div>',
            unsafe_allow_html=True
        )

        if final_result == 1:

            st.error(
                f"⚠️ The ensemble system indicates a "
                f"possible {disease}."
            )

        else:

            st.success(
                f"✅ The ensemble system indicates no "
                f"predicted {disease}."
            )


        # -------------------------------------------------
        # MODEL RESULTS
        # -------------------------------------------------

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            result = (
                "Disease"
                if svm_prediction == 1
                else "No Disease"
            )
            st.metric("SVM", result)

        with r2:
            result = (
                "Disease"
                if knn_prediction == 1
                else "No Disease"
            )
            st.metric("KNN", result)

        with r3:
            result = (
                "Disease"
                if rf_prediction == 1
                else "No Disease"
            )
            st.metric("Random Forest", result)

        with r4:
            result = (
                "Disease"
                if ensemble_prediction == 1
                else "No Disease"
            )
            st.metric("Ensemble", result)


        # -------------------------------------------------
        # MODEL AGREEMENT
        # -------------------------------------------------

        st.markdown("### 🤝 Model Agreement")

        if agreement == 3:

            st.success(
                "All 3 individual models produced "
                "the same classification."
            )

        else:

            st.info(
                f"{agreement} out of 3 models agreed "
                "on the majority classification."
            )


        # -------------------------------------------------
        # PATIENT SUMMARY
        # -------------------------------------------------

        st.markdown("### 👤 Patient Summary")

        summary1, summary2, summary3 = st.columns(3)

        with summary1:
            st.write(f"**Patient:** {patient_name}")

        with summary2:
            st.write(f"**Age:** {patient_age}")

        with summary3:
            st.write(f"**Disease:** {disease}")


        # -------------------------------------------------
        # DOWNLOAD REPORT
        # -------------------------------------------------

        report = f"""
MEDIPREDICT
SMART DISEASE PREDICTION SYSTEM
========================================

PATIENT INFORMATION
----------------------------------------
Patient Name: {patient_name}
Age: {patient_age}
Gender: {gender}
Phone: {phone}
Disease Analysis: {disease}

PREDICTION RESULTS
----------------------------------------
SVM: {"Disease" if svm_prediction == 1 else "No Disease"}
KNN: {"Disease" if knn_prediction == 1 else "No Disease"}
Random Forest: {"Disease" if rf_prediction == 1 else "No Disease"}
Ensemble: {"Disease" if ensemble_prediction == 1 else "No Disease"}

MODEL AGREEMENT
----------------------------------------
Agreement: {agreement}/3 models

FINAL RESULT
----------------------------------------
{"Possible Disease Detected" if final_result == 1 else "No Disease Predicted"}

DATE
----------------------------------------
{datetime.now().strftime("%d %B %Y, %I:%M %p")}

----------------------------------------
Educational ML system only.
This prediction does not replace professional
medical diagnosis.
"""

        st.download_button(
            label="📄 Download Prediction Report",
            data=report,
            file_name=(
                f"{patient_name.replace(' ', '_')}_"
                f"{model_name}_report.txt"
            ),
            mime="text/plain",
            use_container_width=True
        )


    except Exception as error:

        st.error(
            "Prediction could not be completed."
        )

        st.exception(error)