import os
import sys

import streamlit as st
import pandas as pd


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)


from database import (
    get_patients,
    get_predictions
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Patient Records | MediPredict",
    page_icon="👥",
    layout="wide"
)


# =========================================================
# LOAD CSS
# =========================================================

css_path = os.path.join(
    BASE_DIR,
    "assets",
    "style.css"
)

if os.path.exists(css_path):

    with open(
        css_path,
        "r",
        encoding="utf-8"
    ) as file:

        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# =========================================================
# LOGIN CHECK
# =========================================================

if not st.session_state.get(
    "logged_in",
    False
):

    st.warning(
        "Please login to access patient records."
    )

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


if st.sidebar.button(
    "🏠 Dashboard",
    use_container_width=True
):

    st.switch_page(
        "pages/1_Dashboard.py"
    )


if st.sidebar.button(
    "🩺 New Prediction",
    use_container_width=True
):

    st.switch_page(
        "pages/2_Prediction.py"
    )


if st.sidebar.button(
    "👥 Patient Records",
    use_container_width=True
):

    st.rerun()


if st.sidebar.button(
    "📊 Model Performance",
    use_container_width=True
):

    st.switch_page(
        "pages/4_Model_Performance.py"
    )


if st.sidebar.button(
    "ℹ️ About",
    use_container_width=True
):

    st.switch_page(
        "pages/5_About.py"
    )


st.sidebar.markdown("---")


if st.sidebar.button(
    "🚪 Logout",
    use_container_width=True
):

    st.session_state.logged_in = False
    st.session_state.user = None

    st.switch_page("app.py")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="medical-badge">'
    '👥 PATIENT MANAGEMENT'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">'
    'Patient Records'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'View registered patients and their prediction history.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

try:

    patients = get_patients(
        user["id"]
    )

    predictions = get_predictions(
        user["id"]
    )

except Exception as error:

    st.error(
        "Unable to load patient records."
    )

    st.exception(error)

    st.stop()


# =========================================================
# SUMMARY CARDS
# =========================================================

total_patients = len(
    patients
)

total_predictions = len(
    predictions
)

heart_predictions = 0
diabetes_predictions = 0

if not predictions.empty:

    heart_predictions = len(
        predictions[
            predictions["disease"]
            .str.contains(
                "Heart",
                case=False,
                na=False
            )
        ]
    )

    diabetes_predictions = len(
        predictions[
            predictions["disease"]
            .str.contains(
                "Diabetes",
                case=False,
                na=False
            )
        ]
    )


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Registered Patients",
        total_patients
    )


with c2:

    st.metric(
        "Total Predictions",
        total_predictions
    )


with c3:

    st.metric(
        "Heart Disease",
        heart_predictions
    )


with c4:

    st.metric(
        "Diabetes",
        diabetes_predictions
    )


st.markdown("")


# =========================================================
# PATIENT RECORDS
# =========================================================

st.markdown(
    '<div class="section-heading">'
    '👤 Registered Patients'
    '</div>',
    unsafe_allow_html=True
)


if patients.empty:

    st.info(
        "No patient records have been created yet. "
        "Create a new prediction to add a patient."
    )

else:

    search = st.text_input(
        "🔎 Search patient",
        placeholder="Search by patient name, gender or phone"
    )


    filtered_patients = patients.copy()


    if search.strip():

        search_value = (
            search.strip()
            .lower()
        )

        filtered_patients = (
            filtered_patients[
                filtered_patients.astype(
                    str
                )
                .apply(
                    lambda row:
                    row.str.lower()
                    .str.contains(
                        search_value,
                        na=False
                    )
                    .any(),
                    axis=1
                )
            ]
        )


    display_patients = filtered_patients.copy()


    if "created_at" in display_patients.columns:

        display_patients[
            "created_at"
        ] = pd.to_datetime(
            display_patients["created_at"],
            errors="coerce"
        ).dt.strftime(
            "%d %b %Y, %I:%M %p"
        )


    st.dataframe(
        display_patients,
        use_container_width=True,
        hide_index=True,
        column_config={

            "id": "Patient ID",

            "patient_name": "Patient Name",

            "email": "Email",

            "age": st.column_config.NumberColumn(
                "Age"
            ),

            "gender": "Gender",

            "phone": "Phone",

            "created_at": "Registered On"
        }
    )


    # -----------------------------------------------------
    # DOWNLOAD PATIENT RECORDS
    # -----------------------------------------------------

    patient_csv = filtered_patients.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Patient Records",
        patient_csv,
        file_name="medipredict_patient_records.csv",
        mime="text/csv"
    )


# =========================================================
# PREDICTION HISTORY
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-heading">'
    '📋 Prediction History'
    '</div>',
    unsafe_allow_html=True
)


if predictions.empty:

    st.info(
        "No predictions have been recorded yet."
    )

else:

    history = predictions.copy()


    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    f1, f2 = st.columns(2)


    with f1:

        disease_filter = st.selectbox(
            "Disease",
            [
                "All",
                "Heart Disease",
                "Diabetes"
            ]
        )


    with f2:

        result_filter = st.selectbox(
            "Prediction Result",
            [
                "All",
                "Disease Predicted",
                "No Disease Predicted"
            ]
        )


    if disease_filter != "All":

        history = history[
            history["disease"]
            == disease_filter
        ]


    if result_filter == "Disease Predicted":

        history = history[
            history["final_result"] == 1
        ]


    elif result_filter == "No Disease Predicted":

        history = history[
            history["final_result"] == 0
        ]


    # -----------------------------------------------------
    # FORMAT HISTORY
    # -----------------------------------------------------

    history_display = history.copy()


    history_display["Result"] = (
        history_display["final_result"]
        .apply(
            lambda value:
            "⚠️ Disease"
            if value == 1
            else "✅ No Disease"
        )
    )


    history_display["Agreement"] = (
        history_display["agreement"]
        .astype(str)
        + "/3"
    )


    history_display["SVM"] = (
        history_display["svm_prediction"]
        .apply(
            lambda value:
            "Disease"
            if value == 1
            else "No Disease"
        )
    )


    history_display["KNN"] = (
        history_display["knn_prediction"]
        .apply(
            lambda value:
            "Disease"
            if value == 1
            else "No Disease"
        )
    )


    history_display["Random Forest"] = (
        history_display[
            "random_forest_prediction"
        ]
        .apply(
            lambda value:
            "Disease"
            if value == 1
            else "No Disease"
        )
    )


    history_display["Ensemble"] = (
        history_display[
            "ensemble_prediction"
        ]
        .apply(
            lambda value:
            "Disease"
            if value == 1
            else "No Disease"
        )
    )


    if "prediction_date" in history_display.columns:

        history_display[
            "prediction_date"
        ] = pd.to_datetime(
            history_display["prediction_date"],
            errors="coerce"
        ).dt.strftime(
            "%d %b %Y, %I:%M %p"
        )


    final_history = history_display[
        [
            "id",
            "patient_name",
            "disease",
            "SVM",
            "KNN",
            "Random Forest",
            "Ensemble",
            "Agreement",
            "Result",
            "prediction_date"
        ]
    ]


    st.dataframe(
        final_history,
        use_container_width=True,
        hide_index=True,
        column_config={

            "id": "Prediction ID",

            "patient_name": "Patient",

            "disease": "Disease",

            "prediction_date": "Date"
        }
    )


    # -----------------------------------------------------
    # DOWNLOAD HISTORY
    # -----------------------------------------------------

    history_csv = history.to_csv(
        index=False
    )


    st.download_button(
        "📥 Download Prediction History",
        history_csv,
        file_name="medipredict_prediction_history.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "MediPredict • Patient records are stored locally "
    "for this academic project."
)