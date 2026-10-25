import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from database import get_predictions, get_patients


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Dashboard | MediPredict",
    page_icon="🏥",
    layout="wide"
)


# ---------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------

if not st.session_state.get("logged_in", False):

    st.warning("Please login first.")

    st.switch_page("app.py")

    st.stop()


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

css_path = os.path.join(
    BASE_DIR,
    "assets",
    "style.css"
)

with open(css_path, "r", encoding="utf-8") as file:

    st.markdown(
        f"<style>{file.read()}</style>",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

user = st.session_state.user

st.sidebar.markdown(
    "## 🏥 MediPredict"
)

st.sidebar.write(
    f"👤 **{user['full_name']}**"
)

st.sidebar.write(
    user["email"]
)

st.sidebar.markdown("---")

if st.sidebar.button(
    "🚪 Logout",
    use_container_width=True
):

    st.session_state.logged_in = False
    st.session_state.user = None

    st.switch_page("app.py")


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

patients = get_patients(user["id"])

predictions = get_predictions(user["id"])


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="hero-title">Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="hero-subtitle">'
    f'Welcome back, {user["full_name"]} 👋'
    f'</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

total_patients = len(patients)
total_predictions = len(predictions)

heart_predictions = 0
diabetes_predictions = 0

if not predictions.empty:

    heart_predictions = len(
        predictions[
            predictions["disease"] == "Heart Disease"
        ]
    )

    diabetes_predictions = len(
        predictions[
            predictions["disease"] == "Diabetes"
        ]
    )


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="card-title">Registered Patients</div>
            <div class="card-value">{total_patients}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="card-title">Total Predictions</div>
            <div class="card-value">{total_predictions}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="card-title">Heart Predictions</div>
            <div class="card-value">{heart_predictions}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="card-title">Diabetes Predictions</div>
            <div class="card-value">{diabetes_predictions}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# QUICK ACTIONS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">Quick Actions</div>',
    unsafe_allow_html=True
)

a1, a2, a3 = st.columns(3)


with a1:

    if st.button(
        "🩺 New Prediction",
        use_container_width=True
    ):

        st.switch_page(
            "pages/2_Prediction.py"
        )


with a2:

    if st.button(
        "👥 Patient Records",
        use_container_width=True
    ):

        st.switch_page(
            "pages/3_Patient_Records.py"
        )


with a3:

    if st.button(
        "📊 Model Performance",
        use_container_width=True
    ):

        st.switch_page(
            "pages/4_Model_Performance.py"
        )


# ---------------------------------------------------------
# RECENT PREDICTIONS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">Recent Predictions</div>',
    unsafe_allow_html=True
)

if predictions.empty:

    st.info(
        "No predictions have been recorded yet."
    )

else:

    recent = predictions.head(5).copy()

    recent["final_result"] = recent[
        "final_result"
    ].map(
        {
            1: "Possible Disease",
            0: "No Disease"
        }
    )

    recent = recent[
        [
            "patient_name",
            "disease",
            "final_result",
            "agreement",
            "prediction_date"
        ]
    ]

    recent.columns = [
        "Patient",
        "Disease",
        "Result",
        "Agreement",
        "Date"
    ]

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "MediPredict is an academic machine learning system and "
    "does not replace professional medical diagnosis."
)