import os
import sys

import streamlit as st


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="About | MediPredict",
    page_icon="ℹ️",
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
        "Please login to access MediPredict."
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

    st.switch_page(
        "pages/3_Patient_Records.py"
    )


if st.sidebar.button(
    "ℹ️ About",
    use_container_width=True
):

    st.rerun()


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
    'ℹ️ ABOUT MEDIPREDICT'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">'
    'About MediPredict'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Smart Disease Prediction with Ensemble Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        🏥 What is MediPredict?
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    MediPredict is an academic machine learning application
    designed to demonstrate how multiple classification
    algorithms can be used together for disease prediction.

    The system currently focuses on two prediction tasks:

    • Heart Disease Prediction

    • Diabetes Prediction

    Instead of depending on a single machine learning model,
    MediPredict uses multiple models and an ensemble approach
    to compare their predictions.
    """
)


# =========================================================
# OBJECTIVE
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        🎯 Project Objective
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    The main objective of this project is to develop a
    user-friendly machine learning system that demonstrates
    the practical application of ensemble learning in
    healthcare-related classification problems.

    The project combines machine learning, data processing,
    database management and a web-based interface into a
    single application.
    """
)


# =========================================================
# MACHINE LEARNING MODELS
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        🧠 Machine Learning Models
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        ### Support Vector Machine

        SVM is a supervised learning algorithm that finds
        a decision boundary for separating different
        classes of data.
        """
    )


with col2:

    st.markdown(
        """
        ### K-Nearest Neighbors

        KNN classifies a new data point by considering
        the classes of nearby observations in the dataset.
        """
    )


with col3:

    st.markdown(
        """
        ### Random Forest

        Random Forest combines multiple decision trees
        to produce a classification result and can handle
        complex relationships between features.
        """
    )


# =========================================================
# ENSEMBLE LEARNING
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        🔗 Why Ensemble Learning?
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    Different machine learning algorithms can learn
    different patterns from the same dataset.

    MediPredict therefore compares the outputs of SVM,
    KNN and Random Forest and uses an ensemble approach
    to obtain a combined prediction.

    The application also displays the individual model
    results and their level of agreement so that the user
    can understand how the final prediction was obtained.
    """
)


# =========================================================
# APPLICATION FEATURES
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        ✨ Application Features
    </div>
    """,
    unsafe_allow_html=True
)

features = [
    "🔐 User registration and secure login",
    "👤 Patient information management",
    "🩺 Heart disease prediction",
    "🩸 Diabetes prediction",
    "🤖 SVM, KNN and Random Forest predictions",
    "🔗 Ensemble-based prediction",
    "📊 Model agreement information",
    "📋 Prediction history",
    "💾 Local storage of patient and prediction records",
    "📥 Downloadable patient and prediction records"
]


for feature in features:

    st.markdown(
        f"""
        <div style="
            padding:10px 14px;
            margin:5px 0;
            border-radius:10px;
            background:rgba(255,255,255,0.06);
        ">
            {feature}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TECHNOLOGY STACK
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        💻 Technology Stack
    </div>
    """,
    unsafe_allow_html=True
)

tech1, tech2, tech3, tech4 = st.columns(4)


with tech1:

    st.markdown(
        """
        **Python**

        Data processing and machine learning
        """
    )


with tech2:

    st.markdown(
        """
        **Scikit-learn**

        Machine learning algorithms
        """
    )


with tech3:

    st.markdown(
        """
        **Streamlit**

        Web application interface
        """
    )


with tech4:

    st.markdown(
        """
        **SQLite**

        Local patient and prediction storage
        """
    )


# =========================================================
# WORKFLOW
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        🔄 How MediPredict Works
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    1. The user creates an account and logs into the system.

    2. Patient information is entered through the prediction
       interface.

    3. The required health parameters are provided.

    4. SVM, KNN and Random Forest process the input.

    5. The individual model predictions are displayed.

    6. The ensemble approach produces the combined result.

    7. Patient and prediction information is stored locally.

    8. Previous records can be viewed from the Patient
       Records section.
    """
)


# =========================================================
# IMPORTANT NOTE
# =========================================================

st.markdown(
    """
    <div class="section-heading">
        ⚠️ Important
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    """
    MediPredict is an academic machine learning project
    created for educational and demonstration purposes.

    Its predictions should not be considered a medical
    diagnosis or a replacement for professional medical
    advice. Real clinical decisions require qualified
    healthcare professionals and appropriate clinical
    evaluation.
    """
)


# =========================================================
# PROJECT DETAILS
# =========================================================

st.markdown(
    "---"
)

st.markdown(
    """
    ### 🎓 Project

    **Smart Disease Prediction with Ensemble Learning**

    **Domain:** Machine Learning / Healthcare

    **Interface:** Streamlit

    **Models:** SVM + KNN + Random Forest + Ensemble

    **Application:** MediPredict
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "---"
)

st.caption(
    "MediPredict • Smart Disease Prediction with "
    "Ensemble Learning • Academic Project"
)