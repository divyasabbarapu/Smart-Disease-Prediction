import os
import streamlit as st

from database import (
    authenticate_user,
    register_user
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MediPredict",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# LOAD CSS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CSS_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "style.css"
)

with open(
    CSS_PATH,
    "r",
    encoding="utf-8"
) as file:

    st.markdown(
        f"<style>{file.read()}</style>",
        unsafe_allow_html=True
    )


# =========================================================
# SESSION STATE
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None


# =========================================================
# ALREADY LOGGED IN
# =========================================================

if st.session_state.logged_in:

    st.success(
        f"Welcome back, {st.session_state.user['full_name']}!"
    )

    if st.button(
        "Go to Dashboard",
        type="primary"
    ):

        st.switch_page(
            "pages/1_Dashboard.py"
        )

    st.stop()


# =========================================================
# MAIN PAGE
# =========================================================

left, right = st.columns(
    [1.15, 0.85],
    gap="large"
)


# =========================================================
# LEFT SIDE
# =========================================================

with left:

    st.markdown(
        '<div class="medical-badge">'
        '🏥 SMART HEALTHCARE AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-title">'
        'MediPredict'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Smart Disease Prediction using '
        'Ensemble Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        MediPredict is an academic healthcare machine learning
        platform that analyzes patient health parameters and
        provides disease predictions using multiple machine
        learning algorithms.
        """
    )

    st.markdown("### 🧠 Intelligent Prediction")

    st.write(
        "Combines Support Vector Machine, K-Nearest Neighbors "
        "and Random Forest models."
    )

    st.markdown("### 📊 Data-Driven Analysis")

    st.write(
        "Compare predictions from individual models and "
        "the ensemble learning system."
    )

    st.markdown("### 👥 Patient Management")

    st.write(
        "Store patient information and maintain prediction "
        "history securely within the application."
    )

    st.markdown("### 🔐 Secure User Access")

    st.write(
        "Users can create an account and access their "
        "own prediction dashboard."
    )


# =========================================================
# RIGHT SIDE
# =========================================================

with right:

    st.markdown(
        "## 🔐 Welcome Back"
    )

    st.write(
        "Login to access your MediPredict dashboard."
    )

    st.markdown("")


    # -----------------------------------------------------
    # LOGIN / REGISTER TABS
    # -----------------------------------------------------

    login_tab, register_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Create Account"
        ]
    )


    # =====================================================
    # LOGIN
    # =====================================================

    with login_tab:

        st.markdown("### Sign in")

        email = st.text_input(
            "Email Address",
            placeholder="Enter your email",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )

        st.markdown("")

        if st.button(
            "Login to MediPredict",
            type="primary",
            use_container_width=True
        ):

            if not email or not password:

                st.warning(
                    "Please enter your email and password."
                )

            else:

                user = authenticate_user(
                    email,
                    password
                )

                if user:

                    st.session_state.logged_in = True
                    st.session_state.user = user

                    st.success(
                        "Login successful!"
                    )

                    st.switch_page(
                        "pages/1_Dashboard.py"
                    )

                else:

                    st.error(
                        "Invalid email or password."
                    )


    # =====================================================
    # REGISTER
    # =====================================================

    with register_tab:

        st.markdown("### Create Account")

        st.write(
            "Create your MediPredict account."
        )

        full_name = st.text_input(
            "Full Name",
            placeholder="Enter your full name",
            key="register_name"
        )

        register_email = st.text_input(
            "Email Address",
            placeholder="Enter your email",
            key="register_email"
        )

        register_password = st.text_input(
            "Password",
            type="password",
            placeholder="Minimum 6 characters",
            key="register_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password",
            key="confirm_password"
        )

        st.markdown("")

        if st.button(
            "Create MediPredict Account",
            use_container_width=True
        ):

            if not full_name:

                st.warning(
                    "Please enter your full name."
                )

            elif not register_email:

                st.warning(
                    "Please enter your email."
                )

            elif len(register_password) < 6:

                st.warning(
                    "Password must contain at least 6 characters."
                )

            elif register_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                success, message = register_user(
                    full_name,
                    register_email,
                    register_password
                )

                if success:

                    st.success(
                        "Account created successfully!"
                    )

                    st.info(
                        "Switch to the Login tab to continue."
                    )

                else:

                    st.error(message)


# =========================================================
# BOTTOM INFORMATION
# =========================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("### 🩺 Diseases")

    st.write(
        "Heart Disease and Diabetes"
    )

with col2:

    st.markdown("### 🤖 Algorithms")

    st.write(
        "SVM • KNN • Random Forest"
    )

with col3:

    st.markdown("### 🧠 Approach")

    st.write(
        "Ensemble Machine Learning"
    )


st.markdown(
    """
    <div class="footer">
        🏥 MediPredict • Smart Disease Prediction with Ensemble Learning
        <br>
        Academic Machine Learning Project
        <br><br>
        This system is intended for educational purposes only
        and does not replace professional medical diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)