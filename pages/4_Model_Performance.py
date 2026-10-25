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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Model Performance | MediPredict",
    page_icon="📊",
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
        "Please login to access model performance."
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
    "📊 Model Performance",
    use_container_width=True
):

    st.rerun()


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
    '📊 MACHINE LEARNING ANALYTICS'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">'
    'Model Performance'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Evaluation and comparison of the machine learning '
    'models used by MediPredict.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# EXPLANATION
# =========================================================

st.info(
    """
    MediPredict uses three individual machine learning
    algorithms — SVM, KNN and Random Forest — and combines
    them using ensemble voting.

    The performance below is calculated using the held-out
    test data from the datasets used during model training.
    """
)


# =========================================================
# LOAD EVALUATION FILES
# =========================================================

heart_file = os.path.join(
    BASE_DIR,
    "reports",
    "heart_disease_evaluation.csv"
)

diabetes_file = os.path.join(
    BASE_DIR,
    "reports",
    "diabetes_evaluation.csv"
)


def load_evaluation(path):

    if not os.path.exists(path):

        return None

    dataframe = pd.read_csv(path)

    return dataframe


heart_results = load_evaluation(
    heart_file
)

diabetes_results = load_evaluation(
    diabetes_file
)


# =========================================================
# DISEASE SELECTOR
# =========================================================

disease = st.radio(
    "Select Evaluation Dataset",
    [
        "Heart Disease",
        "Diabetes"
    ],
    horizontal=True
)


if disease == "Heart Disease":

    results = heart_results

else:

    results = diabetes_results


# =========================================================
# CHECK FILE
# =========================================================

if results is None:

    st.error(
        "Evaluation results were not found."
    )

    st.write(
        "Expected file:"
    )

    st.code(
        heart_file
        if disease == "Heart Disease"
        else diabetes_file
    )

    st.stop()


# =========================================================
# NORMALIZE COLUMN NAMES
# =========================================================

results.columns = [
    str(column).strip()
    for column in results.columns
]


# =========================================================
# FIND METRIC COLUMNS
# =========================================================

def find_column(
    dataframe,
    names
):

    for name in names:

        for column in dataframe.columns:

            if (
                str(column).lower()
                == name.lower()
            ):

                return column

    return None


model_column = find_column(
    results,
    [
        "Model",
        "model",
        "Algorithm",
        "algorithm"
    ]
)

accuracy_column = find_column(
    results,
    [
        "Accuracy",
        "accuracy"
    ]
)

precision_column = find_column(
    results,
    [
        "Precision",
        "precision"
    ]
)

recall_column = find_column(
    results,
    [
        "Recall",
        "recall"
    ]
)

f1_column = find_column(
    results,
    [
        "F1 Score",
        "F1",
        "f1",
        "f1_score"
    ]
)


# =========================================================
# PERFORMANCE TABLE
# =========================================================

st.markdown(
    '<div class="section-heading">'
    '📈 Performance Metrics'
    '</div>',
    unsafe_allow_html=True
)


if model_column is None:

    st.warning(
        "The evaluation file does not contain "
        "a recognizable model column."
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

else:

    display_columns = [
        column
        for column in [
            model_column,
            accuracy_column,
            precision_column,
            recall_column,
            f1_column
        ]
        if column is not None
    ]

    performance = results[
        display_columns
    ].copy()


    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# METRIC CARDS
# =========================================================

if (
    model_column is not None
    and accuracy_column is not None
):

    st.markdown(
        '<div class="section-heading">'
        '🎯 Accuracy Overview'
        '</div>',
        unsafe_allow_html=True
    )


    metric_columns = st.columns(
        len(results)
    )


    for index, row in results.iterrows():

        model_name = row[
            model_column
        ]

        accuracy = row[
            accuracy_column
        ]


        with metric_columns[index]:

            try:

                accuracy_display = (
                    f"{float(accuracy) * 100:.2f}%"
                )

            except:

                accuracy_display = str(
                    accuracy
                )


            st.metric(
                str(model_name),
                accuracy_display
            )


# =========================================================
# BAR CHART
# =========================================================

st.markdown(
    '<div class="section-heading">'
    '📊 Model Comparison'
    '</div>',
    unsafe_allow_html=True
)


if (
    model_column is not None
    and accuracy_column is not None
):

    chart_data = results[
        [
            model_column,
            accuracy_column
        ]
    ].copy()


    chart_data = chart_data.set_index(
        model_column
    )


    st.bar_chart(
        chart_data
    )


# =========================================================
# ALL METRICS CHART
# =========================================================

if (
    model_column is not None
    and accuracy_column is not None
    and precision_column is not None
    and recall_column is not None
    and f1_column is not None
):

    st.markdown(
        '<div class="section-heading">'
        '📐 Accuracy, Precision, Recall & F1'
        '</div>',
        unsafe_allow_html=True
    )


    metrics_chart = results[
        [
            model_column,
            accuracy_column,
            precision_column,
            recall_column,
            f1_column
        ]
    ].copy()


    metrics_chart = metrics_chart.set_index(
        model_column
    )


    st.bar_chart(
        metrics_chart
    )


# =========================================================
# CONFUSION MATRICES
# =========================================================

st.markdown(
    '<div class="section-heading">'
    '🔲 Confusion Matrix Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    """
    Confusion matrices show how many test samples were
    correctly and incorrectly classified by each model.
    """
)


disease_prefix = (
    "heart_disease"
    if disease == "Heart Disease"
    else "diabetes"
)


models = [
    (
        "SVM",
        f"{disease_prefix}_svm_confusion_matrix.png"
    ),

    (
        "KNN",
        f"{disease_prefix}_knn_confusion_matrix.png"
    ),

    (
        "Random Forest",
        f"{disease_prefix}_random_forest_confusion_matrix.png"
    ),

    (
        "Ensemble",
        f"{disease_prefix}_ensemble_confusion_matrix.png"
    )
]


available_images = []


for model_name, filename in models:

    image_path = os.path.join(
        BASE_DIR,
        "reports",
        filename
    )

    if os.path.exists(image_path):

        available_images.append(
            (
                model_name,
                image_path
            )
        )


if not available_images:

    st.info(
        "Confusion matrix images have not been generated yet."
    )

else:

    image_columns = st.columns(2)


    for index, (
        model_name,
        image_path
    ) in enumerate(available_images):

        with image_columns[index % 2]:

            st.markdown(
                f"### {model_name}"
            )

            st.image(
                image_path,
                use_container_width=True
            )


# =========================================================
# DOWNLOAD PERFORMANCE DATA
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-heading">'
    '📥 Export Evaluation Results'
    '</div>',
    unsafe_allow_html=True
)


csv_data = results.to_csv(
    index=False
)


st.download_button(
    "Download Model Evaluation CSV",
    csv_data,
    file_name=(
        "heart_disease_model_evaluation.csv"
        if disease == "Heart Disease"
        else "diabetes_model_evaluation.csv"
    ),
    mime="text/csv",
    use_container_width=True
)


# =========================================================
# METHODOLOGY
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-heading">'
    '🧠 Model Methodology'
    '</div>',
    unsafe_allow_html=True
)


method1, method2, method3 = st.columns(3)


with method1:

    st.markdown(
        """
        ### Support Vector Machine

        SVM identifies a decision boundary that separates
        different classes in the feature space.

        **Used for:** disease classification
        """
    )


with method2:

    st.markdown(
        """
        ### K-Nearest Neighbors

        KNN predicts a class using the labels of nearby
        observations in the feature space.

        **Used for:** pattern-based classification
        """
    )


with method3:

    st.markdown(
        """
        ### Random Forest

        Random Forest combines multiple decision trees to
        produce a classification result.

        **Used for:** robust classification
        """
    )


st.markdown("")

st.success(
    """
    🧠 Ensemble Learning

    MediPredict combines the predictions of multiple
    machine learning models to produce an ensemble-based
    result. This allows the application to compare
    different algorithms rather than relying on a single
    classifier.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "MediPredict • Academic Machine Learning Project • "
    "Smart Disease Prediction with Ensemble Learning"
)