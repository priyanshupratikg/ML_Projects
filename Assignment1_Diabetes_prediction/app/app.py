from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DiabDetect | Vajra",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# Project structure:
#
# Diabetes_Prediction/
# ├── app/
# │   └── app.py
# ├── models/
# │   ├── final_random_forest.pkl
# │   ├── standard_scaler.pkl
# │   └── median_imputer.pkl
# └── ...

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "final_random_forest.pkl"
SCALER_PATH = MODEL_DIR / "standard_scaler.pkl"
IMPUTER_PATH = MODEL_DIR / "median_imputer.pkl"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
"""<style>
/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(220, 25, 105, 0.16),
            transparent 30%
        ),
        radial-gradient(
            circle at 10% 90%,
            rgba(20, 100, 150, 0.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #061522 0%,
            #081b2a 50%,
            #101326 100%
        );
    color: #ffffff;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    background: transparent;
}

footer {
    visibility: hidden;
}


/* =========================================================
   MAIN CONTENT
   ========================================================= */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #061522 0%,
            #07131f 100%
        );
    border-right: 1px solid rgba(100, 180, 220, 0.15);
}

.sidebar-brand {
    text-align: center;
    padding: 8px 0 25px 0;
    border-bottom: 1px solid rgba(130, 190, 220, 0.12);
    margin-bottom: 25px;
}

.sidebar-logo {
    font-size: 40px;
    line-height: 1;
}

.sidebar-name {
    color: #ffffff;
    font-size: 17px;
    font-weight: 800;
    margin-top: 7px;
}

.sidebar-by {
    color: #ff3c8d;
    font-size: 8px;
    font-weight: 900;
    letter-spacing: 3px;
}

.sidebar-heading {
    color: #ffffff;
    font-size: 13px;
    font-weight: 800;
    margin-top: 20px;
    margin-bottom: 8px;
}

.sidebar-text {
    color: #9ab4c4;
    font-size: 11px;
    line-height: 1.7;
}

.sidebar-item {
    color: #8fa9b9;
    font-size: 10px;
    margin: 6px 0;
}

.warning {
    margin-top: 25px;
    padding: 14px;
    border-radius: 10px;
    background: rgba(255, 193, 7, 0.06);
    border: 1px solid rgba(255, 193, 7, 0.22);
    color: #aebfc9;
    font-size: 9px;
    line-height: 1.7;
}


/* =========================================================
   HERO
   ========================================================= */

.hero-container {
    text-align: center;
    padding: 15px 10px 42px 10px;
}

.hero-icon {
    font-size: 58px;
    line-height: 1;
    margin-bottom: 10px;
}

.hero-title {
    color: #ffffff;
    font-size: 48px;
    font-weight: 900;
    letter-spacing: -2px;
    margin: 0;
}

.hero-by {
    color: #ff3c8d;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 5px;
    margin-top: 5px;
}

.hero-description {
    max-width: 680px;
    margin: 18px auto 0 auto;
    color: #8daabd;
    font-size: 11px;
    line-height: 1.8;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-label {
    color: #ff3c8d;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 3px;
}

.section-title {
    color: #ffffff;
    font-size: 25px;
    font-weight: 850;
}

.section-subtitle {
    color: #819dad;
    font-size: 10px;
}


/* =========================================================
   FORM CARD
   ========================================================= */

.form-card {
    padding: 25px;
    border-radius: 18px;
    background:
        linear-gradient(
            145deg,
            rgba(18, 49, 68, 0.95),
            rgba(7, 28, 42, 0.98)
        );
    border: 1px solid rgba(87, 174, 220, 0.25);
    box-shadow: 0 20px 55px rgba(0, 0, 0, 0.28);
}


/* =========================================================
   INPUTS
   ========================================================= */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background-color: #eef2f7 !important;
    color: #17232d !important;
    border-radius: 8px !important;
}

[data-baseweb="select"] {
    background-color: #eef2f7 !important;
    border-radius: 8px !important;
}

[data-baseweb="select"] * {
    color: #17232d !important;
}

[data-testid="stNumberInput"] button {
    background-color: #eef2f7 !important;
    color: #4c5963 !important;
    border: none !important;
}

[data-testid="stNumberInput"] button:hover {
    color: #ed2379 !important;
}

label {
    color: #b9d8e9 !important;
    font-size: 11px !important;
}

.range-text {
    color: #5f879c;
    font-size: 8px;
    margin-top: -8px;
    margin-bottom: 12px;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;
    min-height: 48px;
    border: none !important;
    border-radius: 10px !important;
    background:
        linear-gradient(
            90deg,
            #f43b8d,
            #df176c
        ) !important;
    color: #ffffff !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    box-shadow:
        0 10px 32px rgba(238, 30, 116, 0.27);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 15px 42px rgba(238, 30, 116, 0.42);
}


/* =========================================================
   RESULT
   ========================================================= */

.result-card {
    margin-top: 28px;
    padding: 35px 25px;
    text-align: center;
    border-radius: 18px;
    background:
        linear-gradient(
            145deg,
            rgba(18, 52, 70, 0.97),
            rgba(7, 28, 42, 0.98)
        );
    border: 1px solid rgba(87, 174, 220, 0.25);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.result-label {
    color: #7195a8;
    font-size: 9px;
    font-weight: 900;
    letter-spacing: 3px;
}

.result-name {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
}

.result-title {
    color: #ff3b8d;
    font-size: 30px;
    font-weight: 900;
    margin-top: 12px;
}

.result-probability {
    color: #b4cad6;
    font-size: 13px;
    margin-top: 8px;
}

.result-note {
    max-width: 700px;
    margin: 18px auto 0 auto;
    color: #7895a6;
    font-size: 9px;
    line-height: 1.8;
}


/* =========================================================
   MOBILE / PHONE
   ========================================================= */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    .hero-container {
        padding-bottom: 25px;
    }

    .hero-icon {
        font-size: 45px;
    }

    .hero-title {
        font-size: 36px;
        letter-spacing: -1px;
    }

    .hero-by {
        font-size: 8px;
        letter-spacing: 4px;
    }

    .hero-description {
        font-size: 10px;
        padding: 0 8px;
    }

    .section-title {
        font-size: 21px;
    }

    .form-card {
        padding: 17px;
        border-radius: 14px;
    }

    .result-title {
        font-size: 24px;
    }
}


@media (max-width: 450px) {

    .hero-title {
        font-size: 31px;
    }

    .hero-icon {
        font-size: 40px;
    }

    .form-card {
        padding: 14px;
    }
}

</style>""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    trained_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    return trained_model, scaler, imputer


# ============================================================
# MODEL CHECK
# ============================================================

if not MODEL_PATH.exists():

    st.error("❌ final_random_forest.pkl was not found.")

    st.write("Expected location:")
    st.code(str(MODEL_PATH))

    st.stop()


if not SCALER_PATH.exists():

    st.error("❌ standard_scaler.pkl was not found.")

    st.write("Expected location:")
    st.code(str(SCALER_PATH))

    st.stop()


if not IMPUTER_PATH.exists():

    st.error("❌ median_imputer.pkl was not found.")

    st.write("Expected location:")
    st.code(str(IMPUTER_PATH))

    st.stop()


try:

    model, scaler, imputer = load_models()

except Exception as e:

    st.error("❌ Unable to load the trained model files.")

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-brand">
    <div class="sidebar-logo">💧</div>
    <div class="sidebar-name">DiabDetect</div>
    <div class="sidebar-by">BY VAJRA</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">🔬 About DiabDetect</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-text">DiabDetect is a machine-learning based diabetes risk prediction system developed as an academic project.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">⚙️ Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-item"><strong>Tuned Random Forest Classifier</strong></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-item">300 estimators</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">🧪 Preprocessing</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-item">Suspicious Zero → Missing Value</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-item">Median Imputation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-item">Standard Scaling</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">📊 Output</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-text">The system provides a predicted diabetes class together with the model probability estimate.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="warning">
⚠️ <strong>Educational Use Only</strong>
<br><br>
This application is intended for academic and demonstration purposes.
The prediction is not a medical diagnosis and should not replace
professional medical advice.
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero-container">
    <div class="hero-icon">💧</div>
    <div class="hero-title">DiabDetect</div>
    <div class="hero-by">BY VAJRA</div>
    <div class="hero-description">
        Intelligent diabetes risk prediction powered by machine learning.
        Enter the patient's clinical information below to generate an
        AI-assisted prediction.
    </div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-label">CLINICAL ASSESSMENT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Enter the patient\'s measured clinical values accurately. Do not enter estimated or assumed values.</div>',
    unsafe_allow_html=True
)


# ============================================================
# FORM
# ============================================================

st.markdown(
    '<div class="form-card">',
    unsafe_allow_html=True
)

st.markdown(
    "**👤 Patient Details**"
)

st.markdown(
    "---"
)

name_col, gender_col = st.columns(
    [2, 1]
)


with name_col:

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient's name"
    )


with gender_col:

    gender = st.selectbox(
        "Gender",
        [
            "Select gender",
            "Male",
            "Female"
        ]
    )


st.markdown(
    "**🧪 Clinical Measurements**"
)

st.markdown(
    "---"
)


# ============================================================
# GLUCOSE / INSULIN
# ============================================================

col1, col2 = st.columns(2)


with col1:

    glucose = st.number_input(
        "Glucose",
        min_value=44.0,
        max_value=199.0,
        value=None,
        step=1.0,
        format="%.0f",
        placeholder="Enter glucose level"
    )

    st.markdown(
        '<div class="range-text">Dataset range: 44–199 mg/dL</div>',
        unsafe_allow_html=True
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=14.0,
        max_value=846.0,
        value=None,
        step=1.0,
        format="%.0f",
        placeholder="Enter insulin level"
    )

    st.markdown(
        '<div class="range-text">Dataset range: 14–846 μU/mL</div>',
        unsafe_allow_html=True
    )


# ============================================================
# BLOOD PRESSURE / BMI
# ============================================================

with col1:

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=24.0,
        max_value=122.0,
        value=None,
        step=1.0,
        format="%.0f",
        placeholder="Enter blood pressure"
    )

    st.markdown(
        '<div class="range-text">Dataset range: 24–122 mmHg</div>',
        unsafe_allow_html=True
    )


with col2:

    bmi = st.number_input(
        "BMI",
        min_value=18.2,
        max_value=67.1,
        value=None,
        step=0.1,
        format="%.1f",
        placeholder="Enter BMI"
    )

    st.markdown(
        '<div class="range-text">Dataset range: 18.2–67.1 kg/m²</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SKIN THICKNESS / AGE
# ============================================================

with col1:

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=7.0,
        max_value=99.0,
        value=None,
        step=1.0,
        format="%.0f",
        placeholder="Enter skin thickness"
    )

    st.markdown(
        '<div class="range-text">Dataset range: 7–99 mm</div>',
        unsafe_allow_html=True
    )


with col2:

    age = st.number_input(
        "Age",
        min_value=21,
        max_value=81,
        value=None,
        step=1,
        format="%d",
        placeholder="Enter age"
    )

    st.markdown(
        '<div class="range-text">Dataset range: 21–81 years</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PREGNANCIES
# ============================================================

pregnancies = 0

if gender == "Female":

    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=17,
        value=None,
        step=1,
        format="%d",
        placeholder="Enter number of pregnancies"
    )

    st.markdown(
        '<div class="range-text">Dataset range: 0–17 pregnancies</div>',
        unsafe_allow_html=True
    )


elif gender == "Male":

    pregnancies = 0


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# ANALYZE
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button(
    "💧  ANALYZE DIABETES RISK",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    missing = []


    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    if not patient_name.strip():
        missing.append("Patient Name")

    if gender == "Select gender":
        missing.append("Gender")


    # --------------------------------------------------------
    # CLINICAL VALUES
    # --------------------------------------------------------

    if glucose is None:
        missing.append("Glucose")

    if insulin is None:
        missing.append("Insulin")

    if blood_pressure is None:
        missing.append("Blood Pressure")

    if bmi is None:
        missing.append("BMI")

    if skin_thickness is None:
        missing.append("Skin Thickness")

    if age is None:
        missing.append("Age")


    if gender == "Female" and pregnancies is None:
        missing.append("Number of Pregnancies")


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if missing:

        st.warning(
            "Please enter the following: "
            + ", ".join(missing)
        )

        st.stop()


    # ========================================================
    # MODEL INPUT
    # ========================================================

    input_data = pd.DataFrame(
        {
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],

            # DPF is intentionally hidden.
            # The trained median imputer handles it.
            "DiabetesPedigreeFunction": [np.nan],

            "Age": [age]
        }
    )


    try:

        # ====================================================
        # MEDIAN IMPUTATION
        # ====================================================

        input_imputed = pd.DataFrame(
            imputer.transform(input_data),
            columns=input_data.columns
        )


        # ====================================================
        # STANDARD SCALING
        # ====================================================

        input_scaled = scaler.transform(
            input_imputed
        )


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0]


        diabetes_probability = probability[1] * 100


        # ====================================================
        # RESULT
        # ====================================================

        if prediction == 1:

            result = "Higher Diabetes Risk"

        else:

            result = "Lower Diabetes Risk"


        st.markdown(
            f"""
<div class="result-card">

<div class="result-label">
PREDICTION RESULT
</div>

<div class="result-name">
{patient_name}
</div>

<div class="result-title">
{result}
</div>

<div class="result-probability">
Estimated diabetes probability:
<strong>{diabetes_probability:.2f}%</strong>
</div>

<div class="result-note">

Prediction generated using the trained Random Forest classifier.

<br><br>

Diabetes Pedigree Function is not requested from the user.
It is handled automatically by the saved median-imputation
preprocessing step.

<br><br>

This result is intended for educational and demonstration
purposes only and is not a medical diagnosis.

</div>

</div>
""",
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            "❌ Prediction could not be completed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "DiabDetect • Diabetes Risk Prediction System • Developed by Vajra"
)