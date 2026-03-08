import streamlit as st
import numpy as np
import pickle
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Title */
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #ff6b6b;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Section headers */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #ff6b6b;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.4rem 0;
    border-bottom: 1px solid #21262d;
    margin: 1.5rem 0 1rem 0;
}

/* Card */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

/* Result boxes */
.result-positive {
    background: linear-gradient(135deg, #3d0000 0%, #1a0000 100%);
    border: 1px solid #ff4444;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #003d1a 0%, #001a0d 100%);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.result-detail {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e;
    letter-spacing: 0.05em;
}

/* Divider */
hr { border-color: #21262d; }

/* Streamlit widget overrides */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-size: 0.8rem !important;
    color: #c9d1d9 !important;
    font-family: 'DM Sans', sans-serif !important;
}

div[data-testid="stButton"] > button {
    background: #ff6b6b;
    color: #0d1117;
    border: none;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    background: #ff4444;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(255,107,107,0.3);
}

.disclaimer {
    font-size: 0.7rem;
    color: #484f58;
    font-family: 'DM Mono', monospace;
    text-align: center;
    margin-top: 1.5rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫀 Heart Risk<br>Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Random Forest · Clinical Feature Analysis</div>', unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.warning("⚠️  No model file found. Place your trained `model.pkl` in the same directory. Running in demo mode.")

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # — Demographics —
    st.markdown('<div class="section-header">01 · Demographics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1)
    with col2:
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    # — Symptoms —
    st.markdown('<div class="section-header">02 · Symptoms</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            index=3,
            format_func=lambda x: {0: "0 – Typical Angina", 1: "1 – Atypical Angina",
                                    2: "2 – Non-Anginal", 3: "3 – Asymptomatic"}[x]
        )
    with col4:
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No")

    # — Vitals —
    st.markdown('<div class="section-header">03 · Vitals & Lab Results</div>', unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        trestbps = st.number_input("Resting BP (mm Hg)", min_value=50, max_value=250, value=140)
    with col6:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=250)
    with col7:
        fbs = st.selectbox("Fasting Blood Sugar >120", options=[0, 1],
                           format_func=lambda x: "Yes" if x == 1 else "No")

    # — Diagnostics —
    st.markdown('<div class="section-header">04 · Diagnostics</div>', unsafe_allow_html=True)
    col8, col9 = st.columns(2)
    with col8:
        restecg = st.selectbox(
            "Resting ECG Result",
            options=[0, 1, 2],
            index=1,
            format_func=lambda x: {0: "0 – Normal", 1: "1 – ST-T Wave", 2: "2 – LVH"}[x]
        )
    with col9:
        thalach = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=250, value=150)

    col10, col11 = st.columns(2)
    with col10:
        oldpeak = st.number_input("ST Depression (Exercise)", min_value=0.0, max_value=10.0,
                                  value=1.5, step=0.1, format="%.1f")
    with col11:
        slope = st.selectbox(
            "ST Segment Slope",
            options=[0, 1, 2],
            index=2,
            format_func=lambda x: {0: "0 – Upsloping", 1: "1 – Flat", 2: "2 – Downsloping"}[x]
        )

    col12, col13 = st.columns(2)
    with col12:
        ca = st.selectbox("Major Vessels (0–3)", options=[0, 1, 2, 3], index=1)
    with col13:
        thal = st.selectbox(
            "Thalassemia",
            options=[3.0, 6.0, 7.0],
            index=2,
            format_func=lambda x: {3.0: "3 – Normal", 6.0: "6 – Fixed Defect",
                                    7.0: "7 – Reversible Defect"}[x]
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("RUN PREDICTION →")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # -- Feature Engineering (must match notebook preprocessing) --
    age_x_chol      = age * chol
    bp_x_chol       = trestbps * chol
    hr_reserve      = 220 - age - thalach
    oldpeak_per_age = oldpeak / (age + 1)
    thalach_ratio   = thalach / (220 - age)

    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal,
                          age_x_chol, bp_x_chol, hr_reserve,
                          oldpeak_per_age, thalach_ratio]])

    if model is not None:
        prediction = model.predict(features)[0]
        try:
            proba = model.predict_proba(features)[0]
            confidence = proba[int(prediction)] * 100
            conf_text = f"Model confidence: {confidence:.1f}%"
        except Exception:
            conf_text = ""
    else:
        # Demo mode: simple heuristic
        risk_score = (age > 55) + (chol > 240) + (thalach < 140) + (oldpeak > 1) + (ca > 0)
        prediction = 1 if risk_score >= 3 else 0
        conf_text = "Demo mode · install model.pkl for real predictions"

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-title">⚠️ High Risk Detected</div>
            <p style="color:#ff9999;font-family:'DM Sans',sans-serif;margin:0.5rem 0;">
                The model predicts a <strong>positive indicator</strong> for heart disease.
            </p>
            <div class="result-detail">{conf_text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-title">✅ Low Risk</div>
            <p style="color:#86efac;font-family:'DM Sans',sans-serif;margin:0.5rem 0;">
                The model predicts <strong>no significant indicator</strong> of heart disease.
            </p>
            <div class="result-detail">{conf_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature summary
    with st.expander("View input summary"):
        labels = ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                  "Fasting BS", "Resting ECG", "Max HR", "Exang", "ST Depression",
                  "Slope", "CA", "Thal"]
        values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                  oldpeak, slope, ca, thal]
        for l, v in zip(labels, values):
            st.markdown(f"`{l}` → **{v}**")

st.markdown("""
<div class="disclaimer">
    ⚕ This tool is for educational and research purposes only.<br>
    It is not a substitute for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)