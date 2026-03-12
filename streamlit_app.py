import time
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pickle
import os
import base64
from report import generate_report
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
with open("style.css") as f:
    st.html(f"<style>{f.read()}</style>")

# ── Local Video Function ──────────────────────────────────────────────────
def get_video_base64(video_path):
    with open(video_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

video_file = "heart_loop.mp4"

if os.path.exists(video_file):
    bin_str = get_video_base64(video_file)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-top: -20px;">
            <video style="width: 100%; height: auto; display: block;" autoplay loop muted playsinline>
                <source src="data:video/mp4;base64,{bin_str}" type="video/mp4">
            </video>
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Random Forest <br/> Clinical Feature Analysis <br/> Contributors: <br/> Foysal · Ashraf · Tasnimul</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-header">01 · Demographics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55, step=1)
    with col2:
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    st.markdown('<div class="section-header">02 · Symptoms</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3], index=3,
            format_func=lambda x: {0: "0 – Typical Angina", 1: "1 – Atypical Angina",
                                    2: "2 – Non-Anginal", 3: "3 – Asymptomatic"}[x]
        )
    with col4:
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown('<div class="section-header">03 · Vitals & Lab Results</div>', unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        trestbps = st.number_input("Resting BP (mm Hg)", min_value=50, max_value=250, value=140)
    with col6:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=250)
    with col7:
        fbs = st.selectbox("Fasting Blood Sugar >120", options=[0, 1],
                           format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown('<div class="section-header">04 · Diagnostics</div>', unsafe_allow_html=True)
    col8, col9 = st.columns(2)
    with col8:
        restecg = st.selectbox(
            "Resting ECG Result", options=[0, 1, 2], index=1,
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
            "ST Segment Slope", options=[0, 1, 2], index=2,
            format_func=lambda x: {0: "0 – Upsloping", 1: "1 – Flat", 2: "2 – Downsloping"}[x]
        )

    col12, col13 = st.columns(2)
    with col12:
        ca = st.selectbox("Major Vessels (0–3)", options=[0, 1, 2, 3], index=1)
    with col13:
        thal = st.selectbox(
            "Thalassemia", options=[3.0, 6.0, 7.0], index=2,
            format_func=lambda x: {3.0: "3 – Normal", 6.0: "6 – Fixed Defect",
                                    7.0: "7 – Reversible Defect"}[x]
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("RUN PREDICTION →")

# ── Human Body Risk Visualizer ────────────────────────────────────────────────
def render_risk_viz(risk_pct: float, is_high_risk: bool,
                    cp: int, chol: int, trestbps: int,
                    thalach: int, oldpeak: float, ca: int):
    """
    SVG donut gauge + organ impact cards.
    No conic-gradient — uses stroke-dasharray on a <circle> so it works everywhere.
    All CSS classes live in style.css; only dynamic values are inline.
    """
    # ── Palette ──────────────────────────────────────────────────
    if risk_pct < 30:
        color     = "#22c55e"
        glow      = "rgba(34,197,94,0.5)"
        label_col = "#86efac"
        tier_txt  = "LOW RISK"
        tier_icon = "🟢"
    elif risk_pct < 60:
        color     = "#f59e0b"
        glow      = "rgba(245,158,11,0.5)"
        label_col = "#fcd34d"
        tier_txt  = "MODERATE RISK"
        tier_icon = "🟡"
    else:
        color     = "#ef4444"
        glow      = "rgba(239,68,68,0.55)"
        label_col = "#fca5a5"
        tier_txt  = "HIGH RISK"
        tier_icon = "🔴"

    # ── SVG donut maths ───────────────────────────────────────────
    # viewBox="0 0 120 120", centre=60,60, radius=52, strokeWidth=10
    r          = 52
    cx = cy    = 60
    circumf    = 2 * 3.14159 * r          # ≈ 326.7
    filled     = circumf * risk_pct / 100
    gap        = circumf - filled
    # rotate so arc starts at 12 o'clock (top): -90deg transform
    pulse_filter = f'filter="url(#glow)"' if is_high_risk else ''

    # ── Organ/metric cards ────────────────────────────────────────
    def organ_bar(icon, label, value_pct, note):
        bar_c  = "#22c55e" if value_pct < 33 else ("#f59e0b" if value_pct < 66 else "#ef4444")
        status = "Normal"  if value_pct < 33 else ("Elevated" if value_pct < 66 else "High")
        return (
            f'<div class="organ-card">'
            f'<div class="organ-top">'
            f'<span class="organ-icon">{icon}</span>'
            f'<span class="organ-label">{label}</span>'
            f'<span class="organ-status" style="color:{bar_c};">{status}</span>'
            f'</div>'
            f'<div class="organ-track">'
            f'<div class="organ-fill" style="width:{value_pct:.0f}%;background:{bar_c};'
            f'box-shadow:0 0 8px {bar_c}88;"></div>'
            f'</div>'
            f'<div class="organ-note">{note}</div>'
            f'</div>'
        )

    chol_pct   = max(0, min(100, (chol - 150)    / 250 * 100))
    bp_pct     = max(0, min(100, (trestbps - 90) / 90  * 100))
    hr_pct     = max(0, min(100, (200 - thalach) / 140 * 100))
    st_pct     = max(0, min(100,  oldpeak        / 6.0 * 100))
    vessel_pct = ca / 3 * 100

    cards = (
        organ_bar("🫀", "Heart Stress",   risk_pct,   "Overall model risk score")
      + organ_bar("🩸", "Cholesterol",    chol_pct,   f"{chol} mg/dl")
      + organ_bar("💉", "Blood Pressure", bp_pct,     f"{trestbps} mm Hg resting")
      + organ_bar("⚡", "ST Depression",  st_pct,     f"{oldpeak:.1f} — exercise ECG")
      + organ_bar("💓", "Max Heart Rate", hr_pct,     f"{thalach} bpm (lower = more concern)")
      + organ_bar("🔬", "Vessel Load",    vessel_pct, f"{ca} major vessel(s) affected")
    )

    return f"""
<div class="viz-wrapper">

  <!-- SVG donut gauge -->
  <svg class="donut-svg" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <!-- Track ring -->
    <circle cx="{cx}" cy="{cy}" r="{r}"
            fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="10"/>
    <!-- Filled arc -->
    <circle cx="{cx}" cy="{cy}" r="{r}"
            fill="none"
            stroke="{color}"
            stroke-width="10"
            stroke-linecap="round"
            stroke-dasharray="{filled:.2f} {gap:.2f}"
            transform="rotate(-90 {cx} {cy})"
            style="filter:drop-shadow(0 0 6px {glow});"
            {pulse_filter}/>
    <!-- Centre text: percentage -->
    <text x="{cx}" y="{cy - 8}" text-anchor="middle"
          font-family="Syne,sans-serif" font-size="20" font-weight="800"
          fill="{label_col}">{risk_pct:.1f}%</text>
    <!-- Centre text: tier label -->
    <text x="{cx}" y="{cy + 10}" text-anchor="middle"
          font-family="Syne,sans-serif" font-size="7" font-weight="700"
          letter-spacing="1.5" fill="{label_col}" opacity="0.8">{tier_icon} {tier_txt}</text>
  </svg>

  <!-- Organ impact cards -->
  <div class="cards-grid">
    {cards}
  </div>

</div>"""


# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
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
        prediction  = model.predict(features)[0]
        try:
            proba      = model.predict_proba(features)[0]
            risk_pct   = float(proba[1]) * 100
        except Exception:
            risk_pct   = 75.0 if prediction == 1 else 25.0
    else:
        # Demo heuristic
        risk_score = (age > 55) + (chol > 240) + (thalach < 140) + (oldpeak > 1) + (ca > 0)
        prediction = 1 if risk_score >= 3 else 0
        risk_pct   = min(risk_score / 5 * 100, 95)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Loading animation ──
    loading = st.empty()
    loading.markdown("""
<div class="loading-wrap">
  <div class="loading-ring">
    <div class="loading-pulse"></div>
    <svg class="loading-spinner" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
      <circle cx="60" cy="60" r="52" fill="none"
              stroke="rgba(255,255,255,0.06)" stroke-width="10"/>
      <circle cx="60" cy="60" r="52" fill="none"
              stroke="#ef4444" stroke-width="10"
              stroke-linecap="round"
              stroke-dasharray="80 246"
              transform="rotate(-90 60 60)"
              class="loading-arc"/>
    </svg>
  </div>
  <div class="loading-text">Analysing clinical data<span class="loading-dots"><span>.</span><span>.</span><span>.</span></span></div>
</div>
""", unsafe_allow_html=True)

    time.sleep(1.2)
    loading.empty()
    viz_html = render_risk_viz(
        risk_pct, is_high_risk=(prediction == 1),
        cp=cp, chol=chol, trestbps=trestbps,
        thalach=thalach, oldpeak=oldpeak, ca=ca
    )
    st.markdown(viz_html, unsafe_allow_html=True)

    # ── PDF Report download ──
    patient_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal, 'prediction': int(prediction)
    }
    pdf_bytes = generate_report(patient_data, risk_pct)
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="📄  Download Full PDF Report",
        data=pdf_bytes,
        file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    with st.expander("View input summary"):
        labels = ["Age","Sex","Chest Pain Type","Resting BP","Cholesterol",
                  "Fasting BS","Resting ECG","Max HR","Exang","ST Depression",
                  "Slope","CA","Thal"]
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