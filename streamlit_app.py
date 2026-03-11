import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pickle
import os
import base64

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
st.markdown('<div class="main-subtitle">Random Forest · Clinical Feature Analysis <br/> Contributors: Foysal · Ashraf · Tasnimul</div>', unsafe_allow_html=True)

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
def render_human_risk(risk_pct: float, is_high_risk: bool):
    """
    Renders via components.html() (iframe) so SVG defs/clipPath are never sanitized.
    """
    if risk_pct < 30:
        fill_top, fill_bot = "#22c55e", "#16a34a"
        glow   = "rgba(34,197,94,0.4)"
        label_color = "#86efac"
        risk_label  = "LOW RISK"
    elif risk_pct < 60:
        fill_top, fill_bot = "#f59e0b", "#d97706"
        glow   = "rgba(245,158,11,0.4)"
        label_color = "#fcd34d"
        risk_label  = "MODERATE RISK"
    else:
        fill_top, fill_bot = "#ef4444", "#b91c1c"
        glow   = "rgba(239,68,68,0.45)"
        label_color = "#fca5a5"
        risk_label  = "HIGH RISK"

    H = 220.0
    fill_y = H * (1 - risk_pct / 100)
    fill_h = H * (risk_pct / 100)
    fill_y_start = min(fill_y + 30, H)
    fill_h_start = max(fill_h - 30, 0)

    pulse_dur = "0.75s" if is_high_risk else "1.5s"

    # Single compound path for the entire human silhouette (used in clipPath)
    # Drawn in one <path> so clipPath works reliably across all browsers
    BODY_PATH = (
        # Head
        "M50,4 C43,4 38,10 38,18 C38,26 43,32 50,32 C57,32 62,26 62,18 C62,10 57,4 50,4Z "
        # Neck
        "M45,32 L45,40 L55,40 L55,32Z "
        # Torso + shoulders
        "M28,40 C22,44 18,56 20,72 L22,84 L78,84 L80,72 C82,56 78,44 72,40 "
        "C66,37 58,35 50,35 C42,35 34,37 28,40Z "
        # Left upper arm
        "M20,44 C12,50 8,66 10,80 C11,86 14,89 18,87 C20,86 22,83 22,78 "
        "L24,60 C25,52 24,47 20,44Z "
        # Right upper arm
        "M80,44 C88,50 92,66 90,80 C89,86 86,89 82,87 C80,86 78,83 78,78 "
        "L76,60 C75,52 76,47 80,44Z "
        # Left forearm + hand
        "M10,80 C8,94 8,108 10,120 C11,126 14,128 17,126 C19,124 20,120 19,114 "
        "L18,92 C16,86 13,82 10,80Z "
        "M7,122 C5,126 6,132 10,133 C14,134 17,130 16,126Z "
        # Right forearm + hand
        "M90,80 C92,94 92,108 90,120 C89,126 86,128 83,126 C81,124 80,120 81,114 "
        "L82,92 C84,86 87,82 90,80Z "
        "M93,122 C95,126 94,132 90,133 C86,134 83,130 84,126Z "
        # Hips
        "M22,84 C20,94 24,100 30,102 L70,102 C76,100 80,94 78,84Z "
        # Left thigh
        "M30,102 C26,116 26,134 28,150 C29,157 34,158 38,156 C42,154 43,149 42,142 "
        "L40,118 C38,108 35,103 30,102Z "
        # Right thigh
        "M70,102 C74,116 74,134 72,150 C71,157 66,158 62,156 C58,154 57,149 58,142 "
        "L60,118 C62,108 65,103 70,102Z "
        # Left shin + foot
        "M28,150 C26,166 26,182 28,196 C29,202 33,204 37,202 C41,200 42,195 41,188 "
        "L40,162 C39,155 35,151 28,150Z "
        "M22,200 C20,204 24,208 30,207 C36,206 39,202 37,199Z "
        # Right shin + foot
        "M72,150 C74,166 74,182 72,196 C71,202 67,204 63,202 C59,200 58,195 59,188 "
        "L60,162 C61,155 65,151 72,150Z "
        "M78,200 C80,204 76,208 70,207 C64,206 61,202 63,199Z"
    )

    pulse_svg = ""
    if is_high_risk:
        pulse_svg = f"""
        <circle cx="50" cy="62" r="5" fill="{fill_top}" opacity="0.8">
          <animate attributeName="r" values="3;10;3" dur="{pulse_dur}" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0.8;0;0.8" dur="{pulse_dur}" repeatCount="indefinite"/>
        </circle>
        <circle cx="50" cy="62" r="3" fill="white" opacity="0.95">
          <animate attributeName="opacity" values="1;0.2;1" dur="{pulse_dur}" repeatCount="indefinite"/>
        </circle>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: #111520;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 28px 20px 24px;
    font-family: 'DM Sans', sans-serif;
    gap: 10px;
  }}
  .risk-label {{
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {label_color};
  }}
  .risk-percent {{
    font-family: 'Syne', sans-serif;
    font-size: 58px;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    color: {label_color};
  }}
  .pct-sign {{
    font-size: 28px;
    font-weight: 600;
    vertical-align: super;
    margin-left: 2px;
  }}
  .body-wrap {{
    filter: drop-shadow(0 0 20px {glow});
  }}
  .meter-wrap {{
    width: 240px;
    margin-top: 6px;
  }}
  .meter-track {{
    width: 100%;
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
  }}
  .meter-fill {{
    height: 100%;
    width: {risk_pct}%;
    background: linear-gradient(90deg, {fill_bot}, {fill_top});
    border-radius: 99px;
    animation: barGrow 1.3s cubic-bezier(.25,.1,.25,1) both;
  }}
  @keyframes barGrow {{
    from {{ width: 0%; }}
    to   {{ width: {risk_pct}%; }}
  }}
  .meter-labels {{
    display: flex;
    justify-content: space-between;
    margin-top: 5px;
    font-size: 10px;
    color: #64748b;
    letter-spacing: 0.05em;
  }}
</style>
</head>
<body>
  <div class="risk-label">{risk_label}</div>
  <div class="risk-percent">{risk_pct:.1f}<span class="pct-sign">%</span></div>

  <div class="body-wrap">
    <svg viewBox="0 0 100 220" width="170" height="374"
         xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{fill_top}" stop-opacity="1"/>
          <stop offset="100%" stop-color="{fill_bot}" stop-opacity="0.9"/>
        </linearGradient>

        <clipPath id="bodyClip">
          <path d="{BODY_PATH}"/>
        </clipPath>

        <linearGradient id="shimmerGrad" x1="0" y1="0" x2="1" y2="0"
                        gradientUnits="userSpaceOnUse">
          <stop offset="0%"   stop-color="white" stop-opacity="0"/>
          <stop offset="50%"  stop-color="white" stop-opacity="0.12"/>
          <stop offset="100%" stop-color="white" stop-opacity="0"/>
          <animateTransform attributeName="gradientTransform" type="translate"
            values="-100 0; 200 0" dur="2.5s" repeatCount="indefinite"/>
        </linearGradient>
      </defs>

      <!-- Ghost body (dim outline bg) -->
      <g clip-path="url(#bodyClip)">
        <rect x="0" y="0" width="100" height="220" fill="rgba(255,255,255,0.055)"/>
      </g>

      <!-- Animated fill rising from bottom -->
      <g clip-path="url(#bodyClip)">
        <rect x="0" y="{fill_y:.2f}" width="100" height="{fill_h:.2f}"
              fill="url(#fillGrad)">
          <animate attributeName="y"
            from="{fill_y_start:.2f}" to="{fill_y:.2f}"
            dur="1.3s" fill="freeze"
            calcMode="spline" keySplines="0.25 0.1 0.25 1"/>
          <animate attributeName="height"
            from="{fill_h_start:.2f}" to="{fill_h:.2f}"
            dur="1.3s" fill="freeze"
            calcMode="spline" keySplines="0.25 0.1 0.25 1"/>
        </rect>
        <!-- Shimmer sweep -->
        <rect x="0" y="{fill_y:.2f}" width="100" height="{fill_h:.2f}"
              fill="url(#shimmerGrad)"/>
      </g>

      <!-- Outline stroke on top -->
      <path d="{BODY_PATH}"
            fill="none"
            stroke="rgba(255,255,255,0.2)"
            stroke-width="0.7"
            stroke-linejoin="round"/>

      <!-- Heart pulse dot (high risk only) -->
      {pulse_svg}
    </svg>
  </div>

  <div class="meter-wrap">
    <div class="meter-track">
      <div class="meter-fill"></div>
    </div>
    <div class="meter-labels">
      <span>0%</span><span>50%</span><span>100%</span>
    </div>
  </div>

</body>
</html>"""

    return html


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
            risk_pct   = float(proba[1]) * 100          # probability of class=1
            conf_text  = f"Model confidence: {proba[int(prediction)]*100:.1f}%"
        except Exception:
            risk_pct   = 75.0 if prediction == 1 else 25.0
            conf_text  = ""
    else:
        # Demo heuristic
        risk_score = (age > 55) + (chol > 240) + (thalach < 140) + (oldpeak > 1) + (ca > 0)
        prediction = 1 if risk_score >= 3 else 0
        risk_pct   = min(risk_score / 5 * 100, 95)
        conf_text  = "Demo mode · install model.pkl for real predictions"

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Human body visualizer ──
    viz_html = render_human_risk(risk_pct, is_high_risk=(prediction == 1))
    st.markdown(viz_html, unsafe_allow_html=True)

    # ── Text result card ──
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