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
def render_risk_viz(risk_pct: float, is_high_risk: bool,
                    cp: int, chol: int, trestbps: int,
                    thalach: int, oldpeak: float, ca: int):
    """
    Radial gauge + organ impact cards. Pure CSS/HTML — no SVG paths.
    Uses a conic-gradient dial and animated bar cards per organ/metric.
    """
    # ── Palette ──────────────────────────────────────────────────
    if risk_pct < 30:
        c1, c2      = "#22c55e", "#16a34a"
        glow        = "rgba(34,197,94,0.35)"
        label_col   = "#86efac"
        tier_txt    = "LOW RISK"
        tier_icon   = "🟢"
    elif risk_pct < 60:
        c1, c2      = "#f59e0b", "#d97706"
        glow        = "rgba(245,158,11,0.35)"
        label_col   = "#fcd34d"
        tier_txt    = "MODERATE RISK"
        tier_icon   = "🟡"
    else:
        c1, c2      = "#ef4444", "#b91c1c"
        glow        = "rgba(239,68,68,0.4)"
        label_col   = "#fca5a5"
        tier_txt    = "HIGH RISK"
        tier_icon   = "🔴"

    # ── Gauge conic-gradient ──────────────────────────────────────
    # Dial sweeps 270° (from 135° to 405°). Map risk_pct → degrees.
    sweep_deg  = risk_pct / 100 * 270          # 0–270
    filled_end = 135 + sweep_deg               # start angle + sweep
    # conic-gradient: filled arc, then gap, then track
    conic = (
        f"conic-gradient("
        f"from 135deg, "
        f"{c1} 0deg, {c2} {sweep_deg:.1f}deg, "
        f"rgba(255,255,255,0.07) {sweep_deg:.1f}deg 270deg, "
        f"transparent 270deg"
        f")"
    )

    # ── Organ/metric cards ────────────────────────────────────────
    def organ_bar(icon, label, value_pct, note):
        if value_pct < 33:
            bar_c = "#22c55e"
            status = "Normal"
        elif value_pct < 66:
            bar_c = "#f59e0b"
            status = "Elevated"
        else:
            bar_c = "#ef4444"
            status = "High"
        return f"""
        <div class="organ-card">
          <div class="organ-top">
            <span class="organ-icon">{icon}</span>
            <span class="organ-label">{label}</span>
            <span class="organ-status" style="color:{bar_c};">{status}</span>
          </div>
          <div class="organ-track">
            <div class="organ-fill" style="width:{value_pct:.0f}%;background:{bar_c};
              box-shadow:0 0 8px {bar_c}88;"></div>
          </div>
          <div class="organ-note">{note}</div>
        </div>"""

    # Normalise each metric to 0–100 for the bar
    cp_pct       = cp / 3 * 100                                # 0=typical angina worst, 3=asymptomatic best → invert
    cp_pct       = (3 - cp) / 3 * 100                         # higher cp value = lower concern
    chol_pct     = max(0, min(100, (chol - 150) / (400 - 150) * 100))
    bp_pct       = max(0, min(100, (trestbps - 90) / (180 - 90) * 100))
    hr_pct       = max(0, min(100, (200 - thalach) / (200 - 60) * 100))
    st_pct       = max(0, min(100, oldpeak / 6.0 * 100))
    vessel_pct   = ca / 3 * 100

    cards_html = (
        organ_bar("🫀", "Heart Stress",   risk_pct,   f"Overall model risk score")
      + organ_bar("🩸", "Cholesterol",    chol_pct,   f"{chol} mg/dl")
      + organ_bar("💉", "Blood Pressure", bp_pct,     f"{trestbps} mm Hg resting")
      + organ_bar("⚡", "ST Depression",  st_pct,     f"{oldpeak:.1f} — exercise ECG")
      + organ_bar("💓", "Max Heart Rate", hr_pct,     f"{thalach} bpm (lower = more concern)")
      + organ_bar("🔬", "Vessel Load",    vessel_pct, f"{ca} major vessel(s) affected")
    )

    pulse_anim = """
      @keyframes pulse-ring {
        0%   { transform: scale(0.92); box-shadow: 0 0 0 0 rgba(239,68,68,0.7); }
        70%  { transform: scale(1);    box-shadow: 0 0 0 14px rgba(239,68,68,0); }
        100% { transform: scale(0.92); box-shadow: 0 0 0 0 rgba(239,68,68,0); }
      }
    """ if is_high_risk else ""

    pulse_style = "animation: pulse-ring 1s ease-out infinite;" if is_high_risk else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    background: #0e1118;
    color: #f1f5f9;
    font-family: 'DM Sans', sans-serif;
    padding: 24px 20px 28px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
  }}

  /* ── Gauge ── */
  .gauge-wrap {{
    position: relative;
    width: 200px;
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .gauge-ring {{
    position: absolute;
    inset: 0;
    border-radius: 50%;
    {conic}
    {pulse_style}
  }}
  /* Mask the centre to make a donut */
  .gauge-ring::after {{
    content: '';
    position: absolute;
    inset: 22px;
    border-radius: 50%;
    background: #0e1118;
  }}
  .gauge-center {{
    position: relative;
    z-index: 2;
    text-align: center;
  }}
  .gauge-pct {{
    font-family: 'Syne', sans-serif;
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    color: {label_col};
    filter: drop-shadow(0 0 12px {glow});
  }}
  .gauge-sign {{
    font-size: 20px;
    vertical-align: super;
    margin-left: 1px;
  }}
  .gauge-tier {{
    font-family: 'Syne', sans-serif;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {label_col};
    margin-top: 2px;
    opacity: 0.85;
  }}

  /* Tick marks on gauge */
  .gauge-ticks {{
    position: absolute;
    inset: -8px;
    border-radius: 50%;
  }}

  /* ── Organ cards ── */
  .cards-grid {{
    width: 100%;
    max-width: 440px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .organ-card {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 10px 14px;
    animation: fadeUp 0.5s ease both;
  }}
  .organ-card:nth-child(1) {{ animation-delay: 0.05s; }}
  .organ-card:nth-child(2) {{ animation-delay: 0.10s; }}
  .organ-card:nth-child(3) {{ animation-delay: 0.15s; }}
  .organ-card:nth-child(4) {{ animation-delay: 0.20s; }}
  .organ-card:nth-child(5) {{ animation-delay: 0.25s; }}
  .organ-card:nth-child(6) {{ animation-delay: 0.30s; }}

  .organ-top {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 7px;
  }}
  .organ-icon  {{ font-size: 15px; }}
  .organ-label {{
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: #cbd5e1;
    flex: 1;
  }}
  .organ-status {{
    font-family: 'Syne', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  .organ-track {{
    height: 4px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
  }}
  .organ-fill {{
    height: 100%;
    border-radius: 99px;
    animation: barGrow 1.1s cubic-bezier(.25,.1,.25,1) both;
  }}
  .organ-note {{
    font-size: 10px;
    color: #475569;
    margin-top: 5px;
    letter-spacing: 0.02em;
  }}

  @keyframes barGrow {{
    from {{ width: 0 !important; }}
  }}
  @keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(8px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
  {pulse_anim}

  /* ── Scale labels ── */
  .scale-row {{
    display: flex;
    justify-content: space-between;
    width: 200px;
    margin-top: -12px;
  }}
  .scale-row span {{
    font-size: 9px;
    color: #475569;
    letter-spacing: 0.06em;
  }}
</style>
</head>
<body>

  <!-- Radial gauge -->
  <div class="gauge-wrap">
    <div class="gauge-ring"></div>
    <div class="gauge-center">
      <div class="gauge-pct">{risk_pct:.0f}<span class="gauge-sign">%</span></div>
      <div class="gauge-tier">{tier_icon} {tier_txt}</div>
    </div>
  </div>
  <div class="scale-row">
    <span>0%</span><span>RISK SCORE</span><span>100%</span>
  </div>

  <!-- Organ impact grid -->
  <div class="cards-grid">
    {cards_html}
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