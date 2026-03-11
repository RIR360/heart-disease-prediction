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
    Renders an SVG human silhouette filled from bottom to top based on risk_pct (0–100).
    Uses a clipPath so the fill color reveals the body shape progressively.
    """
    # Color palette based on risk level
    if risk_pct < 30:
        fill_color_top   = "#22c55e"   # green
        fill_color_bot   = "#16a34a"
        glow_color       = "rgba(34,197,94,0.35)"
        label_color      = "#86efac"
        risk_label       = "LOW RISK"
    elif risk_pct < 60:
        fill_color_top   = "#f59e0b"   # amber
        fill_color_bot   = "#d97706"
        glow_color       = "rgba(245,158,11,0.35)"
        label_color      = "#fcd34d"
        risk_label       = "MODERATE RISK"
    else:
        fill_color_top   = "#ef4444"   # red
        fill_color_bot   = "#b91c1c"
        glow_color       = "rgba(239,68,68,0.4)"
        label_color      = "#fca5a5"
        risk_label       = "HIGH RISK"

    # SVG viewBox is 0 0 100 220 (100 wide, 220 tall)
    # Fill rect starts from bottom: y = 220*(1 - risk_pct/100)
    total_height = 220
    fill_y = total_height * (1 - risk_pct / 100)
    fill_h = total_height * (risk_pct / 100)

    # Pulse animation speed based on risk
    pulse_dur = "0.7s" if is_high_risk else "1.4s"

    html = f"""
    <div class="risk-viz-wrapper">

      <!-- Risk label above -->
      <div class="risk-label" style="color:{label_color};">{risk_label}</div>

      <!-- Percentage counter -->
      <div class="risk-percent" style="color:{label_color};">{risk_pct:.1f}<span class="pct-sign">%</span></div>

      <!-- SVG Body -->
      <div class="svg-body-container" style="filter: drop-shadow(0 0 18px {glow_color});">
        <svg viewBox="0 0 100 220" xmlns="http://www.w3.org/2000/svg" class="body-svg">
          <defs>
            <!-- Gradient for fill -->
            <linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="{fill_color_top}" stop-opacity="1"/>
              <stop offset="100%" stop-color="{fill_color_bot}" stop-opacity="0.85"/>
            </linearGradient>

            <!-- ClipPath = the human body silhouette -->
            <clipPath id="bodyClip">
              <!-- HEAD -->
              <ellipse cx="50" cy="18" rx="12" ry="14"/>
              <!-- NECK -->
              <rect x="44" y="30" width="12" height="8" rx="3"/>
              <!-- TORSO -->
              <path d="M28,38 Q22,55 24,80 L76,80 Q78,55 72,38 Q62,34 50,34 Q38,34 28,38Z"/>
              <!-- LEFT ARM -->
              <path d="M24,40 Q14,50 12,75 Q11,82 15,84 Q19,86 21,80 L26,58 Q28,48 28,42Z"/>
              <!-- RIGHT ARM -->
              <path d="M76,40 Q86,50 88,75 Q89,82 85,84 Q81,86 79,80 L74,58 Q72,48 72,42Z"/>
              <!-- LEFT FOREARM + HAND -->
              <path d="M15,84 Q11,100 10,115 Q9,122 12,124 Q14,126 16,122 L20,106 Q21,96 21,88Z"/>
              <ellipse cx="11" cy="126" rx="4" ry="6"/>
              <!-- RIGHT FOREARM + HAND -->
              <path d="M85,84 Q89,100 90,115 Q91,122 88,124 Q86,126 84,122 L80,106 Q79,96 79,88Z"/>
              <ellipse cx="89" cy="126" rx="4" ry="6"/>
              <!-- HIPS/PELVIS -->
              <path d="M24,80 Q22,92 28,96 L72,96 Q78,92 76,80Z"/>
              <!-- LEFT LEG -->
              <path d="M28,96 Q24,120 26,145 Q27,152 32,152 Q37,152 38,145 L40,120 Q40,108 38,96Z"/>
              <!-- RIGHT LEG -->
              <path d="M72,96 Q76,120 74,145 Q73,152 68,152 Q63,152 62,145 L60,120 Q60,108 62,96Z"/>
              <!-- LEFT SHIN + FOOT -->
              <path d="M26,145 Q24,168 26,188 Q27,196 32,196 Q37,196 38,188 L38,165 Q38,155 38,148Z"/>
              <ellipse cx="30" cy="196" rx="8" ry="5"/>
              <!-- RIGHT SHIN + FOOT -->
              <path d="M74,145 Q76,168 74,188 Q73,196 68,196 Q63,196 62,188 L62,165 Q62,155 62,148Z"/>
              <ellipse cx="70" cy="196" rx="8" ry="5"/>
            </clipPath>

            <!-- Shimmer animation mask -->
            <linearGradient id="shimmer" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="white" stop-opacity="0"/>
              <stop offset="50%" stop-color="white" stop-opacity="0.15"/>
              <stop offset="100%" stop-color="white" stop-opacity="0"/>
              <animateTransform attributeName="gradientTransform" type="translate"
                from="-1 0" to="2 0" dur="2s" repeatCount="indefinite"/>
            </linearGradient>
          </defs>

          <!-- ① Outline/ghost of the full body (dim) -->
          <g clip-path="url(#bodyClip)">
            <rect x="0" y="0" width="100" height="220" fill="rgba(255,255,255,0.06)"/>
          </g>

          <!-- ② Filled portion from bottom -->
          <g clip-path="url(#bodyClip)">
            <rect x="0" y="{fill_y:.2f}" width="100" height="{fill_h:.2f}" fill="url(#fillGrad)">
              <animate attributeName="y" from="{min(fill_y + 10, total_height):.2f}" to="{fill_y:.2f}"
                dur="1.2s" fill="freeze" calcMode="spline"
                keySplines="0.25 0.1 0.25 1"/>
              <animate attributeName="height" from="{max(fill_h - 10, 0):.2f}" to="{fill_h:.2f}"
                dur="1.2s" fill="freeze" calcMode="spline"
                keySplines="0.25 0.1 0.25 1"/>
            </rect>
            <!-- Shimmer overlay -->
            <rect x="0" y="{fill_y:.2f}" width="100" height="{fill_h:.2f}" fill="url(#shimmer)"/>
          </g>

          <!-- ③ Body outline stroke -->
          <g clip-path="url(#bodyClip)">
            <rect x="0" y="0" width="100" height="220" fill="none"/>
          </g>
          <!-- Stroke the silhouette edges -->
          <g fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="0.8">
            <ellipse cx="50" cy="18" rx="12" ry="14"/>
            <rect x="44" y="30" width="12" height="8" rx="3"/>
            <path d="M28,38 Q22,55 24,80 L76,80 Q78,55 72,38 Q62,34 50,34 Q38,34 28,38Z"/>
            <path d="M24,40 Q14,50 12,75 Q11,82 15,84 Q19,86 21,80 L26,58 Q28,48 28,42Z"/>
            <path d="M76,40 Q86,50 88,75 Q89,82 85,84 Q81,86 79,80 L74,58 Q72,48 72,42Z"/>
            <path d="M15,84 Q11,100 10,115 Q9,122 12,124 Q14,126 16,122 L20,106 Q21,96 21,88Z"/>
            <ellipse cx="11" cy="126" rx="4" ry="6"/>
            <path d="M85,84 Q89,100 90,115 Q91,122 88,124 Q86,126 84,122 L80,106 Q79,96 79,88Z"/>
            <ellipse cx="89" cy="126" rx="4" ry="6"/>
            <path d="M24,80 Q22,92 28,96 L72,96 Q78,92 76,80Z"/>
            <path d="M28,96 Q24,120 26,145 Q27,152 32,152 Q37,152 38,145 L40,120 Q40,108 38,96Z"/>
            <path d="M72,96 Q76,120 74,145 Q73,152 68,152 Q63,152 62,145 L60,120 Q60,108 62,96Z"/>
            <path d="M26,145 Q24,168 26,188 Q27,196 32,196 Q37,196 38,188 L38,165 Q38,155 38,148Z"/>
            <ellipse cx="30" cy="196" rx="8" ry="5"/>
            <path d="M74,145 Q76,168 74,188 Q73,196 68,196 Q63,196 62,188 L62,165 Q62,155 62,148Z"/>
            <ellipse cx="70" cy="196" rx="8" ry="5"/>
          </g>

          <!-- ④ Heartbeat pulse on chest when high risk -->
          {"" if not is_high_risk else f'''
          <circle cx="50" cy="58" r="6" fill="{fill_color_top}" opacity="0.7">
            <animate attributeName="r" values="4;9;4" dur="{pulse_dur}" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.7;0;0.7" dur="{pulse_dur}" repeatCount="indefinite"/>
          </circle>
          <circle cx="50" cy="58" r="3" fill="white" opacity="0.9">
            <animate attributeName="opacity" values="1;0.3;1" dur="{pulse_dur}" repeatCount="indefinite"/>
          </circle>
          '''}
        </svg>
      </div>

      <!-- Risk meter bar below -->
      <div class="risk-meter-bar">
        <div class="risk-meter-track">
          <div class="risk-meter-fill" style="width:{risk_pct}%; background: linear-gradient(90deg, {fill_color_bot}, {fill_color_top});"></div>
        </div>
        <div class="risk-meter-labels">
          <span>0%</span><span>50%</span><span>100%</span>
        </div>
      </div>

    </div>
    """
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