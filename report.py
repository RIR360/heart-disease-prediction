"""
report.py  –  Heart Disease Risk PDF Report Generator
"""

import io
import math
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# ── Palette (light / print-friendly) ─────────────────────────────────────────
BG          = (1.0,   1.0,   1.0)     # white page
CARD        = (0.94,  0.96,  0.98)    # very light grey card
BORDER      = (0.78,  0.82,  0.88)    # soft grey border
WHITE       = (0.10,  0.12,  0.16)    # primary text (dark on light)
MUTED       = (0.42,  0.48,  0.56)    # secondary text
RED         = (0.75,  0.08,  0.08)    # deep red — prints well
AMBER       = (0.72,  0.40,  0.00)    # dark amber
GREEN       = (0.06,  0.50,  0.20)    # dark green
ACCENT_RED  = (0.55,  0.04,  0.04)    # darker red accent

W, H = A4   # 595 x 842 pts

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rgb(c):
    return c[0], c[1], c[2]

def set_fill(c, col):
    c.setFillColorRGB(*col)

def set_stroke(c, col):
    c.setStrokeColorRGB(*col)

def risk_color(pct):
    if pct < 30:   return GREEN
    if pct < 60:   return AMBER
    return RED

def risk_label(pct):
    if pct < 30:   return "LOW RISK"
    if pct < 60:   return "MODERATE RISK"
    return "HIGH RISK"

def metric_color(pct):
    if pct < 33:   return GREEN
    if pct < 66:   return AMBER
    return RED

def metric_status(pct):
    if pct < 33:   return "Normal"
    if pct < 66:   return "Elevated"
    return "High"

# ── Drawing primitives ────────────────────────────────────────────────────────

def rounded_rect(c, x, y, w, h, r=6, fill=True, stroke=False):
    p = c.beginPath()
    p.moveTo(x + r, y)
    p.lineTo(x + w - r, y)
    p.arcTo(x + w - r, y, x + w, y + r, 270, 90)
    p.lineTo(x + w, y + h - r)
    p.arcTo(x + w - r, y + h - r, x + w, y + h, 0, 90)
    p.lineTo(x + r, y + h)
    p.arcTo(x, y + h - r, x + r, y + h, 90, 90)
    p.lineTo(x, y + r)
    p.arcTo(x, y, x + r, y + r, 180, 90)
    p.close()
    c.drawPath(p, fill=1 if fill else 0, stroke=1 if stroke else 0)

def draw_donut(c, cx, cy, radius, thickness, pct, color):
    """Draw a donut arc from top (12 o'clock), filled pct% of full circle."""
    set_stroke(c, (0.14, 0.17, 0.25))
    c.setLineWidth(thickness)
    c.circle(cx, cy, radius, fill=0, stroke=1)

    if pct <= 0:
        return
    arc_col = color
    set_stroke(c, arc_col)
    c.setLineWidth(thickness)
    c.setLineCap(1)  # round caps

    # Draw arc using bezier approximation for the filled portion
    # Start at top (90°), go clockwise = decreasing angle
    start_angle = 90
    end_angle   = 90 - (pct / 100 * 360)
    # reportlab arc goes counter-clockwise; we flip
    # Use multiple small bezier segments
    steps  = max(int(pct / 100 * 60), 2)
    deg_step = -(pct / 100 * 360) / steps
    angles = [math.radians(start_angle + i * (-pct/100*360/steps))
              for i in range(steps + 1)]

    path = c.beginPath()
    x0 = cx + radius * math.cos(angles[0])
    y0 = cy + radius * math.sin(angles[0])
    path.moveTo(x0, y0)
    for i in range(1, len(angles)):
        xi = cx + radius * math.cos(angles[i])
        yi = cy + radius * math.sin(angles[i])
        # simple lineTo for each small step — looks smooth at this resolution
        path.lineTo(xi, yi)
    c.drawPath(path, fill=0, stroke=1)

def bar(c, x, y, w, h, pct, color):
    """Horizontal progress bar."""
    # Track
    set_fill(c, BORDER)
    rounded_rect(c, x, y, w, h, r=h/2)
    # Fill
    if pct > 0:
        fw = max(w * pct / 100, h)  # min width = height so caps look right
        fw = min(fw, w)
        set_fill(c, color)
        rounded_rect(c, x, y, fw, h, r=h/2)

# ── Page background ───────────────────────────────────────────────────────────

def draw_background(c):
    set_fill(c, BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)

# ── Header ────────────────────────────────────────────────────────────────────

def draw_header(c, generated_at):
    # Red accent bar at top
    set_fill(c, RED)
    c.rect(0, H - 5, W, 5, fill=1, stroke=0)

    # Light header background
    set_fill(c, CARD)
    c.rect(0, H - 70, W, 65, fill=1, stroke=0)

    # Bottom border line
    set_stroke(c, BORDER)
    c.setLineWidth(0.5)
    c.line(0, H - 70, W, H - 70)

    # Title
    set_fill(c, WHITE)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(28, H - 36, "Heart Disease Risk Report")

    # Subtitle
    set_fill(c, MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(28, H - 52, "Random Forest Model  \u00b7  Clinical Feature Analysis")
    c.drawString(28, H - 64, f"Generated: {generated_at}")

    # Heart icon
    set_fill(c, RED)
    c.setFont("Helvetica-Bold", 28)
    c.drawRightString(W - 28, H - 50, "\u2665")

# ── Risk summary card ─────────────────────────────────────────────────────────

def draw_risk_summary(c, risk_pct, patient):
    col = risk_color(risk_pct)
    lbl = risk_label(risk_pct)
    y_top = H - 85

    # Card background
    set_fill(c, CARD)
    set_stroke(c, BORDER)
    c.setLineWidth(0.5)
    rounded_rect(c, 20, y_top - 130, W - 40, 130, r=10, fill=True, stroke=True)

    # Left: donut
    cx, cy = 105, y_top - 65
    draw_donut(c, cx, cy, 42, 9, risk_pct, col)

    # Percentage text inside donut
    set_fill(c, col)
    c.setFont("Helvetica-Bold", 18)
    label_w = c.stringWidth(f"{risk_pct:.1f}%", "Helvetica-Bold", 18)
    c.drawString(cx - label_w/2, cy - 6, f"{risk_pct:.1f}%")

    # Tier label inside donut
    set_fill(c, MUTED)
    c.setFont("Helvetica", 6)
    tier_w = c.stringWidth(lbl, "Helvetica", 6)
    c.drawString(cx - tier_w/2, cy - 16, lbl)

    # Right: patient info + tier badge
    rx = 175
    # Tier badge
    set_fill(c, col)
    rounded_rect(c, rx, y_top - 30, 120, 20, r=4)
    set_fill(c, BG)
    c.setFont("Helvetica-Bold", 9)
    badge_txt = f"  {lbl}"
    c.drawString(rx + 8, y_top - 23, badge_txt)

    # Patient data
    fields = [
        ("Age",        f"{patient['age']} years"),
        ("Sex",        "Male" if patient['sex'] == 1 else "Female"),
        ("Prediction", "Positive (Disease Likely)" if patient['prediction'] == 1
                       else "Negative (No Disease)"),
    ]
    for i, (k, v) in enumerate(fields):
        yy = y_top - 58 - i * 20
        set_fill(c, MUTED)
        c.setFont("Helvetica", 8)
        c.drawString(rx, yy, k)
        set_fill(c, WHITE)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(rx + 90, yy, v)

# ── Section heading ───────────────────────────────────────────────────────────

def section_heading(c, text, y):
    set_fill(c, RED)
    c.rect(20, y - 1, 3, 13, fill=1, stroke=0)
    set_fill(c, WHITE)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(30, y, text)
    set_stroke(c, BORDER)
    c.setLineWidth(0.4)
    c.line(30, y - 4, W - 20, y - 4)
    return y - 20

# ── Clinical metrics table ────────────────────────────────────────────────────

def draw_metrics_table(c, patient, y_start):
    y = section_heading(c, "Clinical Input Summary", y_start)

    headers = ["Parameter", "Value", "Reference Range", "Status"]
    col_x   = [22, 180, 310, 450]
    col_w   = [155, 128, 138, 100]
    row_h   = 17

    # Reference ranges for each feature
    metrics = [
        ("Age",                   f"{patient['age']} yrs",         "18 – 80 yrs",     None),
        ("Sex",                   "Male" if patient['sex']==1 else "Female", "—",      None),
        ("Chest Pain Type",       {0:"Typical Angina",1:"Atypical Angina",
                                   2:"Non-Anginal",3:"Asymptomatic"}[patient['cp']],
                                  "Typical/Atypical preferred", None),
        ("Resting BP",            f"{patient['trestbps']} mm Hg",  "90 – 120 mm Hg",
         max(0,min(100,(patient['trestbps']-90)/90*100))),
        ("Cholesterol",           f"{patient['chol']} mg/dl",      "< 200 mg/dl",
         max(0,min(100,(patient['chol']-150)/250*100))),
        ("Fasting Blood Sugar",   "Yes" if patient['fbs']==1 else "No",  "< 120 mg/dl", None),
        ("Resting ECG",           {0:"Normal",1:"ST-T Wave",2:"LVH"}[patient['restecg']],
                                  "Normal preferred", None),
        ("Max Heart Rate",        f"{patient['thalach']} bpm",     "100 – 170 bpm",
         max(0,min(100,(200-patient['thalach'])/140*100))),
        ("Exercise Angina",       "Yes" if patient['exang']==1 else "No",  "No", None),
        ("ST Depression",         f"{patient['oldpeak']:.1f}",     "0.0 – 1.0",
         max(0,min(100,patient['oldpeak']/6*100))),
        ("ST Slope",              {0:"Upsloping",1:"Flat",2:"Downsloping"}[patient['slope']],
                                  "Upsloping preferred", None),
        ("Major Vessels (CA)",    str(patient['ca']),              "0",
         patient['ca']/3*100),
        ("Thalassemia",           {3.0:"Normal",6.0:"Fixed Defect",
                                   7.0:"Reversible Defect"}[patient['thal']],
                                  "Normal preferred", None),
    ]

    # Header row
    set_fill(c, BORDER)
    rounded_rect(c, 20, y - row_h + 4, W - 40, row_h, r=4)
    set_fill(c, MUTED)
    c.setFont("Helvetica-Bold", 7.5)
    for i, h in enumerate(headers):
        c.drawString(col_x[i] + 4, y - row_h + 8, h.upper())

    y -= row_h

    for idx, (name, value, ref, pct) in enumerate(metrics):
        # Alternating row
        if idx % 2 == 0:
            set_fill(c, (0.91,  0.93,  0.96))
            rounded_rect(c, 20, y - row_h + 4, W - 40, row_h, r=3)

        set_fill(c, MUTED)
        c.setFont("Helvetica", 8)
        c.drawString(col_x[0] + 4, y - row_h + 7, name)

        set_fill(c, WHITE)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(col_x[1] + 4, y - row_h + 7, str(value))

        set_fill(c, MUTED)
        c.setFont("Helvetica", 7.5)
        c.drawString(col_x[2] + 4, y - row_h + 7, ref)

        # Status badge
        if pct is not None:
            sc = metric_color(pct)
            st = metric_status(pct)
            set_fill(c, sc)
            badge_w = c.stringWidth(st, "Helvetica-Bold", 7) + 10
            rounded_rect(c, col_x[3] + 4, y - row_h + 5, badge_w, 11, r=3)
            set_fill(c, BG)
            c.setFont("Helvetica-Bold", 7)
            c.drawString(col_x[3] + 9, y - row_h + 8, st)

        y -= row_h

    return y - 8

# ── Organ impact bars ─────────────────────────────────────────────────────────

def draw_organ_bars(c, patient, risk_pct, y_start):
    y = section_heading(c, "Clinical Risk Indicators", y_start)

    chol_pct   = max(0, min(100, (patient['chol']     - 150) / 250 * 100))
    bp_pct     = max(0, min(100, (patient['trestbps'] -  90) /  90 * 100))
    hr_pct     = max(0, min(100, (200 - patient['thalach'])  / 140 * 100))
    st_pct     = max(0, min(100,  patient['oldpeak']         / 6.0 * 100))
    vessel_pct = patient['ca'] / 3 * 100

    indicators = [
        ("Heart Stress",   risk_pct,   "Overall model risk probability"),
        ("Cholesterol",    chol_pct,   f"{patient['chol']} mg/dl"),
        ("Blood Pressure", bp_pct,     f"{patient['trestbps']} mm Hg resting"),
        ("ST Depression",  st_pct,     f"{patient['oldpeak']:.1f}  exercise ECG"),
        ("Max Heart Rate", hr_pct,     f"{patient['thalach']} bpm"),
        ("Vessel Load",    vessel_pct, f"{patient['ca']} major vessel(s) affected"),
    ]

    bar_h    = 5
    row_h    = 26
    bar_x    = 160
    bar_w    = W - bar_x - 100
    label_w  = 135

    for icon_txt, pct, note in indicators:
        col = metric_color(pct)
        st  = metric_status(pct)

        # Row card
        set_fill(c, CARD)
        set_stroke(c, BORDER)
        c.setLineWidth(0.4)
        rounded_rect(c, 20, y - row_h + 4, W - 40, row_h - 2, r=5,
                     fill=True, stroke=True)

        # Label
        set_fill(c, WHITE)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(30, y - 13, icon_txt)

        # Note
        set_fill(c, MUTED)
        c.setFont("Helvetica", 7)
        c.drawString(30, y - 22, note)

        # Bar track + fill
        bar(c, bar_x, y - 16, bar_w, bar_h, pct, col)

        # Pct label
        set_fill(c, col)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(bar_x + bar_w + 6, y - 14, f"{pct:.0f}%")

        # Status badge
        set_fill(c, col)
        badge_w = c.stringWidth(st, "Helvetica-Bold", 6.5) + 10
        rounded_rect(c, W - 68, y - 20, badge_w, 11, r=3)
        set_fill(c, BG)
        c.setFont("Helvetica-Bold", 6.5)
        c.drawString(W - 63, y - 16, st)

        y -= row_h

    return y - 8

# ── Interpretation ────────────────────────────────────────────────────────────

def draw_interpretation(c, patient, risk_pct, y_start):
    y = section_heading(c, "Clinical Interpretation", y_start)

    col  = risk_color(risk_pct)
    lbl  = risk_label(risk_pct)

    lines = []

    if risk_pct >= 60:
        lines.append(
            f"The model assigns a HIGH risk probability of {risk_pct:.1f}%. "
            "This indicates multiple clinical features consistent with coronary artery disease. "
            "Immediate consultation with a cardiologist is strongly recommended."
        )
    elif risk_pct >= 30:
        lines.append(
            f"The model assigns a MODERATE risk probability of {risk_pct:.1f}%. "
            "Several clinical markers are elevated. Lifestyle modifications and follow-up "
            "with a physician are advised."
        )
    else:
        lines.append(
            f"The model assigns a LOW risk probability of {risk_pct:.1f}%. "
            "Clinical markers are largely within normal ranges. Routine monitoring "
            "and preventive care are recommended."
        )

    if patient['chol'] > 240:
        lines.append(
            f"Cholesterol ({patient['chol']} mg/dl) is above the high-risk threshold of "
            "240 mg/dl. Dietary intervention and lipid-lowering therapy may be warranted."
        )
    elif patient['chol'] > 200:
        lines.append(
            f"Cholesterol ({patient['chol']} mg/dl) is borderline high (200-239 mg/dl). "
            "Dietary improvements are recommended."
        )

    if patient['trestbps'] > 140:
        lines.append(
            f"Resting blood pressure ({patient['trestbps']} mm Hg) indicates Stage 2 "
            "hypertension. Blood pressure management is strongly advised."
        )
    elif patient['trestbps'] > 120:
        lines.append(
            f"Resting blood pressure ({patient['trestbps']} mm Hg) is elevated. "
            "Lifestyle changes to manage blood pressure are recommended."
        )

    if patient['oldpeak'] > 2.0:
        lines.append(
            f"ST depression of {patient['oldpeak']:.1f} during exercise is significant "
            "and may indicate myocardial ischemia. Stress testing follow-up is advised."
        )

    if patient['ca'] > 0:
        lines.append(
            f"{patient['ca']} major vessel(s) are affected based on fluoroscopy findings, "
            "which is a strong indicator of coronary artery disease."
        )

    if patient['thalach'] < 120:
        lines.append(
            f"Maximum heart rate ({patient['thalach']} bpm) is notably low, which may "
            "indicate chronotropic incompetence or beta-blocker use."
        )

    # Draw each paragraph
    max_w = W - 56
    for para in lines:
        # Wrap text manually
        words     = para.split()
        cur_line  = ""
        wrapped   = []
        c.setFont("Helvetica", 8)
        for word in words:
            test = cur_line + (" " if cur_line else "") + word
            if c.stringWidth(test, "Helvetica", 8) > max_w:
                wrapped.append(cur_line)
                cur_line = word
            else:
                cur_line = test
        if cur_line:
            wrapped.append(cur_line)

        # Bullet
        set_fill(c, col)
        c.circle(28, y - 5, 2.5, fill=1, stroke=0)

        for li, line in enumerate(wrapped):
            set_fill(c, WHITE if li == 0 else MUTED)
            c.setFont("Helvetica" if li > 0 else "Helvetica", 8)
            c.drawString(36, y - (li * 11), line)

        y -= len(wrapped) * 11 + 8

    return y - 6

# ── Footer ────────────────────────────────────────────────────────────────────

def draw_footer(c, page_num):
    # Light footer with top border
    set_fill(c, CARD)
    c.rect(0, 0, W, 26, fill=1, stroke=0)
    set_stroke(c, BORDER)
    c.setLineWidth(0.5)
    c.line(0, 26, W, 26)

    set_fill(c, MUTED)
    c.setFont("Helvetica", 7)
    c.drawString(20, 9,
        "DISCLAIMER: This report is for educational and research purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment.")
    c.drawRightString(W - 20, 9, f"Page {page_num}")

# ── Main entry point ──────────────────────────────────────────────────────────

def generate_report(patient: dict, risk_pct: float) -> bytes:
    """
    Generate a full PDF report.

    patient keys: age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                  exang, oldpeak, slope, ca, thal, prediction
    risk_pct: 0-100 float
    Returns: bytes (PDF)
    """
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=A4)
    c.setTitle("Heart Disease Risk Report")
    c.setAuthor("Heart Disease Predictor")

    generated_at = datetime.now().strftime("%d %B %Y, %H:%M")

    # ── Page 1 ────────────────────────────────────────────────────
    draw_background(c)
    draw_header(c, generated_at)
    draw_footer(c, 1)

    y = H - 92
    draw_risk_summary(c, risk_pct, patient)

    y = H - 240
    y = draw_organ_bars(c, patient, risk_pct, y)

    y = draw_interpretation(c, patient, risk_pct, y)

    # ── Page 2 ────────────────────────────────────────────────────
    c.showPage()
    draw_background(c)
    draw_header(c, generated_at)
    draw_footer(c, 2)

    y = H - 92
    y = draw_metrics_table(c, patient, y)

    c.save()
    buf.seek(0)
    return buf.read()