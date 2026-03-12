"""
report.py  -  Heart Disease Risk PDF Report Generator
Light/white theme — print-friendly.
Consistent 40pt page margins, improved Clinical Interpretation layout.
"""

import io
import math
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# ── Palette ───────────────────────────────────────────────────────────────────
BG         = (1.00, 1.00, 1.00)
CARD       = (0.94, 0.96, 0.98)
CARD_ALT   = (0.97, 0.98, 0.99)
BORDER     = (0.78, 0.82, 0.88)
TEXT       = (0.10, 0.12, 0.16)
MUTED      = (0.42, 0.48, 0.56)
RED        = (0.75, 0.08, 0.08)
AMBER      = (0.72, 0.40, 0.00)
GREEN      = (0.06, 0.50, 0.20)

W, H     = A4
PAD      = 40
INNER_W  = W - PAD * 2

def set_fill(c, col):   c.setFillColorRGB(*col)
def set_stroke(c, col): c.setStrokeColorRGB(*col)

def risk_color(p):  return GREEN if p < 30 else (AMBER if p < 60 else RED)
def risk_label(p):  return "LOW RISK" if p < 30 else ("MODERATE RISK" if p < 60 else "HIGH RISK")
def metric_color(p): return GREEN if p < 33 else (AMBER if p < 66 else RED)
def metric_status(p): return "Normal" if p < 33 else ("Elevated" if p < 66 else "High")

def rounded_rect(c, x, y, w, h, r=5, fill=True, stroke=False):
    r = min(r, w/2, h/2)
    p = c.beginPath()
    p.moveTo(x+r, y); p.lineTo(x+w-r, y)
    p.arcTo(x+w-r, y, x+w, y+r, 270, 90)
    p.lineTo(x+w, y+h-r)
    p.arcTo(x+w-r, y+h-r, x+w, y+h, 0, 90)
    p.lineTo(x+r, y+h)
    p.arcTo(x, y+h-r, x+r, y+h, 90, 90)
    p.lineTo(x, y+r)
    p.arcTo(x, y, x+r, y+r, 180, 90)
    p.close()
    c.drawPath(p, fill=1 if fill else 0, stroke=1 if stroke else 0)

def draw_donut(c, cx, cy, radius, thickness, pct, color):
    set_stroke(c, BORDER); c.setLineWidth(thickness)
    c.circle(cx, cy, radius, fill=0, stroke=1)
    if pct <= 0: return
    set_stroke(c, color); c.setLineWidth(thickness); c.setLineCap(1)
    steps  = max(int(pct/100*72), 3)
    angles = [math.radians(90 - (pct/100*360)*i/steps) for i in range(steps+1)]
    path   = c.beginPath()
    path.moveTo(cx + radius*math.cos(angles[0]), cy + radius*math.sin(angles[0]))
    for a in angles[1:]:
        path.lineTo(cx + radius*math.cos(a), cy + radius*math.sin(a))
    c.drawPath(path, fill=0, stroke=1)

def progress_bar(c, x, y, w, h, pct, color):
    set_fill(c, BORDER); rounded_rect(c, x, y, w, h, r=h/2)
    if pct > 0:
        fw = max(min(w*pct/100, w), h)
        set_fill(c, color); rounded_rect(c, x, y, fw, h, r=h/2)

def wrap_text(c, text, font, size, max_w):
    c.setFont(font, size)
    words, lines, cur = text.split(), [], ""
    for word in words:
        test = (cur + " " + word).strip()
        if c.stringWidth(test, font, size) <= max_w:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = word
    if cur: lines.append(cur)
    return lines

def draw_background(c):
    set_fill(c, BG); c.rect(0, 0, W, H, fill=1, stroke=0)

def draw_header(c, generated_at):
    set_fill(c, RED); c.rect(0, H-5, W, 5, fill=1, stroke=0)
    set_fill(c, CARD); c.rect(0, H-72, W, 67, fill=1, stroke=0)
    set_stroke(c, BORDER); c.setLineWidth(0.5); c.line(0, H-72, W, H-72)
    set_fill(c, TEXT); c.setFont("Helvetica-Bold", 20)
    c.drawString(PAD, H-37, "Heart Disease Risk Report")
    set_fill(c, MUTED); c.setFont("Helvetica", 9)
    c.drawString(PAD, H-52, "Random Forest Model  \u00b7  Clinical Feature Analysis")
    c.drawString(PAD, H-64, f"Generated: {generated_at}")
    set_fill(c, RED); c.setFont("Helvetica-Bold", 26)
    c.drawRightString(W-PAD, H-48, "\u2665")

def draw_footer(c, page_num, total=2):
    set_fill(c, CARD); c.rect(0, 0, W, 28, fill=1, stroke=0)
    set_stroke(c, BORDER); c.setLineWidth(0.5); c.line(0, 28, W, 28)
    set_fill(c, MUTED); c.setFont("Helvetica", 7)
    c.drawString(PAD, 10,
        "DISCLAIMER: This report is for educational and research purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment.")
    c.drawRightString(W-PAD, 10, f"Page {page_num} of {total}")

def section_heading(c, text, y):
    set_fill(c, RED); c.rect(PAD, y-1, 3, 13, fill=1, stroke=0)
    set_fill(c, TEXT); c.setFont("Helvetica-Bold", 10)
    c.drawString(PAD+9, y, text)
    set_stroke(c, BORDER); c.setLineWidth(0.4)
    c.line(PAD+9, y-5, W-PAD, y-5)
    return y - 22

def draw_risk_summary(c, risk_pct, patient):
    col   = risk_color(risk_pct)
    lbl   = risk_label(risk_pct)
    y_top = H - 86

    set_fill(c, CARD); set_stroke(c, BORDER); c.setLineWidth(0.5)
    rounded_rect(c, PAD, y_top-132, INNER_W, 132, r=8, fill=True, stroke=True)

    cx, cy = PAD+78, y_top-66
    draw_donut(c, cx, cy, 44, 9, risk_pct, col)
    set_fill(c, col); c.setFont("Helvetica-Bold", 17)
    lw = c.stringWidth(f"{risk_pct:.1f}%", "Helvetica-Bold", 17)
    c.drawString(cx-lw/2, cy-4, f"{risk_pct:.1f}%")
    set_fill(c, MUTED); c.setFont("Helvetica", 6)
    tw = c.stringWidth(lbl, "Helvetica", 6)
    c.drawString(cx-tw/2, cy-15, lbl)

    rx = PAD + 178
    set_fill(c, col); rounded_rect(c, rx, y_top-28, 140, 19, r=4)
    set_fill(c, BG); c.setFont("Helvetica-Bold", 9)
    c.drawString(rx+10, y_top-21, lbl)

    fields = [
        ("Age",        f"{patient['age']} years"),
        ("Sex",        "Male" if patient['sex']==1 else "Female"),
        ("Prediction", "Positive — Disease Likely" if patient['prediction']==1
                       else "Negative — No Disease Detected"),
    ]
    for i, (k, v) in enumerate(fields):
        yy = y_top - 56 - i*22
        set_fill(c, MUTED); c.setFont("Helvetica", 8); c.drawString(rx, yy, k)
        set_fill(c, TEXT);  c.setFont("Helvetica-Bold", 9); c.drawString(rx+85, yy, v)

def draw_organ_bars(c, patient, risk_pct, y_start):
    y = section_heading(c, "Clinical Risk Indicators", y_start)

    chol_pct   = max(0, min(100, (patient['chol']     - 150) / 250 * 100))
    bp_pct     = max(0, min(100, (patient['trestbps'] -  90) /  90 * 100))
    hr_pct     = max(0, min(100, (200 - patient['thalach'])  / 140 * 100))
    st_pct     = max(0, min(100,  patient['oldpeak']          / 6.0 * 100))
    vessel_pct = patient['ca'] / 3 * 100

    indicators = [
        ("Heart Stress",   risk_pct,   f"Overall model risk probability"),
        ("Cholesterol",    chol_pct,   f"{patient['chol']} mg/dl"),
        ("Blood Pressure", bp_pct,     f"{patient['trestbps']} mm Hg resting"),
        ("ST Depression",  st_pct,     f"{patient['oldpeak']:.1f}  (exercise ECG)"),
        ("Max Heart Rate", hr_pct,     f"{patient['thalach']} bpm"),
        ("Vessel Load",    vessel_pct, f"{patient['ca']} major vessel(s) affected"),
    ]

    row_h   = 27
    bar_x   = PAD + 148
    bar_w   = INNER_W - 148 - 72
    badge_x = W - PAD - 56

    for label, pct, note in indicators:
        col = metric_color(pct)
        st  = metric_status(pct)
        set_fill(c, CARD); set_stroke(c, BORDER); c.setLineWidth(0.4)
        rounded_rect(c, PAD, y-row_h+5, INNER_W, row_h-3, r=5, fill=True, stroke=True)
        set_fill(c, TEXT);  c.setFont("Helvetica-Bold", 8.5); c.drawString(PAD+10, y-12, label)
        set_fill(c, MUTED); c.setFont("Helvetica", 7);        c.drawString(PAD+10, y-21, note)
        progress_bar(c, bar_x, y-16, bar_w, 5, pct, col)
        set_fill(c, col); c.setFont("Helvetica-Bold", 7.5)
        c.drawString(bar_x+bar_w+6, y-13, f"{pct:.0f}%")
        bw = c.stringWidth(st, "Helvetica-Bold", 7) + 10
        set_fill(c, col); rounded_rect(c, badge_x, y-20, bw, 12, r=3)
        set_fill(c, BG);  c.setFont("Helvetica-Bold", 7); c.drawString(badge_x+5, y-15, st)
        y -= row_h

    return y - 10

def draw_interpretation(c, patient, risk_pct, y_start):
    y = section_heading(c, "Clinical Interpretation", y_start)

    findings = []

    if risk_pct >= 60:
        findings.append((RED, "High Risk Assessment",
            f"The model assigns a high risk probability of {risk_pct:.1f}%. Multiple clinical "
            "features are consistent with coronary artery disease. Immediate consultation "
            "with a cardiologist is strongly recommended."))
    elif risk_pct >= 30:
        findings.append((AMBER, "Moderate Risk Assessment",
            f"The model assigns a moderate risk probability of {risk_pct:.1f}%. Several "
            "clinical markers are elevated. Lifestyle modifications and physician follow-up "
            "are advised."))
    else:
        findings.append((GREEN, "Low Risk Assessment",
            f"The model assigns a low risk probability of {risk_pct:.1f}%. Clinical markers "
            "are largely within normal ranges. Routine monitoring and preventive care "
            "are recommended."))

    if patient['chol'] > 240:
        findings.append((RED, "Elevated Cholesterol",
            f"Cholesterol of {patient['chol']} mg/dl exceeds the high-risk threshold of "
            "240 mg/dl. Dietary intervention and lipid-lowering therapy may be warranted."))
    elif patient['chol'] > 200:
        findings.append((AMBER, "Borderline Cholesterol",
            f"Cholesterol of {patient['chol']} mg/dl is borderline high (200–239 mg/dl). "
            "Dietary improvements and regular monitoring are recommended."))

    if patient['trestbps'] > 140:
        findings.append((RED, "Stage 2 Hypertension",
            f"Resting blood pressure of {patient['trestbps']} mm Hg indicates Stage 2 "
            "hypertension. Active blood pressure management is strongly advised."))
    elif patient['trestbps'] > 120:
        findings.append((AMBER, "Elevated Blood Pressure",
            f"Resting blood pressure of {patient['trestbps']} mm Hg is above the normal "
            "range. Lifestyle changes to manage blood pressure are recommended."))

    if patient['oldpeak'] > 2.0:
        findings.append((RED, "Significant ST Depression",
            f"ST depression of {patient['oldpeak']:.1f} during exercise is clinically "
            "significant and may indicate myocardial ischemia. Stress test follow-up is advised."))
    elif patient['oldpeak'] > 1.0:
        findings.append((AMBER, "Mild ST Depression",
            f"ST depression of {patient['oldpeak']:.1f} is mildly elevated. Further evaluation "
            "during routine cardiology review is suggested."))

    if patient['ca'] >= 2:
        findings.append((RED, "Multiple Vessels Affected",
            f"{patient['ca']} major vessels are affected based on fluoroscopy findings. "
            "This is a strong indicator of significant coronary artery disease."))
    elif patient['ca'] == 1:
        findings.append((AMBER, "Single Vessel Involvement",
            "1 major vessel is affected based on fluoroscopy. This warrants further "
            "cardiology assessment and monitoring."))

    if patient['thalach'] < 120:
        findings.append((AMBER, "Low Maximum Heart Rate",
            f"Maximum heart rate of {patient['thalach']} bpm is notably low, which may "
            "indicate chronotropic incompetence or the effect of rate-limiting medication."))

    if patient['exang'] == 1:
        findings.append((RED, "Exercise-Induced Angina",
            "The presence of exercise-induced angina is a significant indicator of "
            "obstructive coronary artery disease. Cardiology referral is advised."))

    # Layout constants
    card_pad  = 10
    title_sz  = 8.5
    body_sz   = 8
    line_h    = 11
    dot_r     = 3.5
    text_x    = PAD + card_pad*2 + dot_r*2 + 4
    text_w    = INNER_W - card_pad*2 - dot_r*2 - 14

    for fcol, title, body in findings:
        body_lines = wrap_text(c, body, "Helvetica", body_sz, text_w)
        card_h     = card_pad + line_h + 4 + len(body_lines)*line_h + card_pad - 2

        if y - card_h < 50:
            break

        # Card
        set_fill(c, CARD); set_stroke(c, BORDER); c.setLineWidth(0.4)
        rounded_rect(c, PAD, y-card_h, INNER_W, card_h, r=5, fill=True, stroke=True)

        # Left colour accent strip
        set_fill(c, fcol)
        rounded_rect(c, PAD, y-card_h, 4, card_h, r=2)

        # Bullet dot
        dot_x = PAD + card_pad + dot_r + 2
        dot_y = y - card_pad - dot_r - 1
        set_fill(c, fcol); c.circle(dot_x, dot_y, dot_r, fill=1, stroke=0)

        # Title
        set_fill(c, TEXT); c.setFont("Helvetica-Bold", title_sz)
        c.drawString(text_x, y-card_pad-1, title)

        # Body
        body_y = y - card_pad - line_h - 4
        for line in body_lines:
            set_fill(c, MUTED); c.setFont("Helvetica", body_sz)
            c.drawString(text_x, body_y, line)
            body_y -= line_h

        y -= card_h + 6

    return y - 4

def draw_metrics_table(c, patient, y_start):
    y = section_heading(c, "Clinical Input Summary", y_start)

    col_x = [PAD, PAD+165, PAD+305, PAD+430]
    row_h = 17

    # Header
    set_fill(c, CARD); rounded_rect(c, PAD, y-row_h+4, INNER_W, row_h, r=4)
    for i, h in enumerate(["Parameter", "Value", "Reference Range", "Status"]):
        set_fill(c, MUTED); c.setFont("Helvetica-Bold", 7.5)
        c.drawString(col_x[i]+5, y-row_h+7, h.upper())
    y -= row_h

    metrics = [
        ("Age",               f"{patient['age']} yrs",         "18 – 80 yrs",             None),
        ("Sex",               "Male" if patient['sex']==1 else "Female", "—",              None),
        ("Chest Pain Type",   {0:"Typical Angina",1:"Atypical Angina",
                               2:"Non-Anginal",3:"Asymptomatic"}[patient['cp']],
                              "Typical / Atypical preferred",                               None),
        ("Resting BP",        f"{patient['trestbps']} mm Hg",  "90 – 120 mm Hg",
         max(0, min(100, (patient['trestbps']-90)/90*100))),
        ("Cholesterol",       f"{patient['chol']} mg/dl",      "< 200 mg/dl",
         max(0, min(100, (patient['chol']-150)/250*100))),
        ("Fasting Blood Sugar","Yes" if patient['fbs']==1 else "No", "< 120 mg/dl",        None),
        ("Resting ECG",       {0:"Normal",1:"ST-T Wave",2:"LVH"}[patient['restecg']],
                              "Normal preferred",                                            None),
        ("Max Heart Rate",    f"{patient['thalach']} bpm",     "100 – 170 bpm",
         max(0, min(100, (200-patient['thalach'])/140*100))),
        ("Exercise Angina",   "Yes" if patient['exang']==1 else "No", "No",                None),
        ("ST Depression",     f"{patient['oldpeak']:.1f}",     "0.0 – 1.0",
         max(0, min(100, patient['oldpeak']/6*100))),
        ("ST Slope",          {0:"Upsloping",1:"Flat",2:"Downsloping"}[patient['slope']],
                              "Upsloping preferred",                                         None),
        ("Major Vessels (CA)",str(patient['ca']),              "0",
         patient['ca']/3*100),
        ("Thalassemia",       {3.0:"Normal",6.0:"Fixed Defect",
                               7.0:"Reversible Defect"}[patient['thal']],
                              "Normal preferred",                                             None),
    ]

    for idx, (name, value, ref, pct) in enumerate(metrics):
        set_fill(c, CARD_ALT if idx%2==0 else BG)
        rounded_rect(c, PAD, y-row_h+4, INNER_W, row_h, r=3)
        set_fill(c, MUTED); c.setFont("Helvetica", 8);      c.drawString(col_x[0]+5, y-row_h+7, name)
        set_fill(c, TEXT);  c.setFont("Helvetica-Bold", 8); c.drawString(col_x[1]+5, y-row_h+7, str(value))
        set_fill(c, MUTED); c.setFont("Helvetica", 7.5);    c.drawString(col_x[2]+5, y-row_h+7, ref)
        if pct is not None:
            sc = metric_color(pct); st = metric_status(pct)
            bw = c.stringWidth(st, "Helvetica-Bold", 7) + 10
            set_fill(c, sc); rounded_rect(c, col_x[3]+5, y-row_h+5, bw, 11, r=3)
            set_fill(c, BG); c.setFont("Helvetica-Bold", 7); c.drawString(col_x[3]+10, y-row_h+8, st)
        y -= row_h

    return y - 10

def generate_report(patient: dict, risk_pct: float) -> bytes:
    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=A4)
    c.setTitle("Heart Disease Risk Report")
    c.setAuthor("Heart Disease Predictor")
    ts = datetime.now().strftime("%d %B %Y, %H:%M")

    draw_background(c); draw_header(c, ts); draw_footer(c, 1)
    draw_risk_summary(c, risk_pct, patient)
    y = H - 242
    y = draw_organ_bars(c, patient, risk_pct, y)
    draw_interpretation(c, patient, risk_pct, y)

    c.showPage()
    draw_background(c); draw_header(c, ts); draw_footer(c, 2)
    draw_metrics_table(c, patient, H-92)

    c.save(); buf.seek(0)
    return buf.read()