import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import math
import time
import os
import stripe
import sqlite3
import hashlib
from collections import deque

# --- 1. CONFIGURATION & STRIPE SETUP ---
# ADDED: initial_sidebar_state="expanded" for mobile users
st.set_page_config(page_title="SluggerAI: Biomechanics Lab", layout="wide", page_icon="⚾", initial_sidebar_state="expanded")

# ---------------------------------------------------------
# 🔒 SECURITY WARNING: PUT YOUR KEYS HERE AGAIN
# ---------------------------------------------------------
stripe.api_key = "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" 
PRO_PRICE_ID = "price_1SWR4iLfYK5wKxPUZUgxflYN"           
# ---------------------------------------------------------

# --- CSS STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .highlight-cyan { color: #00f2ea; font-weight: bold; }
    .highlight-gold { color: #ffd700; font-weight: bold; }

    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #111111, #1a1a1a);
        border: 1px solid #333;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 242, 234, 0.1);
        text-align: center;
    }
    div[data-testid="metric-container"] label { color: #888; font-size: 0.8rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        background: -webkit-linear-gradient(#eee, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Arial Black', sans-serif;
    }

    .paywall-box {
        background: linear-gradient(45deg, #1a1a1a, #2d2d2d);
        border: 2px solid #ffd700;
        padding: 30px;
        text-align: center;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
    }
    .report-card {
        background-color: #0f1116;
        border-left: 4px solid #ff0055;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 20px;
    }
    .drill-box {
        background-color: #0a1f1c;
        border: 1px solid #00f2ea;
        color: #ccfbf9;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
    h1, h2, h3 { font-family: 'Verdana', sans-serif; }
    .stProgress > div > div > div > div { background-color: #ff0055; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATABASE ENGINE (SQLite) ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, password TEXT, tier TEXT)''')
    conn.commit()
    conn.close()

def create_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Hash password for security
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (email, password, tier) VALUES (?, ?, ?)", 
                  (email, hashed_pw, 'Free'))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # User already exists
    finally:
        conn.close()

def check_login(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT tier FROM users WHERE email=? AND password=?", (email, hashed_pw))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def upgrade_user_tier(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("UPDATE users SET tier='Pro' WHERE email=?", (email,))
    conn.commit()
    conn.close()

# Initialize DB on load
init_db()

# --- 3. SESSION & STATE MANAGEMENT ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_tier' not in st.session_state:
    st.session_state.user_tier = 'Free'
if 'username' not in st.session_state:
    st.session_state.username = ''

# CHECK FOR PAYMENT RETURN (Stripe Success)
try:
    if st.query_params.get("paid") == "true":
        if st.session_state.username:
            upgrade_user_tier(st.session_state.username) # SAVE TO DB
            st.session_state.user_tier = "Pro"
            st.toast("Payment Confirmed! Account Upgraded.", icon="🎉")
            st.query_params.clear()
except:
    pass # Ignore errors on older streamlit versions

def create_checkout_session():
    try:
        # REPLACE THIS WITH YOUR ACTUAL REPLIT APP URL
        base_url = "https://cb1edd87-9afc-4d36-bc67-276adebc4661-00-315xqx894frsu.janeway.replit.dev" 

        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{'price': PRO_PRICE_ID, 'quantity': 1}],
            mode='payment',
            success_url=base_url + "/?paid=true",
            cancel_url=base_url + "/?paid=false",
        )
        return checkout_session.url
    except Exception as e:
        st.error(f"Stripe Error: {e}")
        return None

def auth_page():
    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.title("⚾ SLUGGER.AI")
        st.markdown("### Biomechanics Lab Login")

        tab1, tab2 = st.tabs(["Login", "Create Account"])

        with tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Sign In", use_container_width=True):
                tier = check_login(email, password)
                if tier:
                    st.session_state.logged_in = True
                    st.session_state.username = email
                    st.session_state.user_tier = tier
                    st.success("Welcome back!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

        with tab2:
            new_email = st.text_input("New Email", key="signup_email")
            new_pass = st.text_input("New Password", type="password", key="signup_pass")
            if st.button("Create Account", use_container_width=True):
                if create_user(new_email, new_pass):
                    st.success("Account created! Please Sign In.")
                else:
                    st.error("User already exists.")

# --- 4. PHYSICS ENGINE (UNCHANGED & ACCURATE) ---
def calculate_physics(wrist_speed_mps, height_m, shoulder_tilt_deg, result_override, capture_fps):
    wrist_mph = wrist_speed_mps * 2.237
    est_bat_speed = wrist_mph * 3.0

    if est_bat_speed > 95: est_bat_speed = 95 
    if est_bat_speed < 40: est_bat_speed = 45 + (wrist_mph * 2)

    smash_factor = 1.20 
    if "Home Run" in result_override: smash_factor = 1.25

    exit_velo = est_bat_speed * smash_factor
    if exit_velo > 120: exit_velo = 120

    launch_angle = 10 + (shoulder_tilt_deg * 0.8) 

    if result_override == "Home Run (Over Fence)":
        if launch_angle < 19: launch_angle = 23 
        if exit_velo < 90: exit_velo = 92.0 
        min_bat = exit_velo / 1.25
        if est_bat_speed < min_bat: est_bat_speed = min_bat

    elif result_override == "Line Drive":
        launch_angle = 12 + (launch_angle * 0.1)
        if exit_velo < 75: exit_velo = 80 
        if est_bat_speed < (exit_velo / 1.2): est_bat_speed = exit_velo / 1.2

    elif result_override == "Ground Ball":
        launch_angle = -5 + abs(launch_angle * 0.1)

    g = 9.81
    v_ms = exit_velo * 0.44704
    theta_rad = math.radians(launch_angle)

    dist_m = (v_ms**2 * math.sin(2 * theta_rad)) / g
    dist_ft = (dist_m * 3.281) * 0.80 

    return est_bat_speed, exit_velo, launch_angle, dist_ft

# --- 5. VIDEO PROCESSING (UNCHANGED) ---
def analyze_swing(video_path, height_m, result_type, focus_pct, video_mode):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    file_fps = cap.get(cv2.CAP_PROP_FPS)
    if file_fps < 20 or math.isnan(file_fps): file_fps = 30.0

    if video_mode == "Slow Motion (120fps)": math_fps = 120.0
    elif video_mode == "Super Slow Mo (240fps)": math_fps = 240.0
    else: math_fps = 30.0 

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    margin = (100 - focus_pct) / 2
    x_start = int((margin / 100) * width)
    x_end = int(((100 - margin) / 100) * width)
    crop_width = x_end - x_start

    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    out = cv2.VideoWriter(tfile_out.name, cv2.VideoWriter_fourcc(*'vp80'), file_fps, (crop_width, height))

    pos_buffer = deque(maxlen=5) 
    max_wrist_speed_mps = 0
    max_shoulder_tilt = 0
    contact_frame = np.zeros((height, crop_width, 3), dtype=np.uint8)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="AI Isolating Hitter...")
    pixels_per_meter = (height * 0.75) / height_m 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        roi = frame[:, x_start:x_end] 
        if frame_count == int(total_frames / 2): contact_frame = roi.copy()
        if frame_count % 10 == 0 and total_frames > 0: progress_bar.progress(min(frame_count/total_frames, 1.0))

        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        img_draw = roi.copy() 

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            tilt = (l_shoulder.y - r_shoulder.y) * 100 
            if tilt > max_shoulder_tilt: max_shoulder_tilt = tilt

            curr_x = r_wrist.x * crop_width
            curr_y = r_wrist.y * height
            pos_buffer.append((curr_x, curr_y))

            if len(pos_buffer) >= 2:
                prev_avg_x = sum([p[0] for p in list(pos_buffer)[:-1]]) / (len(pos_buffer)-1)
                prev_avg_y = sum([p[1] for p in list(pos_buffer)[:-1]]) / (len(pos_buffer)-1)
                curr_x, curr_y = pos_buffer[-1]
                dist_px = math.sqrt((curr_x - prev_avg_x)**2 + (curr_y - prev_avg_y)**2)

                if dist_px > 3 and dist_px < (crop_width * 0.10): 
                    dist_m = dist_px / pixels_per_meter
                    speed_mps = dist_m * math_fps 
                    if speed_mps > max_wrist_speed_mps:
                        max_wrist_speed_mps = speed_mps
                        contact_frame = roi.copy() 

            mp.solutions.drawing_utils.draw_landmarks(img_draw, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(img_draw)

    cap.release()
    out.release()
    progress_bar.empty()

    bat_speed, exit_velo, launch_angle, distance = calculate_physics(max_wrist_speed_mps, height_m, max_shoulder_tilt, result_type, math_fps)
    contact_frame_rgb = cv2.cvtColor(contact_frame, cv2.COLOR_BGR2RGB)
    return tfile_out.name, contact_frame_rgb, bat_speed, exit_velo, launch_angle, distance

# --- 6. NEW AI PARAGRAPH LOGIC ---
def get_ai_feedback(bat_speed, launch_angle):
    strengths = []
    fixes = []
    drills = []
    player_comp = ""

    # Construct the AI Paragraph
    summary = ""

    # 1. Bat Speed Analysis
    if bat_speed > 75:
        summary += f"🔥 **Elite Power:** Your bat speed of **{int(bat_speed)} MPH** is MLB caliber. You are generating massive rotational force from your core. "
        strengths.append("Elite Hand Speed")
        player_comp = "Bryce Harper (Power)"
    elif bat_speed > 65:
        summary += f"✅ **Solid Contact:** Your speed of **{int(bat_speed)} MPH** is competitive. You have a strong foundation, but increasing rotational speed by 5-10% could turn gap-hits into home runs. "
        strengths.append("Solid Bat Speed")
        player_comp = "Mookie Betts (Gap)"
    else:
        summary += f"⚠️ **Speed Warning:** At **{int(bat_speed)} MPH**, you are likely 'casting' or arm-swinging. We need to engage your hips earlier to create more whip through the zone. "
        fixes.append("Low Bat Speed")
        player_comp = "Contact Hitter Style"
        drills.append({"name": "Weighted Bat Swings", "description": "Overload/Underload training."})

    # 2. Launch Angle Analysis
    if launch_angle > 40:
        summary += f"However, your launch angle (**{int(launch_angle)}°**) is too steep, which often leads to non-productive pop-ups. You are likely dipping your back shoulder too early."
        fixes.append("Uppercut / Pop-up risk")
        drills.append({"name": "High Tee Drill", "description": "Hit balls off a high tee to flatten path."})
    elif launch_angle < 10:
        summary += f"You are also chopping down on the ball (**{int(launch_angle)}°**), causing ground balls. Focus on keeping the barrel in the zone longer to create natural lift."
        fixes.append("Groundball risk")
        drills.append({"name": "Low Tee Drill", "description": "Focus on digging the ball out."})
    else:
        summary += f"Your attack angle of **{int(launch_angle)}°** is optimal for driving the ball into the gaps. You are matching the plane of the pitch perfectly."
        strengths.append("Optimal Launch Angle")

    return {
        "scouting_report": {"summary": summary, "strengths": strengths, "fix_priorities": fixes, "player_comp": player_comp},
        "drills": drills
    }

# --- 7. MAIN APP LOGIC ---
if not st.session_state.logged_in:
    auth_page()
else:
    with st.sidebar:
        st.header("👤 Profile")
        st.write(f"User: **{st.session_state.username}**")

        if st.session_state.user_tier == "Pro":
            st.success("🌟 PRO MEMBER")
        else:
            st.info("🆓 FREE TIER")

        st.markdown("---")
        st.header("⚙️ Calibration")
        height_ft = st.number_input("Player Height (ft)", 4.0, 7.0, 6.0)
        result_type = st.selectbox("Actual Swing Result", ["Unknown", "Ground Ball", "Line Drive", "Fly Ball", "Home Run (Over Fence)"])
        video_mode = st.selectbox("Video Type", ["Normal Speed (Standard)", "Slow Motion (120fps)", "Super Slow Mo (240fps)"])
        st.divider()
        st.header("🎯 Target Lock")
        focus_pct = st.slider("Focus Zone Width % (If the AI is tracking the umpire or catcher, decrease or increase this to zoom in on the hitter.)", 20, 100, 60)

        st.markdown("---")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_tier = 'Free'
            st.rerun()

    st.title("⚾ SLUGGER.AI")
    st.caption(f"Biomechanics Lab | Plan: **{st.session_state.user_tier.upper()}**")

    # ADDED: Instructions
    with st.expander("ℹ️ **How to use SluggerAI**", expanded=True):
        st.write("""
        1. **Open the Sidebar** (arrow top-left) to calibrate your height and video type.
        2. **Upload a Video** of your swing (Side view works best).
        3. **Wait for AI Analysis** to track your skeleton and calculate physics.
        4. **Unlock Pro Mode** to see advanced metrics like Exit Velo and Distance.
        """)

    uploaded_file = st.file_uploader("📂 Upload Swing Video", type=['mp4', 'mov'])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        vid_path, snap, bat_s, exit_v, launch_a, dist = analyze_swing(tfile.name, height_ft*0.3048, result_type, focus_pct, video_mode)
        feedback = get_ai_feedback(bat_s, launch_a)
        report = feedback['scouting_report']

        is_pro = st.session_state.user_tier == "Pro"

        st.markdown("### 📊 Swing Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bat Speed", f"{int(bat_s)} MPH", delta="Est. Tip Speed")

        if is_pro:
            c2.metric("Exit Velo", f"{int(exit_v)} MPH", delta="Ball Speed")
            c3.metric("Launch Angle", f"{int(launch_a)}°", delta="Attack Angle")
            c4.metric("Est. Distance", f"{int(dist)} ft", delta="Trajectory")
        else:
            c2.metric("Exit Velo", "🔒 PRO", delta="Upgrade")
            c3.metric("Launch Angle", "🔒 PRO", delta="Upgrade")
            c4.metric("Est. Distance", "🔒 PRO", delta="Upgrade")

        st.divider()
        col_vid, col_rep = st.columns([1, 1])
        with col_vid:
            st.markdown(f"**🔍 AI Vision (Focused {focus_pct}%)**")
            st.video(vid_path)
            st.markdown("**📸 Point of Max Velocity**")
            st.image(snap, use_container_width=True)

        with col_rep:
            if is_pro:
                # UPDATED: Uses the new AI paragraph
                st.markdown(f"""<div class="report-card"><h3 style="margin-top:0">📝 Scouting Report</h3><p style="font-size:1.1rem; color:#ddd;">{report['summary']}</p><p><strong class="highlight-cyan">Pro Comparison:</strong> {report['player_comp']}</p></div>""", unsafe_allow_html=True)
                c_str, c_fix = st.columns(2)
                with c_str:
                    st.success("**✅ Strengths**")
                    for s in report['strengths']: st.write(f"• {s}")
                with c_fix:
                    st.error("**⚠️ Improvements**")
                    for f in report['fix_priorities']: st.write(f"• {f}")
                st.markdown("### 🏋️ Prescribed Drills")
                if feedback['drills']:
                    for d in feedback['drills']:
                        st.markdown(f"""<div class="drill-box"><strong>🔹 {d['name']}</strong><br><span style="font-size:0.9rem; opacity:0.8;">{d['description']}</span></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="report-card"><h3 style="margin-top:0">📝 Scouting Report</h3><p style="font-size:1.1rem; color:#ddd;">Your bat speed is <strong>{int(bat_s)} MPH</strong>.</p><p style="filter: blur(4px); opacity: 0.6;">Your swing mechanics resemble Mike Trout but you need to fix your upper cut angle to stop popping out. Focus on keeping the barrel level through the zone.</p></div>""", unsafe_allow_html=True)
                st.markdown("""<div class="paywall-box"><h2 class="highlight-gold">🔒 UNLOCK PRO ANALYSIS</h2><p>Get Exit Velocity, Launch Angle, Mechanical Flaws, and Custom Drills.</p></div>""", unsafe_allow_html=True)

                # CHECKOUT BUTTON
                if st.button("💳 UPGRADE NOW ($5.00)", key="stripe_btn"):
                    with st.spinner("Creating Secure Checkout..."):
                        checkout_url = create_checkout_session()
                        if checkout_url:
                            html_button = f'<a href="{checkout_url}" target="_self" style="background-color:#ffd700; color:black; padding:10px 20px; border-radius:5px; text-decoration:none; font-weight:bold;">👉 CLICK HERE TO PAY</a>'
                            st.markdown(html_button, unsafe_allow_html=True)