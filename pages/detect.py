# local no hardcoded

import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
import os
import time
import requests
import psycopg2
import pandas as pd

# ===========================
# CONFIG
# ===========================
HELMET_MODEL_PATH = "helmet_best.pt"
ACCIDENT_MODEL_PATH = "accident_best.pt"
OUTPUT_DIR = "web_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# SIDEBAR INPUTS
# ===========================
with st.sidebar:
    st.header("Database Connection")
    DB_HOST = st.text_input("DB host")
    DB_PORT = st.text_input("DB port", "5432")
    DB_NAME = st.text_input("DB name")
    DB_USER = st.text_input("DB user")
    DB_PASS = st.text_input("DB password", type="password")

    st.header("Telegram Bot")
    BOT_TOKEN = st.text_input("Bot Token", type="password")
    CHAT_ID = st.text_input("Chat ID",type="password")


# ===========================
# TELEGRAM FUNCTIONS
# ===========================
def send_telegram_message(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text})
        return True
    except:
        return False

def send_telegram_photo(image_path: str, caption: str = "") -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                      data={"chat_id": CHAT_ID, "caption": caption},
                      files={"photo": open(image_path,"rb")})
        return True
    except:
        return False

def send_telegram_video(video_path: str, caption: str = "") -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        with open(video_path, "rb") as f:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo",
                          data={"chat_id": CHAT_ID, "caption": caption},
                          files={"video": f})
        return True
    except Exception as e:
        print("Telegram video error:", e)
        return False

# ===========================
# DATABASE FUNCTIONS
# ===========================
def get_db_conn():
    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS]):
        st.warning("Database credentials not provided.")
        return None
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=DB_PORT)

def create_table_if_not_exists():
    conn = get_db_conn()
    if conn is None:
        return
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            file_name TEXT,
            mode TEXT,
            result TEXT,
            alert_sent BOOLEAN,
            image_path TEXT,
            extra_info TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_log(file_name, mode, result, alert_sent, image_path, extra_info=""):
    conn = get_db_conn()
    if conn is None:
        return
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO detection_logs (timestamp, file_name, mode, result, alert_sent, image_path, extra_info)
        VALUES (NOW(), %s, %s, %s, %s, %s, %s);
    """, (file_name, mode, result, alert_sent, image_path, extra_info))
    conn.commit()
    cur.close()
    conn.close()

create_table_if_not_exists()

# ===========================
# LOAD MODELS
# ===========================
@st.cache_resource
def load_yolo(path):
    return YOLO(path)

helmet_model = load_yolo(HELMET_MODEL_PATH)
accident_model = load_yolo(ACCIDENT_MODEL_PATH)

# ===========================
# STREAMLIT UI MAIN
# ===========================
st.set_page_config(page_title="🚨 SafeRide AI", layout="wide")
st.title("🚨 SafeRide AI — Helmet & Accident Detection")
st.write("Upload an image or video for detection. Results will appear below.")

st.header("Detection Settings")
detection_mode = st.selectbox("Detection Mode", ["Helmet Detection", "Accident Detection", "Both"])
uploaded = st.file_uploader("Upload Image or Video", type=["jpg","jpeg","png","mp4"])

# ===========================
# HELPER FUNCTIONS
# ===========================
def load_image_opencv(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect_classes(model, img):
    res = model(img)
    detected_class = None
    for box in res[0].boxes:
        cls_name = res[0].names[int(box.cls[0])].lower()
        detected_class = cls_name
        break
    return detected_class, res[0].plot()

# ===========================
# MAIN DETECTION LOGIC
# ===========================
if uploaded is not None:
    ext = uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    if ext in ["jpg","jpeg","png"]:
        img = load_image_opencv(tmp_path)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        col1, col2 = st.columns(2)
        helmet_result, accident_result = "No Helmet", "No Accident"
        alert_sent = False

        # Helmet Detection
        if detection_mode in ["Helmet Detection", "Both"]:
            col1.subheader("Helmet Detection")
            helmet_cls, annotated_h = detect_classes(helmet_model, img)
            col1.image(annotated_h, caption="Helmet Output", use_container_width=True)
            helmet_result = "Helmet" if helmet_cls == "helmet" else "No Helmet"
            helmet_image_path = os.path.join(OUTPUT_DIR, f"{helmet_cls}_{int(time.time())}.jpg")
            cv2.imwrite(helmet_image_path, cv2.cvtColor(annotated_h, cv2.COLOR_RGB2BGR))
            insert_log(uploaded.name, f"{helmet_result} Detection", helmet_result, False, helmet_image_path, "Image detection")

        # Accident Detection
        if detection_mode in ["Accident Detection", "Both"]:
            col2.subheader("Accident Detection")
            accident_cls, annotated_a = detect_classes(accident_model, img)
            col2.image(annotated_a, caption="Accident Output", use_container_width=True)
            accident_result = "Accident" if accident_cls == "accident" else "No Accident"
            accident_image_path = os.path.join(OUTPUT_DIR, f"{accident_cls}_{int(time.time())}.jpg")
            cv2.imwrite(accident_image_path, cv2.cvtColor(annotated_a, cv2.COLOR_RGB2BGR))

            # Telegram alert
            if accident_result == "Accident":
                msg = f"🚨 Accident Detected!\n🕒 Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                alert_sent = send_telegram_message(msg)
                send_telegram_photo(accident_image_path, "🚨 Accident detected!")
                st.error("🚨 Accident detected! Telegram alert sent.")

            insert_log(uploaded.name, f"{accident_result} Detection", accident_result, alert_sent, accident_image_path, "Image detection")

    elif ext == "mp4":
        st.video(tmp_path)
        st.info("Processing video...")
        cap = cv2.VideoCapture(tmp_path)
        helmet_found, no_helmet_found, accident_found = False, False, False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        annotated_video_path = os.path.join(OUTPUT_DIR, f"annotated_{int(time.time())}.mp4")
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_to_write = frame_rgb

            if detection_mode in ["Helmet Detection", "Both"]:
                helmet_cls, annotated_h = detect_classes(helmet_model, frame_rgb)
                frame_to_write = annotated_h
                if helmet_cls == "helmet":
                    helmet_found = True
                else:
                    no_helmet_found = True

            if detection_mode in ["Accident Detection", "Both"]:
                accident_cls, annotated_a = detect_classes(accident_model, frame_to_write)
                frame_to_write = annotated_a
                if accident_cls == "accident":
                    accident_found = True

            out.write(cv2.cvtColor(frame_to_write, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()
        st.success("Video processing completed.")
        st.video(annotated_video_path)

        # Insert logs
        logs_to_insert = []
        if detection_mode in ["Helmet Detection", "Both"]:
            helmet_result = "Helmet" if helmet_found else "No Helmet"
            logs_to_insert.append({"mode":"Helmet Detection", "result":helmet_result, "alert_sent":False})
        if detection_mode in ["Accident Detection", "Both"]:
            accident_result = "Accident" if accident_found else "No Accident"
            alert_sent = False
            if accident_found:
                msg = f"🚨 Accident detected in video!\n🕒 Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                alert_sent = send_telegram_message(msg)
                send_telegram_video(annotated_video_path, "🚨 Accident detected in video!")
                st.error("🚨 Accident detected in video! Telegram alert sent.")
            logs_to_insert.append({"mode":"Accident Detection", "result":accident_result, "alert_sent":alert_sent})

        for log_entry in logs_to_insert:
            insert_log(uploaded.name, log_entry["mode"], log_entry["result"], log_entry["alert_sent"], annotated_video_path, "Video detection")
