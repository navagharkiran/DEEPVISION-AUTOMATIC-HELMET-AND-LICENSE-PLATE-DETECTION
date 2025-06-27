import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO
import easyocr

# Load model and OCR
model = YOLO("best.pt")
reader = easyocr.Reader(['en'], gpu=False)
names = model.names
processed_ids = set()

# Polygon area
area = [(1, 173), (62, 468), (608, 431), (364, 155)]

# Setup log file
log_file = "logs.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Number Plate", "Date", "Time"]).to_csv(log_file, index=False)

# Streamlit UI
st.title("🛵 Helmet Violation & Number Plate Detection (EasyOCR)")
st.markdown("Upload an MP4 video to detect helmet violations and extract number plates.")

video = st.file_uploader("📤 Upload MP4 Video", type=["mp4"])

if video is not None:
    with open("temp_video.mp4", 'wb') as f:
        f.write(video.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    stframe = st.empty()
    ocr_status = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = model.track(frame, persist=True)

        no_helmet = False
        numberplate_box = None
        numberplate_id = None

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                label = names[cls_id]
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    if label == "no-helmet":
                        no_helmet = True
                    elif label == "numberplate":
                        numberplate_box = box
                        numberplate_id = track_id

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if no_helmet and numberplate_box and numberplate_id not in processed_ids:
                x1, y1, x2, y2 = numberplate_box
                crop = frame[y1:y2, x1:x2]

                # Optional: save for debugging
                timestamp = datetime.now().strftime('%H%M%S')
                debug_path = f"easy_debug_crop_{timestamp}.jpg"
                cv2.imwrite(debug_path, crop)

                # Run OCR with EasyOCR
                try:
                    result = reader.readtext(crop)
                    if result:
                        text = ''.join([line[1] for line in result])
                    else:
                        text = "NoText"
                    ocr_status.info(f"OCR Detected: {text}")
                except Exception as e:
                    text = "OCR_Failed"
                    ocr_status.warning("OCR failed: " + str(e))

                # Log to CSV
                now = datetime.now()
                date = now.strftime('%Y-%m-%d')
                time = now.strftime('%H:%M:%S')
                try:
                    pd.DataFrame([[text, date, time]],
                                 columns=["Number Plate", "Date", "Time"]).to_csv(
                        log_file, mode='a', index=False, header=False)
                except PermissionError:
                    st.warning("⚠️ Close logs.csv file if it’s open in Excel.")

                processed_ids.add(numberplate_id)

        cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 2)
        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("✅ Video processing complete!")

# Show and download logs
if os.path.exists(log_file):
    df = pd.read_csv(log_file)
    st.subheader("📄 Helmet Violations Log")
    st.dataframe(df)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="helmet_violations.csv")
