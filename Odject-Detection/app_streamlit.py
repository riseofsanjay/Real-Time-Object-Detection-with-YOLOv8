import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Real-time (Web)", page_icon="🟡")

st.title("🟡 YOLOv8 Real‑time Object Detection (WebRTC)")
st.caption("Runs entirely on your machine. Allow camera access when prompted.")
st.caption("This app uses YOLOv8 for real-time object detection.")

# Sidebar controls
weights = st.sidebar.text_input("Weights path (use default for COCO):", "yolov8n.pt")
conf_thres = st.sidebar.slider("Confidence", 0.05, 0.95, 0.35, 0.01)
iou_thres = st.sidebar.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.01)
imgsz = st.sidebar.select_slider("Image size", options=[320, 416, 512, 640, 800], value=640)

@st.cache_resource(show_spinner=True)
def load_model(path):
    return YOLO(path)

model = load_model(weights)
names = model.model.names if hasattr(model, "model") else model.names

def draw(frame, boxes, classes, confs):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(classes[i]) if classes is not None else -1
        label = names.get(cls_id, str(cls_id))
        label = f"{label} {confs[i]:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(20, y1 - 10)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 4, y_text + 4), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(img, conf=conf_thres, iou=iou_thres, imgsz=imgsz, verbose=False)
    if len(results) > 0:
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else []
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        img = draw(img, boxes, classes, confs)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="yolo-webrtc",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)