import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio

# Решение проблемы с asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Загрузка модели YOLOv8
model = YOLO("yolov8n.pt")

# Класс для обработки видеопотока
class YOLOVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Обработка кадра с YOLOv8
        results = model.predict(img, conf=0.5)
        processed_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)  # Исправленный вывод

        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# Интерфейс Streamlit
st.title("YOLOv8 Object Detection via WebRTC")

webrtc_streamer(
    key="yolo-stream",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
