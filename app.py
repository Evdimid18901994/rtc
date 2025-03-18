import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO("yolov8n.pt")


# Класс для обработки видеопотока
class YOLOVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Обработка кадра с YOLOv8
        results = model.predict(img, conf=0.5)
        processed_img = results[0].plot()

        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# Интерфейс Streamlit
st.title("YOLOv8 Object Detection via WebRTC")

webrtc_streamer(
    key="yolo-stream",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
