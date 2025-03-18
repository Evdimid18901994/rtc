import streamlit as st
import cv2
import yt_dlp
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO("yolov8n.pt")

# Функция загрузки YouTube-видео
def get_youtube_stream_url(video_url):
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        return info_dict.get("url", None)

# Функция обработки кадра с YOLOv8
def process_frame(img):
    res = model.predict(img, conf=0.5)
    img = res[0].plot()
    return img

# Интерфейс Streamlit
st.title("YOLOv8 YouTube Object Detection")

# Ввод ссылки на YouTube
youtube_url = st.text_input("Введите ссылку на YouTube-видео:")
if st.button("Запустить"):
    if youtube_url:
        stream_url = get_youtube_stream_url(youtube_url)
        if stream_url:
            st.session_state["video_url"] = stream_url
        else:
            st.error("Не удалось получить поток")
    else:
        st.error("Введите ссылку на YouTube")

# Запуск обработки видео, если есть видео
if "video_url" in st.session_state:
    cap = cv2.VideoCapture(st.session_state["video_url"])
    stframe = st.empty()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.warning("Видео закончилось или произошла ошибка.")
            break

        # Обработка кадра с YOLOv8
        processed_img = process_frame(img)

        # Отображение обработанного кадра
        stframe.image(processed_img, channels="BGR", use_container_width=True)

    cap.release()