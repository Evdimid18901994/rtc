import streamlit as st
import cv2
import pafy
import numpy as np
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO("yolov8n.pt")

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
        try:
            # Получаем видео с помощью Pafy
            video = pafy.new(youtube_url)
            best = video.getbest(preftype="mp4")
            st.session_state["video_url"] = best.url
        except Exception as e:
            st.error(f"Ошибка при загрузке видео: {e}")
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