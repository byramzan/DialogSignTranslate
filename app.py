import logging
import queue
from collections import deque
import os
import pygame
from gtts import gTTS
from threading import Thread
import time

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from utils import SLInference

logger = logging.getLogger(__name__)

def speak(text):
    tts = gTTS(text=text, lang='ru')
    tts.save("temp.mp3")
    
    pygame.mixer.init()
    pygame.mixer.music.load("temp.mp3")
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    os.remove("temp.mp3")

def speak_async(gesture_queue):
    while True:
        if gesture_queue:
            text = gesture_queue.popleft()
            speak(text)
        time.sleep(0.1)

def main(config_path):
    """
    Main function of the app.
    """
    inference_thread = SLInference(config_path)
    inference_thread.start()

    webrtc_ctx = webrtc_streamer(
        key="video-sendonly",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": True},
    )

    gestures_deque = deque(maxlen=5)
    gesture_queue = deque()

    # Запуск потока для озвучивания жестов
    speak_thread = Thread(target=speak_async, args=(gesture_queue,))
    speak_thread.daemon = True
    speak_thread.start()

    # Set up Streamlit interface
    st.title("Диалог")
    image_place = st.empty()
    text_output = st.empty()
    last_5_gestures = st.empty()
    st.markdown(
        """
    Переводчик русского языка жестов.
        """
    )

    while True:
        if webrtc_ctx.video_receiver:
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                logger.warning("Очередь пустая")
                continue

            img_rgb = video_frame.to_ndarray(format="rgb24")
            image_place.image(img_rgb)
            inference_thread.input_queue.append(video_frame.reformat(224,224).to_ndarray(format="rgb24"))

            gesture = inference_thread.pred
            if gesture not in ['no', '']:
                if not gestures_deque or gesture != gestures_deque[-1]:
                    gestures_deque.append(gesture)
                    gesture_queue.append(gesture)  # добавляем жест в очередь на озвучивание

            text_output.markdown(f'<p style="font-size:20px"> Текущий жест: {gesture}</p>',
                                 unsafe_allow_html=True)
            last_5_gestures.markdown(f'<p style="font-size:20px"> Последние 5 жестов: {" ".join(gestures_deque)}</p>',
                                 unsafe_allow_html=True)
            print(gestures_deque)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main("configs/config.json")
