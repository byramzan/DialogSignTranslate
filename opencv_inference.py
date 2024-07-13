from collections import deque
import cv2
import os
import pygame
from gtts import gTTS
from threading import Thread
import time

from utils import SLInference

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

    cap = cv2.VideoCapture(0)

    inference_thread = SLInference(config_path)
    inference_thread.start()

    gestures_deque = deque(maxlen=5)
    gesture_queue = deque()

    # Запуск потока для озвучивания жестов
    speak_thread = Thread(target=speak_async, args=(gesture_queue,))
    speak_thread.daemon = True
    speak_thread.start()

    while True:
        _, img = cap.read()
        img_resized = cv2.resize(img, (224, 224))
        inference_thread.input_queue.append(img_resized)

        gesture = inference_thread.pred
        if gesture not in ['no', '']:
            if not gestures_deque or gesture != gestures_deque[-1]:
                gestures_deque.append(gesture)
                gesture_queue.append(gesture)  # добавляем жест в очередь на озвучивание

        if gestures_deque:
            print(gestures_deque[-1])

        cv2.imshow("img", img)
        cv2.waitKey(1)
        time.sleep(0.1)

if __name__ == "__main__":
    main("configs/config.json")
