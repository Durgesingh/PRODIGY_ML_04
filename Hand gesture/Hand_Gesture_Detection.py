'''
Task 04 : Develop a hand gesture recognition model that can accurately identify and 
classify different hand gestures from image or video data, 
enabling intuitive human-computer interaction and gesture-based control systems.
'''

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gesture_model = load_model('mp_hand_gesture')

with open('gesture.names', 'r') as file:
    class_names = file.read().split('\n')
print(class_names)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    frame_height, frame_width, _ = frame.shape
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    gesture_class = ''
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmark_x = int(landmark.x * frame_width)
                landmark_y = int(landmark.y * frame_height)
                landmarks.append([landmark_x, landmark_y])

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            prediction = gesture_model.predict([landmarks])
            class_id = np.argmax(prediction)
            gesture_class = class_names[class_id]

    cv2.putText(frame, gesture_class, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()