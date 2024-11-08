import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import pickle

from pydantic import BaseModel, ConfigDict

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Load the model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Labels for digits 0-9
labels = [str(i) for i in range(10)]

def predict_gesture(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Collect landmark data
            landmark_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])

            # Prediction
            landmark_flatten = np.array(landmark_list).flatten().reshape(1, -1)
            prediction = model.predict(landmark_flatten)
            predicted_label = labels[prediction[0]]

            return predicted_label
    else:
        return "No hand detected"

# Gradio Interface
iface = gr.Interface(fn=predict_gesture,
                     inputs="image",
                     outputs="text",
                     live=True)

iface.launch()
