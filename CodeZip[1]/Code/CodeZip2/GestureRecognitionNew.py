import cv2
import mediapipe as mp
import numpy as np
import pickle
from gtts import gTTS
import os
import platform
import subprocess

# Load model
with open(r"C:\Users\hp\Desktop\CodeZip[1]\Code\gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Labels for digits 0-9
labels = [str(i) for i in range(10)]
cap = cv2.VideoCapture(0)
ans = []
predicted_label = ""

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Bounding box calculation
            x_min, x_max = float('inf'), -float('inf')
            y_min, y_max = float('inf'), -float('inf')
            h, w, _ = img.shape

            # Calculate the bounding box of the hand
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Check if the bounding box is within the frame
            if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                # Show a message on the screen if the hand is out of the boundary
                cv2.putText(img, "Hand out of boundary", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                # Skip prediction if part of the hand is outside the frame
                continue

            # Draw hand landmarks on the image
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect landmark data
            landmark_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])

            # Prediction
            landmark_flatten = np.array(landmark_list).flatten().reshape(1, -1)
            prediction = model.predict(landmark_flatten)
            predicted_label = labels[prediction[0]]

            # Display the predicted label
            cv2.putText(img, predicted_label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    if key == ord('s'):
        ans.append(predicted_label)

# Convert the final recognized text into speech
res = ' '.join(ans)
cap.release()
cv2.destroyAllWindows()
print(res)

# Text-to-Speech Conversion and Play
if res:
    # Language in which you want to convert (Gujarati)
    language = 'gu'

    # Convert the predicted text to speech
    try:
        myobj = gTTS(text=res, lang=language, slow=False)

        # Define the output path for the mp3 file
        output_path = r"C:\Users\hp\Desktop\recognized_output.mp3"

        # Saving the converted audio to the file
        myobj.save(output_path)
        print(f"Audio file saved at {output_path}")

    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")

    # Playing the converted file
    try:
        # Check the operating system to use the correct command
        if platform.system() == "Windows":
            subprocess.run(['start', output_path], shell=True)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(['open', output_path])
        else:  # Linux/Unix
            subprocess.run(['xdg-open', output_path])

        print("Playing the audio file...")

    except Exception as e:
        print(f"Error playing the audio file: {e}")
