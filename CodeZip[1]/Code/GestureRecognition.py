import speech_recognition as sr
import pyttsx3
from PIL import Image
import os
import cv2
import threading  # Import threading for running audio listening in a separate thread

r = sr.Recognizer()
mic_active = False  # To track microphone state
audio_thread = None  # To hold the thread for listening to audio

def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

word_to_image = {
    "સફરજન": r"C:\Users\hp\Downloads\speechToText\speechToText\images\apple.jpg",
    "કેળું": r"C:\Users\hp\Downloads\speechToText\speechToText\images\banana.jpeg",
    "બિલાડી": r"C:\Users\hp\Downloads\speechToText\speechToText\images\cat.jpeg",
    # Add more words as needed...
}

def display_image(word):
    if word in word_to_image:
        image_path = word_to_image[word]
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img.show()
        else:
            print(f"Image for {word} not found at {image_path}")
    else:
        print(f"No image mapped for word: {word}")

def log_recognized_text(text):
    with open("recognized_text.txt", "a", encoding="utf-8") as file:
        file.write(text + "\n")

def listen_for_audio():
    global mic_active
    while mic_active:
        with sr.Microphone() as source2:
            print("Adjusting for ambient noise... Please wait.")
            r.adjust_for_ambient_noise(source2, duration=1)

            print("Listening...")
            audio2 = r.listen(source2)

            print("Recognizing speech...")
            try:
                MyText = r.recognize_google(audio2, language='gu-IN')
                MyText = MyText.lower()

                log_recognized_text(MyText)
                print(f"Recognized Text (logged to file): {MyText}")

                SpeakText(MyText)

                words = MyText.split()
                for word in words:
                    display_image(word)

            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio. Please try again.")

def toggle_microphone():
    global mic_active, audio_thread
    mic_active = not mic_active
    if mic_active:
        print("Microphone activated. Press 'm' again to deactivate.")
        audio_thread = threading.Thread(target=listen_for_audio)
        audio_thread.start()  # Start the audio listening in a separate thread
    else:
        print("Microphone deactivated.")

# Create a dummy OpenCV window to capture keyboard input
cv2.namedWindow("Microphone Control")

print("Press 'm' to toggle the microphone on/off, and 'q' to quit.")

while True:
    # Capture keyboard input
    key = cv2.waitKey(1)  # waits for a key press

    if key == ord('m'):  # Toggle microphone on/off
        toggle_microphone()
    
    if key == ord('q'):  # Quit the program
        print("Quitting the program.")
        mic_active = False  # Ensure the microphone is deactivated
        if audio_thread is not None:
            audio_thread.join()  # Wait for the thread to finish
        break

cv2.destroyAllWindows()  # Close the OpenCV window when done
