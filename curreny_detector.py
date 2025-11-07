import cv2
import numpy as np
import pyttsx3
import threading
import time
from tflite_runtime.interpreter import Interpreter
from gpiozero import Button

MODEL_PATH = "/home/pi/currency_model.tflite"
LABELS = ['10', '20', '50', '100', '200', '500', '2000']
BUTTON_PIN = 17
DETECTION_INTERVAL = 10  # seconds

print("Loading TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

camera = cv2.VideoCapture(0)
speaker = pyttsx3.init()
button = Button(BUTTON_PIN, pull_up=False)

detected_notes = []
lock = threading.Lock()

def speak(text):
    def _speak():
        speaker.say(text)
        speaker.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def predict_note(frame):
    try:
        img = cv2.resize(frame, (224, 224)).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        note = LABELS[np.argmax(preds)]
        return note
    except Exception as e:
        print("Prediction error:", e)
        return None

def announce_total():
    with lock:
        if not detected_notes:
            speak("No currency notes detected yet.")
            return
        total = sum(map(int, detected_notes))
        speak(f"The total amount is {total} rupees.")
        detected_notes.clear()
        print(f"Announced total: ₹{total}")

def button_pressed():
    print("Button pressed — announcing total.")
    announce_total()

button.when_pressed = button_pressed

def detection_loop():
    print("Currency detector started. Press the button anytime for total.")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Camera read failed, retrying...")
            time.sleep(1)
            continue

        note = predict_note(frame)
        if note:
            with lock:
                detected_notes.append(note)
            print(f"Detected ₹{note}")
            speak(f"The detected currency note is {note} rupees.")
        else:
            print("No note detected.")
        time.sleep(DETECTION_INTERVAL)

try:
    detection_loop()
except KeyboardInterrupt:
    print("Exiting...")
finally:
    camera.release()
