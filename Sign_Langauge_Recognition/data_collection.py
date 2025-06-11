import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from utils.mediapipe_utils import *
import keyboard

# Define the actions (signs) that will be recorded and stored in the dataset
actions = np.array([
    'okay', 'again', 'ready', 'sleep', 'wake', 'walk', 'run', 'sit', 'stand', 'open',
    'close', 'hot', 'cold', 'fast', 'slow', 'big', 'small', 'tired', 'hungry', 'full',
    'bathroom', 'clean', 'dirty', 'money', 'buy', 'sell', 'read', 'write', 'listen', 'speak',
    'call', 'text', 'phone', 'charge', 'battery', 'light', 'dark', 'cloudy', 'sunny', 'rain',
    'snow', 'wind', 'pain', 'sick', 'doctor', 'hospital', 'medicine', 'safe', 'danger', 'careful',
    'drive', 'car', 'bus', 'train', 'bike', 'road', 'traffic', 'left', 'right', 'up',
    'down', 'inside', 'outside', 'near', 'far', 'behind', 'front', 'together', 'alone', 'everyone',
    'family', 'brother', 'sister', 'baby', 'child', 'man', 'woman', 'boy', 'girl', 'old',
    'young', 'fun', 'play', 'game', 'music', 'dance', 'movie', 'tv', 'news', 'story',
    'time', 'hour', 'minute', 'second', 'day', 'week', 'month', 'year', 'now', 'later'
])

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
PATH = 'data'

for action, sequence in product(actions, range(sequences)):
    os.makedirs(os.path.join(PATH, action, str(sequence)), exist_ok=True)

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75,
                                    min_tracking_confidence=0.75) as holistic:
    try:
        for action, sequence, frame in product(actions, range(sequences), range(frames)):
            ret, image = cap.read()
            if not ret:
                print("Error: Unable to read from camera.")
                break

            # Immediately make the image writable
            image = image.copy()

            if frame == 0:
                while True:
                    if keyboard.is_pressed(' '):
                        break

                    ret, image = cap.read()
                    if not ret:
                        print("Error: Unable to read from camera.")
                        break

                    image = image.copy()
                    results = image_process(image, holistic)
                    draw_landmarks(image, results)

                    image = image.copy()
                    cv2.putText(image, f'Recording data for "{action}". Sequence {sequence}.',
                                (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Pause.', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'Press "Space" when you are ready.', (20, 450),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.imshow('Camera', image)
                    if cv2.waitKey(1) & 0xFF == 27:  # Exit if 'Esc' is pressed
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                    if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                        break
            else:
                ret, image = cap.read()
                if not ret:
                    print("Error: Unable to read from camera.")
                    break

                image = image.copy()
                results = image_process(image, holistic)
                draw_landmarks(image, results)

                image = image.copy()
                cv2.putText(image, f'Recording data for "{action}". Sequence {sequence}.',
                            (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                break

            # Extract landmarks and save
            keypoints = keypoint_extraction(results)
            frame_path = os.path.join(PATH, action, str(sequence), str(frame))
            np.save(frame_path, keypoints)

    finally:
        cap.release()
        cv2.destroyAllWindows()
