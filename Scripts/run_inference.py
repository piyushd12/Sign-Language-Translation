import numpy as np
import os
import string
import mediapipe as mp
import cv2
from utils.mediapipe_utils import *
import keyboard
from tensorflow.keras.models import load_model

PATH = os.path.join('data')

actions = np.array(os.listdir(PATH))

model = load_model('models/SLR.keras')

sentence, keypoints, last_prediction = [], [], ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():
        _, image = cap.read()
        results = image_process(image, holistic)
        draw_landmarks(image, results)
        keypoints.append(keypoint_extraction(results))

        # Check if 10 frames have been accumulated
        if len(keypoints) == 10:
            keypoints = np.array(keypoints)
            prediction = model.predict(keypoints[np.newaxis, :, :])

            # Clear the keypoints list for the next set of frames
            keypoints = []

            if np.amax(prediction) > 0.9:
                if last_prediction != actions[np.argmax(prediction)]:
                    sentence.append(actions[np.argmax(prediction)])
                    last_prediction = actions[np.argmax(prediction)]

        # Limit the sentence length to 7 elements to make sure it fits on the screen (change it according to screen size)
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset if the "Spacebar" is pressed
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction = [], [], ""

        if sentence:
            sentence[0] = sentence[0].capitalize()

        if len(sentence) >= 2:
            if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

        # Prepare the sentence to be displayed
        display_text = ' '.join(sentence)
        textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X_coord = (image.shape[1] - textsize[0]) // 2
        cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera', image)

        cv2.waitKey(1)

        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
