import cv2
import base64
import numpy as np
from hand_detection import process_hand_detection
# OpenCV Video Capture
cap = cv2.VideoCapture(0)
async def process_video():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)  # Flip horizontally
        processed_frame = process_hand_detection(frame)
        # Encode frame to Base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')
        yield frame_encoded


