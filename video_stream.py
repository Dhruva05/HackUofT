import cv2
import base64
import numpy as np
from hand_detection import process_white_ring_detection  # Import the updated function for white ring detection

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

async def process_video():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally

        # Process the frame to detect and outline white rings
        processed_frame = process_white_ring_detection(frame)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', processed_frame)
        
        # Encode the buffer to Base64
        frame_encoded = base64.b64encode(buffer).decode('utf-8')

        # Yield the Base64-encoded frame
        yield frame_encoded


