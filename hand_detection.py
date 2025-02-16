import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add trusted host middleware to allow for proper host handling
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Define color range for white (plastic rings)
lower_white = np.array([0, 0, 180])  # Lower bound for white color (for rings)
upper_white = np.array([180, 50, 255])  # Upper bound for white color (for rings)

# Capture video
cap = cv2.VideoCapture(0)

def process_white_ring_detection(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask creation for white ring detection
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours in the white mask
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 100  # Filter small contours by area
    for contour in contours_white:
        if cv2.contourArea(contour) > min_area:
            # Draw contours for white rings
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)  # Blue for white rings

    return frame