import cv2
import numpy as np

# Define color range for white (plastic rings)
lower_white = np.array([0, 0, 180])  # Lower bound for white color (for rings)
upper_white = np.array([180, 50, 255])  # Upper bound for white color (for rings)

def process_white_ring_detection(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask creation for white ring detection
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours in the white mask
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finger_tips = []  # List to store coordinates of detected white ring tips

    min_area = 100  # Filter small contours by area
    for contour in contours_white:
        if cv2.contourArea(contour) > min_area:
            # Get the center of the contour as a fingertip
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                finger_tips.append((cx, cy))
                # Draw contours for white rings
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)  # Blue for white rings

    return frame, finger_tips  # Return frame and fingertip coordinates
