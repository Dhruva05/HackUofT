from fastapi import FastAPI, WebSocket
import cv2
import mediapipe as mp
import base64
import numpy as np
import asyncio
import uvicorn

app = FastAPI()

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define HSV color ranges for red, green, and blue tape detection
lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([170, 100, 100]), np.array([180, 255, 255])
lower_green, upper_green = np.array([35, 100, 100]), np.array([85, 255, 255])
lower_blue, upper_blue = np.array([100, 100, 100]), np.array([140, 255, 255])

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

async def process_video():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Color detection masks
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Create an empty mask for the hand region
        hand_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_points = np.array([(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in hand_landmarks.landmark], dtype=np.int32)
                cv2.fillConvexPoly(hand_mask, hand_points, 255)

                # Draw hand landmarks
                for point in hand_points:
                    cv2.circle(frame_bgr, tuple(point), 5, (255, 255, 255), -1)

                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Apply color masks within the hand region
        mask_red_hand = cv2.bitwise_and(mask_red, mask_red, mask=hand_mask)
        mask_green_hand = cv2.bitwise_and(mask_green, mask_green, mask=hand_mask)
        mask_blue_hand = cv2.bitwise_and(mask_blue, mask_blue, mask=hand_mask)

        # Find contours for color detection
        contours_red, _ = cv2.findContours(mask_red_hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green_hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue_hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the hand
        cv2.drawContours(frame_bgr, contours_red, -1, (0, 0, 255), 2)  # Red
        cv2.drawContours(frame_bgr, contours_green, -1, (0, 255, 0), 2)  # Green
        cv2.drawContours(frame_bgr, contours_blue, -1, (255, 0, 0), 2)  # Blue

        # Encode frame to Base64 for WebSocket
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')

        yield frame_encoded
        await asyncio.sleep(0.03)  # Stream at ~30 FPS

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async for frame_encoded in process_video():
            await websocket.send_text(f"data:image/jpeg;base64,{frame_encoded}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




