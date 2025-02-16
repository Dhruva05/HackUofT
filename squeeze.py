import serial
import time
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Replace 'COM3' with the correct port for your HC-05 module
ser = serial.Serial('COM3', 9600, timeout=1)

app = FastAPI()

# Allow CORS from any origin for all endpoints (you can restrict it to specific origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or replace with specific origins like ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, adjust as needed
    allow_headers=["*"],  # Allow all headers
)

class Dimensions(BaseModel):
    width: int  # Only width is needed now

def send_data(data: int):
    ser.write(str(data).encode())  # Send the width as integer data to HC-05
    time.sleep(1)  # Wait for a second

@app.post("/send_dimensions")
async def send_dimensions(dimensions: Dimensions):
    width = dimensions.width
    send_data(width)  # Send only the width to HC-05
    return {"status": "success", "message": f"Sent width: {width}"}

@app.on_event("shutdown")
def shutdown():
    ser.close()


