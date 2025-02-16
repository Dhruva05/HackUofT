from fastapi import WebSocket
from video_stream import process_video

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async for frame_encoded in process_video():
            await websocket.send_text(f"data:image/jpeg;base64,{frame_encoded}")
    except Exception as e:
        print(f"WebSocket error: {e}")