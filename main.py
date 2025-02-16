from fastapi import FastAPI
import uvicorn
from websocket_handler import websocket_endpoint
app = FastAPI()
# Include WebSocket route
app.add_api_websocket_route("/ws", websocket_endpoint)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)