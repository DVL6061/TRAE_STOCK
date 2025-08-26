from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import asyncio
import json
from datetime import datetime
from app.models.market import MarketData, PredictionData
from app.services.market_service import MarketService
from app.services.prediction_service import PredictionService

app = FastAPI(title="Stock Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                self.disconnect(connection)

manager = ConnectionManager()
market_service = MarketService()
prediction_service = PredictionService()

# WebSocket endpoint for real-time market data
@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Get real-time market data
            market_data = await market_service.get_real_time_data()
            await manager.broadcast({
                "type": "market_update",
                "payload": market_data.dict()
            })
            
            # Get real-time predictions
            predictions = await prediction_service.get_latest_predictions()
            await manager.broadcast({
                "type": "prediction_update",
                "payload": predictions.dict()
            })
            
            await asyncio.sleep(1)  # Update every second
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# REST endpoints
@app.get("/api/market/current")
async def get_current_market_data():
    return await market_service.get_current_data()

@app.get("/api/portfolio/summary")
async def get_portfolio_summary():
    return await market_service.get_portfolio_summary()

@app.get("/api/predictions/latest")
async def get_latest_predictions():
    return await prediction_service.get_latest_predictions()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)