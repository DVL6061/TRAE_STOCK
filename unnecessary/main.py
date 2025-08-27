import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import logging
from dotenv import load_dotenv

# Import API routers
from api.stock_data import router as stock_data_router
from api.predictions import router as predictions_router
from api.news import router as news_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction System",
    description="Enterprise-grade stock prediction system using ML, RL, and Transformers",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# Include routers
app.include_router(stock_data_router, prefix="/api/stock", tags=["Stock Data"])
app.include_router(predictions_router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(news_router, prefix="/api/news", tags=["News"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to the Stock Prediction System API"}

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}

# Serve static files (if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Set to False in production
    )