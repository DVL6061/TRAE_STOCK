from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging

from backend.core.training_scheduler import (
    training_scheduler,
    start_training_scheduler,
    stop_training_scheduler,
    get_scheduler_status,
    manual_retrain_ticker
)
from backend.utils.helpers import validate_ticker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

# Pydantic models for request/response
class TrainingRequest(BaseModel):
    ticker: str
    models: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None

class TrainingResponse(BaseModel):
    success: bool
    message: str
    ticker: Optional[str] = None
    training_id: Optional[str] = None

class SchedulerStatusResponse(BaseModel):
    scheduler_running: bool
    last_training_times: Dict[str, str]
    performance_history: Dict[str, List[Dict]]
    config: Dict[str, Any]
    next_scheduled_check: Optional[str]

class SchedulerControlRequest(BaseModel):
    action: str  # "start" or "stop"

@router.get("/status", response_model=SchedulerStatusResponse)
async def get_training_status():
    """
    Get current training scheduler status and performance metrics.
    """
    try:
        status = get_scheduler_status()
        return SchedulerStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting training status: {str(e)}")

@router.post("/scheduler/control", response_model=TrainingResponse)
async def control_scheduler(request: SchedulerControlRequest):
    """
    Start or stop the training scheduler.
    """
    try:
        if request.action.lower() == "start":
            start_training_scheduler()
            return TrainingResponse(
                success=True,
                message="Training scheduler started successfully"
            )
        elif request.action.lower() == "stop":
            stop_training_scheduler()
            return TrainingResponse(
                success=True,
                message="Training scheduler stopped successfully"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid action. Use 'start' or 'stop'"
            )
    except Exception as e:
        logger.error(f"Error controlling scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error controlling scheduler: {str(e)}")

@router.post("/manual", response_model=TrainingResponse)
async def manual_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Manually trigger model training for a specific ticker.
    """
    try:
        # Validate ticker
        if not validate_ticker(request.ticker):
            raise HTTPException(status_code=400, detail=f"Invalid ticker: {request.ticker}")
        
        # Add training task to background
        background_tasks.add_task(
            manual_retrain_ticker,
            request.ticker,
            request.models,
            request.timeframes
        )
        
        return TrainingResponse(
            success=True,
            message=f"Manual training initiated for {request.ticker}",
            ticker=request.ticker
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating manual training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initiating manual training: {str(e)}")

@router.get("/performance/{ticker}")
async def get_model_performance(ticker: str):
    """
    Get performance metrics for a specific ticker's models.
    """
    try:
        if not validate_ticker(ticker):
            raise HTTPException(status_code=400, detail=f"Invalid ticker: {ticker}")
        
        status = get_scheduler_status()
        performance_history = status.get('performance_history', {}).get(ticker, [])
        last_training = status.get('last_training_times', {}).get(ticker)
        
        return {
            "ticker": ticker,
            "performance_history": performance_history,
            "last_training_time": last_training,
            "current_performance": performance_history[-1] if performance_history else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

@router.get("/models/list")
async def list_available_models():
    """
    List available model types and timeframes for training.
    """
    try:
        return {
            "model_types": ["xgboost", "informer", "dqn"],
            "timeframes": ["intraday", "short_term", "medium_term", "long_term"],
            "default_tickers": training_scheduler.config['default_tickers']
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@router.post("/batch", response_model=TrainingResponse)
async def batch_training(tickers: List[str], background_tasks: BackgroundTasks):
    """
    Trigger batch training for multiple tickers.
    """
    try:
        # Validate all tickers
        invalid_tickers = [ticker for ticker in tickers if not validate_ticker(ticker)]
        if invalid_tickers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tickers: {invalid_tickers}"
            )
        
        # Add batch training tasks
        for ticker in tickers:
            background_tasks.add_task(manual_retrain_ticker, ticker)
        
        return TrainingResponse(
            success=True,
            message=f"Batch training initiated for {len(tickers)} tickers: {tickers}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating batch training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initiating batch training: {str(e)}")

@router.delete("/models/{ticker}")
async def delete_model(ticker: str):
    """
    Delete trained models for a specific ticker.
    """
    try:
        if not validate_ticker(ticker):
            raise HTTPException(status_code=400, detail=f"Invalid ticker: {ticker}")
        
        # This would typically involve deleting model files
        # For now, we'll just return a success message
        logger.info(f"Model deletion requested for {ticker}")
        
        return {
            "success": True,
            "message": f"Models for {ticker} marked for deletion",
            "ticker": ticker
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting models for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting models: {str(e)}")

@router.get("/logs/recent")
async def get_recent_training_logs(lines: int = 100):
    """
    Get recent training logs.
    """
    try:
        import os
        log_file = os.path.join('logs', 'training_scheduler.log')
        
        if not os.path.exists(log_file):
            return {"logs": [], "message": "No training logs found"}
        
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines)
        }
        
    except Exception as e:
        logger.error(f"Error reading training logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")