from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta

# Import data fetching utilities
from core.data_fetcher import (
    fetch_historical_data,
    fetch_realtime_data,
    fetch_technical_indicators
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Models for request and response
class StockDataRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: Optional[str] = "1d"  # 1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo

class TechnicalIndicatorRequest(BaseModel):
    ticker: str
    indicators: List[str]  # List of indicators to calculate
    period: Optional[int] = 14  # Default period for indicators
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# Endpoints
@router.get("/info/{ticker}")
async def get_stock_info(ticker: str):
    """Get basic information about a stock"""
    try:
        # This would be implemented in core.data_fetcher
        # For now, return a placeholder
        return {
            "ticker": ticker,
            "name": "Tata Motors Limited" if ticker == "TATAMOTORS.NS" else ticker,
            "sector": "Automotive",
            "industry": "Auto Manufacturers",
            "country": "India",
            "currency": "INR",
            "exchange": "NSE"
        }
    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock info: {str(e)}")

@router.post("/historical")
async def get_historical_data(request: StockDataRequest):
    """Get historical OHLCV data for a stock"""
    try:
        # Set default dates if not provided
        if not request.end_date:
            request.end_date = datetime.now().strftime("%Y-%m-%d")
        if not request.start_date:
            # Default to 1 year of data
            start_date = datetime.now() - timedelta(days=365)
            request.start_date = start_date.strftime("%Y-%m-%d")
            
        # This would call the actual implementation in core.data_fetcher
        data = await fetch_historical_data(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval
        )
        return {"data": data}
    except Exception as e:
        logger.error(f"Error fetching historical data for {request.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data: {str(e)}")

@router.get("/realtime/{ticker}")
async def get_realtime_data(ticker: str):
    """Get real-time stock data"""
    try:
        # This would call the actual implementation in core.data_fetcher
        data = await fetch_realtime_data(ticker=ticker)
        return data
    except Exception as e:
        logger.error(f"Error fetching real-time data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch real-time data: {str(e)}")

@router.post("/indicators")
async def get_technical_indicators(request: TechnicalIndicatorRequest):
    """Calculate technical indicators for a stock"""
    try:
        # This would call the actual implementation in core.data_fetcher
        indicators = await fetch_technical_indicators(
            ticker=request.ticker,
            indicators=request.indicators,
            period=request.period,
            start_date=request.start_date,
            end_date=request.end_date
        )
        return {"indicators": indicators}
    except Exception as e:
        logger.error(f"Error calculating indicators for {request.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate indicators: {str(e)}")

@router.get("/search")
async def search_stocks(query: str = Query(..., min_length=1)):
    """Search for stocks by name or ticker"""
    try:
        # This would be implemented to search for stocks
        # For now, return a placeholder
        if "tata" in query.lower():
            return {
                "results": [
                    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors Limited"},
                    {"ticker": "TATASTEEL.NS", "name": "Tata Steel Limited"},
                    {"ticker": "TCS.NS", "name": "Tata Consultancy Services Limited"}
                ]
            }
        return {"results": []}
    except Exception as e:
        logger.error(f"Error searching stocks with query '{query}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search stocks: {str(e)}")