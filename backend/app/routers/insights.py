"""
Enhanced insights endpoint combining correlation, forecasting, and equity analysis.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime, timedelta
import logging
from app.core.correlation_analysis import CorrelationAnalyzer
from app.core.predictive_forecasting import PredictiveForecaster
from app.core.equity_analysis import EquityAnalyzer
from app.data.storage import get_events, get_kpis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/insights", tags=["insights"])


@router.get("/correlations")
async def get_correlations(window_hours: int = 48):
    """Get correlation analysis between patient types and outcomes."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)
        
        events = await get_events(start_time, end_time)
        kpis = await get_kpis(start_time, end_time)
        
        analyzer = CorrelationAnalyzer()
        correlations = await analyzer.analyze_correlations(events, kpis, window_hours)
        
        return {
            "status": "ok",
            "correlations": correlations,
            "window_hours": window_hours
        }
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecasts")
async def get_forecasts(window_hours: int = 48, horizon_hours: int = 2):
    """Get predictive forecasts for ED metrics."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)
        
        events = await get_events(start_time, end_time)
        kpis = await get_kpis(start_time, end_time)
        
        forecaster = PredictiveForecaster(horizon_hours=horizon_hours)
        forecasts = await forecaster.forecast_metrics(kpis, events, window_hours)
        
        return {
            "status": "ok",
            "forecasts": forecasts,
            "horizon_hours": horizon_hours,
            "window_hours": window_hours
        }
    except Exception as e:
        logger.error(f"Forecasting failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/equity")
async def get_equity_analysis(window_hours: int = 48):
    """Get equity analysis stratified by ESI, arrival mode, and time."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)
        
        events = await get_events(start_time, end_time)
        kpis = await get_kpis(start_time, end_time)
        
        analyzer = EquityAnalyzer()
        equity = await analyzer.analyze_equity(events, kpis, window_hours)
        
        return {
            "status": "ok",
            "equity": equity,
            "window_hours": window_hours
        }
    except Exception as e:
        logger.error(f"Equity analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comprehensive")
async def get_comprehensive_insights(window_hours: int = 48, horizon_hours: int = 2):
    """Get all insights: correlations, forecasts, and equity in one call."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)
        
        events = await get_events(start_time, end_time)
        kpis = await get_kpis(start_time, end_time)
        
        # Run all analyses in parallel
        import asyncio
        
        corr_analyzer = CorrelationAnalyzer()
        forecaster = PredictiveForecaster(horizon_hours=horizon_hours)
        equity_analyzer = EquityAnalyzer()
        
        correlations, forecasts, equity = await asyncio.gather(
            corr_analyzer.analyze_correlations(events, kpis, window_hours),
            forecaster.forecast_metrics(kpis, events, window_hours),
            equity_analyzer.analyze_equity(events, kpis, window_hours),
            return_exceptions=True
        )
        
        # Handle exceptions gracefully
        if isinstance(correlations, Exception):
            logger.warning(f"Correlation analysis failed: {correlations}")
            correlations = {}
        if isinstance(forecasts, Exception):
            logger.warning(f"Forecasting failed: {forecasts}")
            forecasts = {}
        if isinstance(equity, Exception):
            logger.warning(f"Equity analysis failed: {equity}")
            equity = {}
        
        return {
            "status": "ok",
            "correlations": correlations,
            "forecasts": forecasts,
            "equity": equity,
            "window_hours": window_hours,
            "horizon_hours": horizon_hours
        }
    except Exception as e:
        logger.error(f"Comprehensive insights failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

