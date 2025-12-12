"""
Real-time metrics endpoints.
"""
import logging
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime, timedelta
from app.data.storage import get_kpis, cache_get, cache_set
from app.data.schemas import KPI

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/metrics")
async def get_metrics(
    window: str = Query("24h", description="Time window (e.g., '24h', '48h')"),
    include_anomalies: bool = Query(True, description="Include anomaly detection")
):
    """
    Get real-time KPIs and metrics.
    
    Returns:
        - Current KPIs
        - Historical trends
        - Anomaly alerts (if enabled)
    """
    try:
        # Parse window
        window_hours = int(window.replace("h", ""))
        
        # Cache disabled (SQLite only) - always fetch fresh data
        # Get ALL KPIs first to find the actual data range
        from app.data.storage import get_all_kpis
        all_kpis = get_all_kpis(limit=10000)
        
        if not all_kpis or len(all_kpis) == 0:
            return {
                "status": "ok",
                "message": "No data available",
                "current_metrics": {
                    "dtd": 0,
                    "los": 0,
                    "lwbs": 0,
                    "bed_utilization": 0,
                    "queue_length": 0
                },
                "historical_kpis": [],
                "anomalies": []
            }
        
        # Sort by timestamp to get the most recent data
        all_kpis.sort(key=lambda k: k.get("timestamp", ""))
        
        # Get the most recent KPI timestamp (from the data, not from now)
        if isinstance(all_kpis[-1].get("timestamp"), str):
            latest_timestamp = datetime.fromisoformat(all_kpis[-1]["timestamp"].replace("Z", "+00:00"))
        else:
            latest_timestamp = all_kpis[-1].get("timestamp")
        
        # Calculate window from the latest data timestamp, not from now
        end_time = latest_timestamp if latest_timestamp else datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)
        
        # Filter KPIs to the requested window (relative to the latest data)
        kpis = [
            k for k in all_kpis
            if (isinstance(k.get("timestamp"), str) and 
                datetime.fromisoformat(k["timestamp"].replace("Z", "+00:00")) >= start_time) or
               (not isinstance(k.get("timestamp"), str) and k.get("timestamp") >= start_time)
        ]
        
        # If no KPIs in window, use the most recent ones available
        if not kpis or len(kpis) == 0:
            kpis = all_kpis[-min(window_hours, len(all_kpis)):]  # Get last N hours worth
        
        # Calculate current metrics based on the SELECTED WINDOW, not a fixed window
        # Use KPIs from the requested window to show metrics that change with window selection
        window_kpis = kpis  # Already filtered to the requested window
        
        # Filter out default values (35.0 for DTD, 180.0 for LOS) which indicate incomplete data
        valid_kpis = [
            k for k in window_kpis
            if not (abs(k.get("dtd", 0) - 35.0) < 0.01 and abs(k.get("los", 0) - 180.0) < 0.01)
        ]
        
        # If no valid KPIs in window, look at ALL KPIs for valid ones
        if not valid_kpis:
            valid_kpis = [
                k for k in all_kpis
                if not (abs(k.get("dtd", 0) - 35.0) < 0.01 and abs(k.get("los", 0) - 180.0) < 0.01)
            ]
        
        # Use ALL valid KPIs from the selected window to show metrics that change with window
        # This ensures 24h vs 48h windows show different averages
        kpis_to_use = valid_kpis if valid_kpis else window_kpis if window_kpis else []
        
        if kpis_to_use:
            # Calculate average across ALL KPIs in the selected window
            current_metrics = {
                "dtd": sum(k.get("dtd", 0) for k in kpis_to_use) / len(kpis_to_use),
                "los": sum(k.get("los", 0) for k in kpis_to_use) / len(kpis_to_use),
                "lwbs": sum(k.get("lwbs", 0) for k in kpis_to_use) / len(kpis_to_use),
                "bed_utilization": sum(k.get("bed_utilization", 0) for k in kpis_to_use) / len(kpis_to_use),
                "queue_length": int(sum(k.get("queue_length", 0) for k in kpis_to_use) / len(kpis_to_use))
            }
        else:
            # Fallback to latest KPI
            latest_kpi = all_kpis[-1] if all_kpis else None
            current_metrics = {
                "dtd": latest_kpi.get("dtd", 0) if latest_kpi else 0,
                "los": latest_kpi.get("los", 0) if latest_kpi else 0,
                "lwbs": latest_kpi.get("lwbs", 0) if latest_kpi else 0,
                "bed_utilization": latest_kpi.get("bed_utilization", 0) if latest_kpi else 0,
                "queue_length": latest_kpi.get("queue_length", 0) if latest_kpi else 0
            }
        
        # Detect anomalies if requested
        anomalies = []
        if include_anomalies:
            anomalies = await _detect_kpi_anomalies(kpis)
        
        result = {
            "status": "ok",
            "window_hours": window_hours,
            "current_metrics": current_metrics,
            "historical_kpis": kpis,
            "anomalies": anomalies,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache disabled - always return fresh data
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


async def _detect_kpi_anomalies(kpis: list) -> list:
    """Detect anomalies in KPI data."""
    if len(kpis) < 10:
        return []
    
    import numpy as np
    from scipy import stats
    
    anomalies = []
    
    # Check DTD
    dtd_values = [k["dtd"] for k in kpis]
    if len(dtd_values) > 0:
        z_scores = np.abs(stats.zscore(dtd_values))
        max_z = np.max(z_scores)
        if max_z > 2.0:
            max_idx = np.argmax(z_scores)
            anomalies.append({
                "metric": "dtd",
                "severity": "high" if max_z > 3.0 else "medium",
                "value": dtd_values[max_idx],
                "z_score": float(max_z),
                "timestamp": kpis[max_idx]["timestamp"].isoformat() if isinstance(kpis[max_idx]["timestamp"], datetime) else kpis[max_idx]["timestamp"]
            })
    
    # Check LWBS
    lwbs_values = [k["lwbs"] for k in kpis]
    if len(lwbs_values) > 0:
        z_scores = np.abs(stats.zscore(lwbs_values))
        max_z = np.max(z_scores)
        if max_z > 2.0:
            max_idx = np.argmax(z_scores)
            anomalies.append({
                "metric": "lwbs",
                "severity": "high" if max_z > 3.0 else "medium",
                "value": lwbs_values[max_idx],
                "z_score": float(max_z),
                "timestamp": kpis[max_idx]["timestamp"].isoformat() if isinstance(kpis[max_idx]["timestamp"], datetime) else kpis[max_idx]["timestamp"]
            })
    
    return anomalies

