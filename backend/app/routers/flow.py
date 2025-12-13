"""
Patient flow analysis endpoint for Sankey cascade visualization.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from app.data.storage import get_events

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/flow", tags=["flow"])


@router.get("/sankey")
async def get_patient_flow_sankey(
    window_hours: int = 24,
    stage_filter: Optional[str] = None
):
    """
    Get patient flow data for Sankey cascade visualization.
    
    Args:
        window_hours: Time window to analyze
        stage_filter: Optional stage to focus on (e.g., "doctor", "imaging")
    
    Returns:
        Flow data with stages, transitions, and patient counts
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)
        
        events = await get_events(start_time, end_time)
        
        if not events:
            return {
                "status": "ok",
                "stages": [],
                "transitions": [],
                "message": "No events found in time window"
            }
        
        # Build patient journeys
        patient_journeys = defaultdict(lambda: {
            "arrival": None,
            "triage": None,
            "doctor_visit": None,
            "labs": None,
            "imaging": None,
            "bed_assign": None,
            "discharge": None,
            "lwbs": None,
            "esi": None
        })
        
        for event in events:
            patient_id = event.get("patient_id")
            if not patient_id:
                continue
            
            event_type = event.get("event_type")
            timestamp = event.get("timestamp")
            
            if event_type == "arrival":
                patient_journeys[patient_id]["arrival"] = timestamp
                patient_journeys[patient_id]["esi"] = event.get("esi")
            elif event_type == "triage":
                patient_journeys[patient_id]["triage"] = timestamp
            elif event_type == "doctor_visit":
                patient_journeys[patient_id]["doctor_visit"] = timestamp
            elif event_type == "labs":
                patient_journeys[patient_id]["labs"] = timestamp
            elif event_type == "imaging":
                patient_journeys[patient_id]["imaging"] = timestamp
            elif event_type == "bed_assign":
                patient_journeys[patient_id]["bed_assign"] = timestamp
            elif event_type == "discharge":
                patient_journeys[patient_id]["discharge"] = timestamp
            elif event_type == "lwbs":
                patient_journeys[patient_id]["lwbs"] = timestamp
        
        # Calculate stage statistics
        stages = []
        stage_counts = defaultdict(int)
        stage_wait_times = defaultdict(list)
        
        for patient_id, journey in patient_journeys.items():
            # Count arrivals
            if journey["arrival"]:
                stage_counts["arrival"] += 1
            
            # Count triage
            if journey["triage"]:
                stage_counts["triage"] += 1
                if journey["arrival"]:
                    wait = (journey["triage"] - journey["arrival"]).total_seconds() / 60
                    if wait > 0 and wait < 120:
                        stage_wait_times["triage"].append(wait)
            
            # Count doctor visits
            if journey["doctor_visit"]:
                stage_counts["doctor"] += 1
                start_time = journey.get("triage") or journey.get("arrival")
                if start_time:
                    wait = (journey["doctor_visit"] - start_time).total_seconds() / 60
                    if wait > 0 and wait < 180:
                        stage_wait_times["doctor"].append(wait)
            
            # Count labs
            if journey["labs"]:
                stage_counts["labs"] += 1
                if journey["doctor_visit"]:
                    wait = (journey["labs"] - journey["doctor_visit"]).total_seconds() / 60
                    if wait > 0 and wait < 120:
                        stage_wait_times["labs"].append(wait)
            
            # Count imaging
            if journey["imaging"]:
                stage_counts["imaging"] += 1
                if journey["doctor_visit"]:
                    wait = (journey["imaging"] - journey["doctor_visit"]).total_seconds() / 60
                    if wait > 0 and wait < 120:
                        stage_wait_times["imaging"].append(wait)
            
            # Count bed assignments
            if journey["bed_assign"]:
                stage_counts["bed"] += 1
                if journey["doctor_visit"]:
                    wait = (journey["bed_assign"] - journey["doctor_visit"]).total_seconds() / 60
                    if wait > 0 and wait < 120:
                        stage_wait_times["bed"].append(wait)
            
            # Count outcomes
            if journey["discharge"]:
                stage_counts["discharge"] += 1
            if journey["lwbs"]:
                stage_counts["lwbs"] += 1
        
        # Build stage data
        stage_order = ["arrival", "triage", "doctor", "labs", "imaging", "bed", "discharge", "lwbs"]
        for stage_name in stage_order:
            count = stage_counts.get(stage_name, 0)
            if count > 0 or stage_name in ["arrival", "discharge", "lwbs"]:
                waits = stage_wait_times.get(stage_name, [])
                avg_wait = sum(waits) / len(waits) if waits else 0
                
                stages.append({
                    "name": stage_name,
                    "stage": stage_name,
                    "patient_count": count,
                    "avg_wait_minutes": round(avg_wait, 1) if avg_wait > 0 else None
                })
        
        # Calculate transitions
        transitions = []
        total_arrivals = stage_counts.get("arrival", 1)
        
        # Arrival → Triage
        triage_count = stage_counts.get("triage", 0)
        if triage_count > 0:
            transitions.append({
                "from": "arrival",
                "to": "triage",
                "count": triage_count,
                "percentage": (triage_count / total_arrivals) * 100,
                "is_bottleneck": stage_filter == "triage"
            })
        
        # Arrival/Triage → Doctor
        doctor_count = stage_counts.get("doctor", 0)
        if doctor_count > 0:
            transitions.append({
                "from": "triage" if triage_count > 0 else "arrival",
                "to": "doctor",
                "count": doctor_count,
                "percentage": (doctor_count / total_arrivals) * 100,
                "is_bottleneck": stage_filter == "doctor"
            })
        
        # Doctor → Labs
        labs_count = stage_counts.get("labs", 0)
        if labs_count > 0:
            transitions.append({
                "from": "doctor",
                "to": "labs",
                "count": labs_count,
                "percentage": (labs_count / doctor_count) * 100 if doctor_count > 0 else 0,
                "is_bottleneck": stage_filter == "labs"
            })
        
        # Doctor → Imaging
        imaging_count = stage_counts.get("imaging", 0)
        if imaging_count > 0:
            transitions.append({
                "from": "doctor",
                "to": "imaging",
                "count": imaging_count,
                "percentage": (imaging_count / doctor_count) * 100 if doctor_count > 0 else 0,
                "is_bottleneck": stage_filter == "imaging"
            })
        
        # Doctor → Bed
        bed_count = stage_counts.get("bed_assign", 0)
        if bed_count > 0:
            transitions.append({
                "from": "doctor",
                "to": "bed",
                "count": bed_count,
                "percentage": (bed_count / doctor_count) * 100 if doctor_count > 0 else 0,
                "is_bottleneck": stage_filter == "bed"
            })
        
        # Doctor → Discharge (direct)
        discharge_count = stage_counts.get("discharge", 0)
        if discharge_count > 0:
            transitions.append({
                "from": "doctor",
                "to": "discharge",
                "count": discharge_count,
                "percentage": (discharge_count / doctor_count) * 100 if doctor_count > 0 else 0,
                "is_bottleneck": False
            })
        
        # Any stage → LWBS
        lwbs_count = stage_counts.get("lwbs", 0)
        if lwbs_count > 0:
            # LWBS can happen from multiple stages, but we'll show from doctor as primary
            transitions.append({
                "from": "doctor",
                "to": "lwbs",
                "count": lwbs_count,
                "percentage": (lwbs_count / total_arrivals) * 100,
                "is_bottleneck": stage_filter == "lwbs"
            })
        
        return {
            "status": "ok",
            "stages": stages,
            "transitions": transitions,
            "total_patients": total_arrivals,
            "window_hours": window_hours,
            "bottleneck_stage": stage_filter
        }
    except Exception as e:
        logger.error(f"Flow analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

