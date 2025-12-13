"""
Data ingestion endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List
from app.data.schemas import EDEvent
from app.data.ingestion import DataIngester
from app.data.storage import reset_sqlite, cache_clear

router = APIRouter()
ingester = DataIngester()


@router.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...), reset_first: bool = Query(True, description="Reset existing data before ingesting")):
    """
    Ingest ED events from CSV file.
    
    Expected format: timestamp,event_type,patient_id,stage,resource_type,resource_id,duration_minutes,esi
    
    By default, resets existing data before ingesting to ensure metrics reflect only the new CSV.
    Set reset_first=false to append data instead.
    """
    import asyncio
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        logger.info(f"📤 CSV upload request received: {file.filename}")
        print(f"[{time.strftime('%H:%M:%S')}] 📤 CSV upload request received: {file.filename}")
        
        if not file.filename.endswith('.csv'):
            logger.warning(f"❌ Invalid file type: {file.filename}")
            print(f"[{time.strftime('%H:%M:%S')}] ❌ Invalid file type: {file.filename}")
            raise HTTPException(status_code=422, detail="File must be CSV format")
        
        # Reset data first if requested (default: True)
        if reset_first:
            logger.info("🔄 Resetting existing data before upload...")
            print(f"[{time.strftime('%H:%M:%S')}] 🔄 Resetting existing data before upload...")
            reset_sqlite()
            await cache_clear()
            logger.info("✅ Data reset complete")
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Data reset complete")
        
        logger.info("📥 Starting CSV ingestion (timeout: 120s)...")
        print(f"[{time.strftime('%H:%M:%S')}] 📥 Starting CSV ingestion (timeout: 120s)...")
        
        # Add timeout to prevent hanging (120 seconds for large files - matches frontend)
        result = await asyncio.wait_for(
            ingester.ingest_csv(file),
            timeout=120.0
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ CSV ingestion completed in {elapsed:.3f}s: {result.get('processed', 0)} events processed")
        print(f"[{time.strftime('%H:%M:%S')}] ✅ CSV ingestion completed in {elapsed:.3f}s: {result.get('processed', 0)} events processed")
        
        return result
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"⏱️ CSV ingestion timed out after {elapsed:.3f}s")
        print(f"[{time.strftime('%H:%M:%S')}] ⏱️ CSV ingestion timed out after {elapsed:.3f}s")
        raise HTTPException(status_code=504, detail="Ingestion timed out. File may be too large. Please try a smaller file or split it into chunks.")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ CSV ingestion failed after {elapsed:.3f}s: {e}", exc_info=True)
        print(f"[{time.strftime('%H:%M:%S')}] ❌ CSV ingestion failed after {elapsed:.3f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/json")
async def ingest_json(events: List[EDEvent]):
    """
    Ingest ED events from JSON payload.
    """
    try:
        data = [event.dict() for event in events]
        result = await ingester.ingest_json(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/generate-advanced")
async def generate_advanced_data(
    num_patients: int = 500,
    days: int = 2,
    max_iterations: int = 5
):
    """
    Generate realistic ED data using the advanced generator with:
    - SDOH integration (transport delays, access scores)
    - Iterative validation (KS-tests)
    - Tuned parameters (LWBS 1.1-1.8%, LOS 4-5h)
    - Behavioral health tails
    
    The generated data is automatically ingested into the system.
    """
    try:
        from generate_sample_data_advanced import generate_events_validated
        from app.data.ingestion import DataIngester
        
        # Generate events
        events, validation = generate_events_validated(
            num_patients=num_patients,
            days=days,
            max_iterations=max_iterations
        )
        
        # Ingest the generated events
        ingester = DataIngester()
        result = await ingester.ingest_json([e for e in events])
        
        # Calculate KPIs
        kpis = await ingester.calculate_kpis(window_hours=days * 24)
        
        # Calculate statistics
        lwbs_count = sum(1 for e in events if e.get("event_type") == "lwbs")
        total_patients = len(set(e.get('patient_id') for e in events if e.get('patient_id')))
        lwbs_rate = (lwbs_count / total_patients * 100) if total_patients > 0 else 0
        
        return {
            "status": "ok",
            "message": "Advanced data generated and ingested successfully",
            "generated": {
                "total_events": len(events),
                "total_patients": total_patients,
                "lwbs_rate": round(lwbs_rate, 2),
                "validation_pass_rate": round(validation.get("pass_rate", 0) * 100, 1)
            },
            "ingested": result,
            "kpis_calculated": len(kpis) if isinstance(kpis, list) else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced data generation failed: {str(e)}")


@router.post("/ingest/calculate-kpis")
async def calculate_kpis(window_hours: int = 24):
    """
    Trigger KPI calculation from ingested events.
    """
    import asyncio
    
    try:
        # Add timeout to prevent hanging (60 seconds - increased for large datasets)
        kpis = await asyncio.wait_for(
            ingester.calculate_kpis(window_hours),
            timeout=60.0
        )
        return {
            "status": "ok",
            "kpis_calculated": len(kpis) if isinstance(kpis, list) else 0,
            "window_hours": window_hours
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="KPI calculation timed out. Please try again with a smaller time window.")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"KPI calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"KPI calculation failed: {str(e)}")


@router.post("/ingest/reset")
async def reset_data():
    """
    Reset all stored data (events, KPIs, staffing). Warning: destructive.
    """
    try:
        reset_sqlite()
        await cache_clear()
        return {"status": "ok", "message": "Data reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

