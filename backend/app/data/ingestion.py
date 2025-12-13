"""
Data ingestion layer for ED events.
"""
import logging
import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import UploadFile
import pandas as pd
from app.data.schemas import EDEvent, EventType, StaffingEvent
from app.data.storage import insert_events
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataIngester:
    """Handles data ingestion from various sources."""
    
    def __init__(self):
        self.rate_limiter = {}  # Simple in-memory rate limiter
    
    async def ingest_csv(
        self,
        file: UploadFile,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest data from CSV file.
        
        Expected CSV format:
        timestamp,event_type,patient_id,stage,resource_type,resource_id,duration_minutes
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"📥 Starting CSV ingestion: {file.filename}")
            print(f"[{time.strftime('%H:%M:%S')}] 📥 Starting CSV ingestion: {file.filename}")
            
            # Read CSV
            logger.info("📖 Reading file contents...")
            print(f"[{time.strftime('%H:%M:%S')}] 📖 Reading file contents...")
            contents = await file.read()
            file_size = len(contents)
            logger.info(f"📊 File size: {file_size / 1024:.2f} KB")
            print(f"[{time.strftime('%H:%M:%S')}] 📊 File size: {file_size / 1024:.2f} KB")
            
            # Reset file pointer for potential re-read
            await file.seek(0)
            
            # Use StringIO for text-based CSV reading
            import io
            logger.info("🔍 Parsing CSV with pandas...")
            print(f"[{time.strftime('%H:%M:%S')}] 🔍 Parsing CSV with pandas...")
            df = pd.read_csv(
                io.StringIO(contents.decode('utf-8')),
                parse_dates=["timestamp"],
                date_parser=pd.to_datetime
            )
            logger.info(f"📊 CSV parsed: {len(df)} rows, {len(df.columns)} columns")
            print(f"[{time.strftime('%H:%M:%S')}] 📊 CSV parsed: {len(df)} rows, {len(df.columns)} columns")
            
            # Validate and convert
            logger.info("✅ Validating and converting rows...")
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Validating and converting rows...")
            events = []
            invalid_count = 0
            
            for idx, (_, row) in enumerate(df.iterrows()):
                try:
                    event = self._row_to_event(row)
                    if validate:
                        # Validate using Pydantic
                        EDEvent(**event)
                    events.append(event)
                    
                    # Log progress for large files
                    if (idx + 1) % 1000 == 0:
                        logger.info(f"  Processed {idx + 1}/{len(df)} rows...")
                        print(f"[{time.strftime('%H:%M:%S')}]   Processed {idx + 1}/{len(df)} rows...")
                except Exception as e:
                    logger.warning(f"Invalid row {idx}: {e}")
                    invalid_count += 1
            
            logger.info(f"✅ Validation complete: {len(events)} valid, {invalid_count} invalid")
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Validation complete: {len(events)} valid, {invalid_count} invalid")
            
            # Insert into storage
            if events:
                logger.info(f"💾 Inserting {len(events)} events into storage...")
                print(f"[{time.strftime('%H:%M:%S')}] 💾 Inserting {len(events)} events into storage...")
                await insert_events(events)
                elapsed = time.time() - start_time
                print(f"✅ Data uploaded successfully: {len(events)} events processed in {elapsed:.3f}s")
                logger.info(f"Data uploaded: {len(events)} events in {elapsed:.3f}s")
                
                # Clear metrics cache (KPI calculation will be done separately via API)
                # This prevents timeout issues during upload
                try:
                    logger.info("🧹 Clearing metrics cache...")
                    print(f"[{time.strftime('%H:%M:%S')}] 🧹 Clearing metrics cache...")
                    from app.data.storage import cache_clear
                    await cache_clear("metrics_")
                    print("✅ Cache cleared - KPIs can be calculated via /ingest/calculate-kpis endpoint")
                    logger.info("Cache cleared")
                except Exception as e:
                    logger.warning(f"Cache clear failed: {e}")
                    print(f"⚠️  Cache clear failed: {e}")
            
            total_elapsed = time.time() - start_time
            logger.info(f"✅ CSV ingestion completed in {total_elapsed:.3f}s")
            print(f"[{time.strftime('%H:%M:%S')}] ✅ CSV ingestion completed in {total_elapsed:.3f}s")
            
            return {
                "status": "ok",
                "processed": len(events),
                "invalid": invalid_count,
                "total": len(df)
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ CSV ingestion failed after {elapsed:.3f}s: {e}", exc_info=True)
            print(f"[{time.strftime('%H:%M:%S')}] ❌ CSV ingestion failed after {elapsed:.3f}s: {e}")
            raise
    
    async def ingest_json(
        self,
        data: List[Dict[str, Any]],
        validate: bool = True
    ) -> Dict[str, Any]:
        """Ingest data from JSON payload."""
        try:
            events = []
            invalid_count = 0
            
            for item in data:
                try:
                    # Convert to event format
                    event = self._dict_to_event(item)
                    if validate:
                        EDEvent(**event)
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Invalid event: {e}")
                    invalid_count += 1
            
            # Rate limiting check
            if not self._check_rate_limit():
                raise ValueError("Rate limit exceeded")
            
            # Insert into storage
            if events:
                await insert_events(events)
                print(f"✅ Data uploaded successfully: {len(events)} events processed")
                logger.info(f"Data uploaded: {len(events)} events")
                
                try:
                    print("📊 Starting analysis (KPI calculation)...")
                    kpis = await self.calculate_kpis(window_hours=720)
                    kpi_count = len(kpis) if isinstance(kpis, list) else 0
                    print(f"✅ Analysis completed: {kpi_count} KPI records calculated")
                    logger.info(f"Analysis completed: {kpi_count} KPIs")
                    from app.data.storage import cache_clear
                    await cache_clear("metrics_")
                except Exception as e:
                    logger.warning(f"Auto KPI calc after JSON ingest failed: {e}")
                    print(f"⚠️  Analysis failed: {e}")
            
            return {
                "status": "ok",
                "processed": len(events),
                "invalid": invalid_count,
                "total": len(data)
            }
        
        except Exception as e:
            logger.error(f"JSON ingestion failed: {e}")
            raise
    
    def _row_to_event(self, row: pd.Series) -> Dict[str, Any]:
        """Convert CSV row to event dict."""
        # Helper to convert empty strings to None
        def to_none_if_empty(val):
            if pd.isna(val) or val == "" or val is None:
                return None
            return val
        
        # Helper to convert duration
        def parse_duration(val):
            if pd.isna(val) or val == "" or val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        
        # Helper to parse ESI (must be 1-5)
        def parse_esi(val):
            if pd.isna(val) or val == "" or val is None:
                return None
            try:
                esi = int(float(val))
                if 1 <= esi <= 5:
                    return esi
                return None
            except (ValueError, TypeError):
                return None
        
        return {
            "timestamp": pd.to_datetime(row["timestamp"]),
            "event_type": row["event_type"],
            "patient_id": str(row.get("patient_id", f"anon_{hash(str(row))}")),
            "stage": to_none_if_empty(row.get("stage")),
            "resource_type": to_none_if_empty(row.get("resource_type")),
            "resource_id": to_none_if_empty(row.get("resource_id")),
            "duration_minutes": parse_duration(row.get("duration_minutes")),
            "esi": parse_esi(row.get("esi")),
            "metadata": {}
        }
    
    def _dict_to_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dict to event format."""
        # Parse ESI if present (must be 1-5)
        esi = data.get("esi")
        if esi is not None:
            try:
                esi = int(float(esi))
                if not (1 <= esi <= 5):
                    esi = None
            except (ValueError, TypeError):
                esi = None
        
        return {
            "timestamp": pd.to_datetime(data["timestamp"]),
            "event_type": data["event_type"],
            "patient_id": str(data.get("patient_id", f"anon_{hash(str(data))}")),
            "stage": data.get("stage"),
            "resource_type": data.get("resource_type"),
            "resource_id": data.get("resource_id"),
            "duration_minutes": data.get("duration_minutes"),
            "esi": esi,
            "metadata": data.get("metadata", {})
        }
    
    def _check_rate_limit(self) -> bool:
        """Simple rate limiting check."""
        import time
        current_minute = int(time.time() / 60)
        
        if current_minute not in self.rate_limiter:
            self.rate_limiter[current_minute] = 0
        
        self.rate_limiter[current_minute] += 1
        
        # Clean old entries
        keys_to_remove = [k for k in self.rate_limiter.keys() if k < current_minute - 1]
        for k in keys_to_remove:
            del self.rate_limiter[k]
        
        return self.rate_limiter[current_minute] <= settings.MAX_EVENTS_PER_MINUTE
    
    async def calculate_kpis(
        self,
        window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Calculate KPIs from events and store them.
        
        This is a simplified version - in production, this would run
        as a background job or be triggered by new events.
        """
        from datetime import timedelta
        from app.data.storage import get_events, init_storage, insert_kpis_sqlite
        
        # Ensure storage is initialized
        await init_storage()
        
        # Get ALL events first to find the actual data range (not "now - window_hours")
        # Use a very wide date range to get all events
        all_events = await get_events(
            datetime(2000, 1, 1),  # Very early date to get all events
            datetime(2100, 1, 1)   # Very late date to get all events
        )
        
        if not all_events:
            logger.warning("No events found to calculate KPIs")
            return []
        
        # Find the actual time range from events
        event_times = [e["timestamp"] for e in all_events if e.get("timestamp")]
        if not event_times:
            logger.warning("No valid timestamps in events")
            return []
        
        # Use the actual data range, not "now"
        actual_end = max(event_times)
        actual_start = actual_end - timedelta(hours=window_hours)
        
        # Filter events to the window
        events = [
            e for e in all_events
            if e.get("timestamp") and actual_start <= e["timestamp"] <= actual_end
        ]
        
        if not events:
            logger.warning(f"No events in window {window_hours}h from actual data range")
            return []
        
        # Group by hour - use the actual event time range
        kpis = []
        if events:
            # Find the actual time range from events
            event_times = [e["timestamp"] for e in events]
            actual_start = min(event_times).replace(minute=0, second=0, microsecond=0)
            actual_end = max(event_times).replace(minute=59, second=59, microsecond=999999)
            
            current_hour = actual_start
            
            while current_hour <= actual_end:
                hour_end = current_hour + timedelta(hours=1)
                hour_events = [
                    e for e in events
                    if current_hour <= e["timestamp"] < hour_end
                ]
                
                if hour_events:
                    kpi = self._calculate_hourly_kpi(hour_events, current_hour)
                    kpis.append(kpi)
                
                current_hour = hour_end
        
        # Store KPIs in SQLite
        if kpis:
            try:
                insert_kpis_sqlite(kpis)
            except Exception as e:
                logger.error(f"Failed to store KPIs: {e}")
        
        return kpis
    
    def _calculate_hourly_kpi(
        self,
        events: List[Dict[str, Any]],
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Calculate KPI for a single hour."""
        # Extract patient journeys
        patients = {}
        for event in events:
            patient_id = event["patient_id"]
            if patient_id not in patients:
                patients[patient_id] = {
                    "arrival": None,
                    "doctor_visit": None,
                    "discharge": None,
                    "bed_assigned": False
                }
            
            if event["event_type"] == "arrival":
                patients[patient_id]["arrival"] = event["timestamp"]
            elif event["event_type"] == "doctor_visit":
                patients[patient_id]["doctor_visit"] = event["timestamp"]
            elif event["event_type"] == "discharge":
                patients[patient_id]["discharge"] = event["timestamp"]
            elif event["event_type"] == "bed_assign":
                patients[patient_id]["bed_assigned"] = True
        
        # Calculate DTD (Door-to-Doctor)
        dtd_values = []
        for patient_id, journey in patients.items():
            if journey["arrival"] and journey["doctor_visit"]:
                dtd = (journey["doctor_visit"] - journey["arrival"]).total_seconds() / 60
                dtd_values.append(dtd)
        
        avg_dtd = sum(dtd_values) / len(dtd_values) if dtd_values else 35.0
        
        # Calculate LOS (Length of Stay)
        los_values = []
        for patient_id, journey in patients.items():
            if journey["arrival"] and journey["discharge"]:
                los = (journey["discharge"] - journey["arrival"]).total_seconds() / 60
                los_values.append(los)
        
        avg_los = sum(los_values) / len(los_values) if los_values else 180.0
        
        # Calculate LWBS rate
        arrivals = len([e for e in events if e["event_type"] == "arrival"])
        lwbs = len([e for e in events if e["event_type"] == "lwbs"])
        lwbs_rate = lwbs / arrivals if arrivals > 0 else 0.0
        
        # Calculate bed utilization
        bed_events = [e for e in events if e.get("resource_type") == "bed"]
        bed_utilization = min(len(bed_events) / 20.0, 1.0) if bed_events else 0.0  # Assuming 20 beds
        
        # Calculate queue length (simplified)
        queue_length = len([e for e in events if e.get("stage") and e["event_type"] != "discharge"])
        
        return {
            "timestamp": timestamp,
            "dtd": avg_dtd,
            "los": avg_los,
            "lwbs": lwbs_rate,
            "bed_utilization": bed_utilization,
            "queue_length": queue_length
        }


        queue_length = len([e for e in events if e.get("stage") and e["event_type"] != "discharge"])
        
        return {
            "timestamp": timestamp,
            "dtd": avg_dtd,
            "los": avg_los,
            "lwbs": lwbs_rate,
            "bed_utilization": bed_utilization,
            "queue_length": queue_length
        }

