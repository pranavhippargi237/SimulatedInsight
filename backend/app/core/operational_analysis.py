"""
Operational Flow Analysis: Deep dive into HOW operations cause bottlenecks.

Traces patient journeys, analyzes operational sequences, and provides
data-driven examples showing the mechanics of bottleneck formation.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from app.data.storage import get_events, get_kpis

logger = logging.getLogger(__name__)


@dataclass
class OperationalSequence:
    """A sequence of operations showing how a bottleneck forms."""
    sequence_id: str
    description: str
    operations: List[Dict[str, Any]]  # Each operation with timestamp, duration, wait
    total_time: float
    bottleneck_point: str  # Where the bottleneck occurs
    contributing_operations: List[str]  # Which operations contribute most


@dataclass
class PatientJourney:
    """Complete patient journey through the ED."""
    patient_id: str
    arrival_time: datetime
    esi: Optional[int]
    events: List[Dict[str, Any]]  # All events in order
    wait_times: Dict[str, float]  # Wait time at each stage
    service_times: Dict[str, float]  # Service time at each stage
    total_los: float
    outcome: str  # "discharge", "admit", "lwbs"


@dataclass
class OperationalMechanics:
    """How operations mechanically cause bottlenecks."""
    bottleneck_name: str
    operational_sequences: List[OperationalSequence]
    example_patient_journeys: List[PatientJourney]
    throughput_analysis: Dict[str, Any]
    utilization_analysis: Dict[str, Any]
    cycle_time_analysis: Dict[str, Any]
    data_driven_examples: List[Dict[str, Any]]


class OperationalAnalyzer:
    """
    Analyzes HOW operations cause bottlenecks by tracing patient journeys
    and operational sequences with actual data.
    """
    
    def __init__(self):
        pass
    
    async def analyze_operational_mechanics(
        self,
        bottleneck: Any,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int = 48
    ) -> OperationalMechanics:
        """
        Analyze HOW operations cause the bottleneck with data-driven examples.
        """
        logger.info(f"Analyzing operational mechanics for {bottleneck.bottleneck_name}")
        
        # 1. Build patient journeys
        patient_journeys = self._build_patient_journeys(events)
        
        # 2. Identify operational sequences that lead to bottleneck
        sequences = self._identify_operational_sequences(
            patient_journeys, bottleneck.stage
        )
        
        # 3. Analyze throughput (patients/hour at each stage)
        throughput = self._analyze_throughput(events, kpis, bottleneck.stage)
        
        # 4. Analyze resource utilization
        utilization = self._analyze_resource_utilization(events, bottleneck.stage)
        
        # 5. Analyze cycle times
        cycle_times = self._analyze_cycle_times(patient_journeys, bottleneck.stage)
        
        # 6. Generate data-driven examples
        examples = self._generate_data_driven_examples(
            patient_journeys, sequences, bottleneck
        )
        
        return OperationalMechanics(
            bottleneck_name=bottleneck.bottleneck_name,
            operational_sequences=sequences,
            example_patient_journeys=patient_journeys[:5],  # Top 5 examples
            throughput_analysis=throughput,
            utilization_analysis=utilization,
            cycle_time_analysis=cycle_times,
            data_driven_examples=examples
        )
    
    def _build_patient_journeys(
        self,
        events: List[Dict[str, Any]]
    ) -> List[PatientJourney]:
        """Build complete patient journeys from events."""
        journeys = {}
        
        # Group events by patient
        for event in events:
            patient_id = event.get("patient_id")
            if not patient_id:
                continue
            
            if patient_id not in journeys:
                journeys[patient_id] = {
                    "arrival_time": None,
                    "esi": None,
                    "events": [],
                    "wait_times": {},
                    "service_times": {},
                    "outcome": "unknown"
                }
            
            journeys[patient_id]["events"].append(event)
            
            # Track arrival
            if event.get("event_type") == "arrival":
                journeys[patient_id]["arrival_time"] = event.get("timestamp")
                journeys[patient_id]["esi"] = event.get("esi")
            
            # Track outcome
            if event.get("event_type") == "discharge":
                journeys[patient_id]["outcome"] = "discharge"
            elif event.get("event_type") == "lwbs":
                journeys[patient_id]["outcome"] = "lwbs"
        
        # Build PatientJourney objects
        patient_journeys = []
        for patient_id, journey_data in journeys.items():
            if not journey_data["arrival_time"]:
                continue
            
            # Sort events by timestamp
            events_sorted = sorted(
                journey_data["events"],
                key=lambda e: e.get("timestamp", datetime.min)
            )
            
            # Calculate wait times and service times
            wait_times = {}
            service_times = {}
            
            prev_event = None
            for event in events_sorted:
                event_type = event.get("event_type")
                timestamp = event.get("timestamp")
                duration = event.get("duration_minutes", 0)
                
                if prev_event:
                    # Wait time = time between previous event end and current event start
                    prev_end = prev_event.get("timestamp")
                    if prev_event.get("duration_minutes"):
                        prev_end += timedelta(minutes=prev_event.get("duration_minutes", 0))
                    
                    wait_time = (timestamp - prev_end).total_seconds() / 60
                    if wait_time > 0:
                        stage = self._get_stage_for_event(prev_event.get("event_type"))
                        if stage:
                            wait_times[stage] = wait_times.get(stage, 0) + wait_time
                
                # Service time
                if duration > 0:
                    stage = self._get_stage_for_event(event_type)
                    if stage:
                        service_times[stage] = duration
            
            # Calculate total LOS
            if events_sorted:
                arrival = events_sorted[0].get("timestamp")
                last_event = events_sorted[-1]
                last_time = last_event.get("timestamp")
                if last_event.get("duration_minutes"):
                    last_time += timedelta(minutes=last_event.get("duration_minutes", 0))
                total_los = (last_time - arrival).total_seconds() / 60
            else:
                total_los = 0
            
            patient_journeys.append(PatientJourney(
                patient_id=patient_id,
                arrival_time=journey_data["arrival_time"],
                esi=journey_data["esi"],
                events=events_sorted,
                wait_times=wait_times,
                service_times=service_times,
                total_los=total_los,
                outcome=journey_data["outcome"]
            ))
        
        # Sort by total LOS (longest first for bottleneck analysis)
        patient_journeys.sort(key=lambda x: x.total_los, reverse=True)
        
        return patient_journeys
    
    def _identify_operational_sequences(
        self,
        patient_journeys: List[PatientJourney],
        bottleneck_stage: str
    ) -> List[OperationalSequence]:
        """Identify operational sequences that lead to bottlenecks."""
        sequences = []
        
        # Find patients who experienced the bottleneck
        bottleneck_patients = [
            pj for pj in patient_journeys
            if bottleneck_stage in pj.wait_times and pj.wait_times[bottleneck_stage] > 15
        ]
        
        for i, patient in enumerate(bottleneck_patients[:10]):  # Top 10 examples
            operations = []
            total_time = 0
            
            prev_timestamp = patient.arrival_time
            bottleneck_point = None
            
            for event in patient.events:
                event_type = event.get("event_type")
                timestamp = event.get("timestamp")
                duration = event.get("duration_minutes", 0)
                
                # Calculate wait time
                wait_time = (timestamp - prev_timestamp).total_seconds() / 60
                if wait_time < 0:
                    wait_time = 0
                
                # Check if this is the bottleneck stage
                stage = self._get_stage_for_event(event_type)
                if stage == bottleneck_stage and wait_time > 15:
                    bottleneck_point = f"{event_type} at {timestamp.strftime('%H:%M')}"
                
                operations.append({
                    "operation": event_type,
                    "timestamp": timestamp.strftime('%H:%M:%S'),
                    "wait_time_minutes": wait_time,
                    "service_time_minutes": duration,
                    "stage": stage
                })
                
                total_time += wait_time + duration
                prev_timestamp = timestamp
                if duration > 0:
                    prev_timestamp += timedelta(minutes=duration)
            
            # Identify contributing operations (longest waits)
            contributing = sorted(
                operations,
                key=lambda x: x["wait_time_minutes"],
                reverse=True
            )[:3]
            
            sequences.append(OperationalSequence(
                sequence_id=f"patient_{patient.patient_id}",
                description=f"Patient {patient.patient_id[:8]} (ESI {patient.esi or 'N/A'}) journey showing {bottleneck_stage} bottleneck",
                operations=operations,
                total_time=total_time,
                bottleneck_point=bottleneck_point or f"{bottleneck_stage} stage",
                contributing_operations=[op["operation"] for op in contributing]
            ))
        
        return sequences
    
    def _analyze_throughput(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        bottleneck_stage: str
    ) -> Dict[str, Any]:
        """Analyze throughput (patients/hour) at each stage."""
        # Group events by hour
        events_by_hour = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            timestamp = event.get("timestamp")
            if not timestamp:
                continue
            
            hour = timestamp.replace(minute=0, second=0, microsecond=0)
            event_type = event.get("event_type")
            
            events_by_hour[hour][event_type] += 1
        
        # Calculate throughput for bottleneck stage
        stage_events = {
            "triage": "triage",
            "doctor": "doctor_visit",
            "bed": "bed_assign",
            "imaging": "imaging",
            "labs": "labs"
        }
        
        bottleneck_event = stage_events.get(bottleneck_stage)
        if not bottleneck_event:
            return {}
        
        throughput_by_hour = []
        for hour, event_counts in events_by_hour.items():
            arrivals = event_counts.get("arrival", 0)
            processed = event_counts.get(bottleneck_event, 0)
            
            throughput_by_hour.append({
                "hour": hour.strftime('%Y-%m-%d %H:00'),
                "arrivals": arrivals,
                "processed": processed,
                "backlog": arrivals - processed,
                "throughput_rate": processed / max(1, arrivals)  # % processed
            })
        
        # Calculate averages
        if throughput_by_hour:
            avg_arrivals = np.mean([t["arrivals"] for t in throughput_by_hour])
            avg_processed = np.mean([t["processed"] for t in throughput_by_hour])
            avg_backlog = np.mean([t["backlog"] for t in throughput_by_hour])
            
            return {
                "hourly_breakdown": throughput_by_hour,
                "average_arrivals_per_hour": avg_arrivals,
                "average_processed_per_hour": avg_processed,
                "average_backlog_per_hour": avg_backlog,
                "throughput_efficiency": avg_processed / max(1, avg_arrivals),
                "bottleneck_event": bottleneck_event
            }
        
        return {}
    
    def _analyze_resource_utilization(
        self,
        events: List[Dict[str, Any]],
        bottleneck_stage: str
    ) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        # Count active resources by time window
        stage_events = {
            "triage": "triage",
            "doctor": "doctor_visit",
            "bed": "bed_assign",
            "imaging": "imaging",
            "labs": "labs"
        }
        
        bottleneck_event = stage_events.get(bottleneck_stage)
        if not bottleneck_event:
            return {}
        
        # Group events by 15-minute windows
        events_by_window = defaultdict(lambda: {"events": [], "unique_resources": set()})
        
        for event in events:
            if event.get("event_type") == bottleneck_event:
                timestamp = event.get("timestamp")
                if not timestamp:
                    continue
                
                # Round to 15-minute window
                window = timestamp.replace(
                    minute=(timestamp.minute // 15) * 15,
                    second=0,
                    microsecond=0
                )
                
                events_by_window[window]["events"].append(event)
                resource_id = event.get("resource_id")
                if resource_id:
                    events_by_window[window]["unique_resources"].add(resource_id)
        
        # Calculate utilization metrics
        utilization_windows = []
        for window, data in sorted(events_by_window.items()):
            event_count = len(data["events"])
            resource_count = len(data["unique_resources"])
            
            # Estimate utilization (events per resource)
            utilization = event_count / max(1, resource_count) if resource_count > 0 else 0
            
            utilization_windows.append({
                "window": window.strftime('%Y-%m-%d %H:%M'),
                "events": event_count,
                "active_resources": resource_count,
                "events_per_resource": utilization,
                "utilization_rate": min(1.0, utilization / 10.0)  # Normalize (assume 10 events/hour max per resource)
            })
        
        if utilization_windows:
            avg_utilization = np.mean([u["utilization_rate"] for u in utilization_windows])
            max_utilization = max([u["utilization_rate"] for u in utilization_windows])
            peak_window = max(utilization_windows, key=lambda x: x["utilization_rate"])
            
            return {
                "utilization_by_window": utilization_windows,
                "average_utilization": avg_utilization,
                "peak_utilization": max_utilization,
                "peak_window": peak_window["window"],
                "peak_events_per_resource": peak_window["events_per_resource"]
            }
        
        return {}
    
    def _analyze_cycle_times(
        self,
        patient_journeys: List[PatientJourney],
        bottleneck_stage: str
    ) -> Dict[str, Any]:
        """Analyze cycle times (time from one stage to next)."""
        cycle_times = []
        
        for journey in patient_journeys:
            if bottleneck_stage not in journey.wait_times:
                continue
            
            # Find cycle time to bottleneck stage
            prev_stage = None
            for event in journey.events:
                stage = self._get_stage_for_event(event.get("event_type"))
                
                if stage == bottleneck_stage:
                    # Calculate time from previous stage
                    if prev_stage:
                        # Find time between previous stage end and bottleneck start
                        prev_events = [e for e in journey.events if self._get_stage_for_event(e.get("event_type")) == prev_stage]
                        if prev_events:
                            prev_end = prev_events[-1].get("timestamp")
                            if prev_events[-1].get("duration_minutes"):
                                prev_end += timedelta(minutes=prev_events[-1].get("duration_minutes", 0))
                            
                            bottleneck_start = event.get("timestamp")
                            cycle_time = (bottleneck_start - prev_end).total_seconds() / 60
                            
                            cycle_times.append({
                                "patient_id": journey.patient_id[:8],
                                "from_stage": prev_stage,
                                "to_stage": bottleneck_stage,
                                "cycle_time_minutes": cycle_time,
                                "wait_time_at_bottleneck": journey.wait_times.get(bottleneck_stage, 0)
                            })
                    break
                
                if stage and stage != bottleneck_stage:
                    prev_stage = stage
        
        if cycle_times:
            avg_cycle_time = np.mean([c["cycle_time_minutes"] for c in cycle_times])
            p95_cycle_time = np.percentile([c["cycle_time_minutes"] for c in cycle_times], 95)
            
            return {
                "cycle_times": cycle_times[:10],  # Top 10 examples
                "average_cycle_time_minutes": avg_cycle_time,
                "p95_cycle_time_minutes": p95_cycle_time,
                "bottleneck_stage": bottleneck_stage
            }
        
        return {}
    
    def _generate_data_driven_examples(
        self,
        patient_journeys: List[PatientJourney],
        sequences: List[OperationalSequence],
        bottleneck: Any
    ) -> List[Dict[str, Any]]:
        """Generate concrete data-driven examples showing HOW operations cause issues."""
        examples = []
        
        # Example 1: Show a specific patient journey
        if patient_journeys:
            example_patient = patient_journeys[0]  # Longest LOS
            
            example = {
                "type": "patient_journey",
                "title": f"Example Patient Journey: {example_patient.patient_id[:8]}",
                "description": f"Shows how operations led to {bottleneck.bottleneck_name}",
                "data": {
                    "patient_id": example_patient.patient_id[:8],
                    "esi": example_patient.esi,
                    "arrival_time": example_patient.arrival_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "total_los_minutes": example_patient.total_los,
                    "outcome": example_patient.outcome,
                    "events": [
                        {
                            "time": e.get("timestamp").strftime('%H:%M:%S'),
                            "event": e.get("event_type"),
                            "duration_minutes": e.get("duration_minutes", 0),
                            "wait_since_previous_minutes": self._calculate_wait_since_previous(
                                example_patient.events, i
                            )
                        }
                        for i, e in enumerate(example_patient.events)
                    ],
                    "wait_times_by_stage": example_patient.wait_times,
                    "service_times_by_stage": example_patient.service_times,
                    "bottleneck_analysis": {
                        "stage": bottleneck.stage,
                        "wait_time_minutes": example_patient.wait_times.get(bottleneck.stage, 0),
                        "contribution_to_los": (example_patient.wait_times.get(bottleneck.stage, 0) / max(1, example_patient.total_los)) * 100
                    }
                }
            }
            examples.append(example)
        
        # Example 2: Show operational sequence
        if sequences:
            example_sequence = sequences[0]
            
            example = {
                "type": "operational_sequence",
                "title": f"Operational Sequence Leading to {bottleneck.bottleneck_name}",
                "description": "Shows step-by-step how operations accumulate to create bottleneck",
                "data": {
                    "sequence_id": example_sequence.sequence_id,
                    "description": example_sequence.description,
                    "total_time_minutes": example_sequence.total_time,
                    "bottleneck_point": example_sequence.bottleneck_point,
                    "operations": example_sequence.operations[:10],  # First 10 operations
                    "contributing_operations": example_sequence.contributing_operations,
                    "analysis": {
                        "total_wait_time": sum(op["wait_time_minutes"] for op in example_sequence.operations),
                        "total_service_time": sum(op["service_time_minutes"] for op in example_sequence.operations),
                        "bottleneck_wait_contribution": sum(
                            op["wait_time_minutes"] for op in example_sequence.operations
                            if op.get("stage") == bottleneck.stage
                        ) / max(1, sum(op["wait_time_minutes"] for op in example_sequence.operations)) * 100
                    }
                }
            }
            examples.append(example)
        
        return examples
    
    def _calculate_wait_since_previous(
        self,
        events: List[Dict[str, Any]],
        current_index: int
    ) -> float:
        """Calculate wait time since previous event."""
        if current_index == 0:
            return 0.0
        
        prev_event = events[current_index - 1]
        current_event = events[current_index]
        
        prev_end = prev_event.get("timestamp")
        if prev_event.get("duration_minutes"):
            prev_end += timedelta(minutes=prev_event.get("duration_minutes", 0))
        
        current_start = current_event.get("timestamp")
        wait_time = (current_start - prev_end).total_seconds() / 60
        
        return max(0.0, wait_time)
    
    def _get_stage_for_event(self, event_type: str) -> Optional[str]:
        """Map event type to stage."""
        mapping = {
            "arrival": "arrival",
            "triage": "triage",
            "doctor_visit": "doctor",
            "bed_assign": "bed",
            "imaging": "imaging",
            "labs": "labs",
            "discharge": "discharge",
            "lwbs": "lwbs"
        }
        return mapping.get(event_type)


