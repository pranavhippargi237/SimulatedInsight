"""
Deep Root Cause Analysis Engine for ED Bottlenecks.

Performs multi-level causal analysis:
1. Immediate causes (symptoms)
2. Underlying causes (process issues)
3. Systemic causes (structural problems)

Uses:
- Temporal pattern analysis
- Correlation and causal inference
- Multi-factor analysis
- Historical trend comparison
- Resource utilization deep-dive
"""
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy import stats
from app.data.storage import get_events, get_kpis
from app.data.schemas import Bottleneck

logger = logging.getLogger(__name__)


@dataclass
class RootCause:
    """A root cause with evidence and confidence."""
    level: str  # "immediate", "underlying", "systemic"
    cause_type: str  # "resource", "process", "temporal", "structural"
    description: str
    evidence: Dict[str, Any]  # Supporting data
    confidence: float  # 0-1
    impact_magnitude: float  # How much this contributes
    related_factors: List[str]  # Other related causes


@dataclass
class RootCauseAnalysis:
    """Complete root cause analysis for a bottleneck."""
    bottleneck: Bottleneck
    immediate_causes: List[RootCause]
    underlying_causes: List[RootCause]
    systemic_causes: List[RootCause]
    causal_chain: List[str]  # How causes connect
    contributing_factors: Dict[str, float]  # Factor -> contribution %
    recommendations: List[str]
    confidence: float
    operational_mechanics: Optional[Any] = None  # HOW operations cause issues


class RootCauseAnalyzer:
    """
    Deep root cause analyzer that goes beyond surface-level symptoms.
    
    Analyzes:
    - Temporal patterns (time-of-day, day-of-week, trends)
    - Resource utilization (actual vs optimal)
    - Process inefficiencies (wait times, handoffs)
    - Structural issues (capacity, layout, policies)
    - Causal chains (A → B → C)
    """
    
    def __init__(self):
        self.analysis_cache = {}
    
    async def analyze_bottleneck(
        self,
        bottleneck: Bottleneck,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int = 48
    ) -> RootCauseAnalysis:
        """
        Perform deep root cause analysis for a bottleneck.
        
        Returns comprehensive analysis with immediate, underlying, and systemic causes.
        """
        logger.info(f"Analyzing root causes for {bottleneck.bottleneck_name}")
        
        # 1. Immediate causes (what's happening right now)
        immediate = await self._analyze_immediate_causes(bottleneck, events, kpis)
        
        # 2. Underlying causes (process and resource issues)
        underlying = await self._analyze_underlying_causes(bottleneck, events, kpis)
        
        # 3. Systemic causes (structural and policy issues)
        systemic = await self._analyze_systemic_causes(bottleneck, events, kpis)
        
        # 4. Operational mechanics (HOW operations cause issues)
        from app.core.operational_analysis import OperationalAnalyzer
        op_analyzer = OperationalAnalyzer()
        operational_mechanics = await op_analyzer.analyze_operational_mechanics(
            bottleneck, events, kpis, window_hours
        )
        
        # 5. Build causal chain (enhanced with operational mechanics)
        causal_chain = self._build_causal_chain(immediate, underlying, systemic, operational_mechanics)
        
        # 6. Calculate contributing factors
        contributing = self._calculate_contributing_factors(immediate, underlying, systemic)
        
        # 7. Generate recommendations based on root causes
        recommendations = self._generate_root_cause_recommendations(
            immediate, underlying, systemic, operational_mechanics
        )
        
        # 8. Calculate overall confidence
        confidence = self._calculate_confidence(immediate, underlying, systemic)
        
        # Store operational mechanics in analysis
        analysis = RootCauseAnalysis(
            bottleneck=bottleneck,
            immediate_causes=immediate,
            underlying_causes=underlying,
            systemic_causes=systemic,
            causal_chain=causal_chain,
            contributing_factors=contributing,
            recommendations=recommendations,
            confidence=confidence
        )
        
        # Add operational mechanics to metadata
        analysis.operational_mechanics = operational_mechanics
        
        return analysis
    
    async def _analyze_immediate_causes(
        self,
        bottleneck: Bottleneck,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]]
    ) -> List[RootCause]:
        """Analyze immediate/symptomatic causes."""
        causes = []
        
        # 1. Resource availability at bottleneck stage
        stage_events = [e for e in events if self._event_matches_stage(e, bottleneck.stage)]
        
        if bottleneck.stage == "doctor":
            # Count active doctors during bottleneck period
            doctor_events = [e for e in stage_events if e.get("event_type") == "doctor_visit"]
            unique_doctors = len(set(e.get("resource_id", "") for e in doctor_events))
            
            # Compare to expected (baseline: 3 doctors for medium ED)
            expected_doctors = 3
            if unique_doctors < expected_doctors:
                shortage = expected_doctors - unique_doctors
                causes.append(RootCause(
                    level="immediate",
                    cause_type="resource",
                    description=f"Insufficient doctors: {unique_doctors} active vs {expected_doctors} expected ({shortage} short)",
                    evidence={
                        "active_doctors": unique_doctors,
                        "expected_doctors": expected_doctors,
                        "shortage": shortage,
                        "bottleneck_period_events": len(doctor_events)
                    },
                    confidence=0.9,
                    impact_magnitude=0.6,
                    related_factors=["Staffing", "Scheduling"]
                ))
        
        elif bottleneck.stage == "triage":
            nurse_events = [e for e in stage_events if e.get("event_type") == "triage"]
            unique_nurses = len(set(e.get("resource_id", "") for e in nurse_events))
            expected_nurses = 2
            
            if unique_nurses < expected_nurses:
                causes.append(RootCause(
                    level="immediate",
                    cause_type="resource",
                    description=f"Insufficient triage nurses: {unique_nurses} active vs {expected_nurses} expected",
                    evidence={
                        "active_nurses": unique_nurses,
                        "expected_nurses": expected_nurses
                    },
                    confidence=0.9,
                    impact_magnitude=0.7,
                    related_factors=["Triage staffing", "Arrival surge"]
                ))
        
        elif bottleneck.stage == "bed":
            # Bed utilization analysis
            bed_util_values = [k.get("bed_utilization", 0) for k in kpis if k.get("bed_utilization") is not None]
            if bed_util_values:
                avg_util = np.mean(bed_util_values)
                max_util = np.max(bed_util_values)
                
                if avg_util > 0.85 or max_util > 0.95:
                    causes.append(RootCause(
                        level="immediate",
                        cause_type="resource",
                        description=f"Bed capacity exceeded: {avg_util:.1%} average, {max_util:.1%} peak utilization",
                        evidence={
                            "avg_utilization": avg_util,
                            "max_utilization": max_util,
                            "threshold": 0.85
                        },
                        confidence=0.95,
                        impact_magnitude=0.8,
                        related_factors=["Bed capacity", "LOS", "Discharge delays"]
                    ))
        
        # 2. Queue length analysis
        queue_values = [k.get("queue_length", 0) for k in kpis if k.get("queue_length") is not None]
        if queue_values:
            avg_queue = np.mean(queue_values)
            max_queue = np.max(queue_values)
            
            if avg_queue > 8 or max_queue > 15:
                causes.append(RootCause(
                    level="immediate",
                    cause_type="process",
                    description=f"Excessive queue length: {avg_queue:.1f} average, {max_queue:.0f} peak patients waiting",
                    evidence={
                        "avg_queue": avg_queue,
                        "max_queue": max_queue,
                        "threshold": 8
                    },
                    confidence=0.85,
                    impact_magnitude=0.7,
                    related_factors=["Arrival rate", "Service rate", "Resource capacity"]
                ))
        
        # 3. Temporal surge analysis
        if len(kpis) >= 10:
            recent_kpis = kpis[-10:]
            baseline_kpis = kpis[:10] if len(kpis) >= 20 else kpis[:5]
            
            recent_dtd = np.mean([k.get("dtd", 0) for k in recent_kpis])
            baseline_dtd = np.mean([k.get("dtd", 0) for k in baseline_kpis])
            
            if recent_dtd > baseline_dtd * 1.3:  # 30% increase
                causes.append(RootCause(
                    level="immediate",
                    cause_type="temporal",
                    description=f"Recent surge: DTD increased from {baseline_dtd:.1f} to {recent_dtd:.1f} min ({((recent_dtd/baseline_dtd - 1)*100):.0f}% increase)",
                    evidence={
                        "baseline_dtd": baseline_dtd,
                        "recent_dtd": recent_dtd,
                        "increase_pct": ((recent_dtd/baseline_dtd - 1) * 100)
                    },
                    confidence=0.8,
                    impact_magnitude=0.6,
                    related_factors=["Arrival surge", "Time-of-day", "Day-of-week"]
                ))
        
        return causes
    
    async def _analyze_underlying_causes(
        self,
        bottleneck: Bottleneck,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]]
    ) -> List[RootCause]:
        """Analyze underlying process and resource issues."""
        causes = []
        
        # 1. Service time analysis (process efficiency)
        if bottleneck.stage in ["doctor", "triage", "bed"]:
            stage_events = [e for e in events if self._event_matches_stage(e, bottleneck.stage)]
            
            # Calculate service times
            service_times = []
            for event in stage_events:
                if event.get("duration_minutes"):
                    service_times.append(event["duration_minutes"])
            
            if len(service_times) >= 10:
                avg_service = np.mean(service_times)
                p95_service = np.percentile(service_times, 95)
                
                # Compare to benchmarks
                benchmarks = {
                    "triage": 5.0,  # 5 min
                    "doctor": 20.0,  # 20 min
                    "bed": 180.0  # 3 hours
                }
                benchmark = benchmarks.get(bottleneck.stage, avg_service)
                
                if avg_service > benchmark * 1.2:  # 20% above benchmark
                    causes.append(RootCause(
                        level="underlying",
                        cause_type="process",
                        description=f"Slow service times: {avg_service:.1f} min average (benchmark: {benchmark:.1f} min, {((avg_service/benchmark - 1)*100):.0f}% slower)",
                        evidence={
                            "avg_service_time": avg_service,
                            "p95_service_time": p95_service,
                            "benchmark": benchmark,
                            "sample_size": len(service_times)
                        },
                        confidence=0.85,
                        impact_magnitude=0.7,
                        related_factors=["Process efficiency", "Staff workload", "Patient complexity"]
                    ))
        
        # 2. Resource utilization efficiency
        if len(kpis) >= 15:
            # Calculate resource efficiency (throughput per resource)
            dtd_values = [k.get("dtd", 0) for k in kpis]
            bed_util = [k.get("bed_utilization", 0) for k in kpis]
            
            # Correlation: high bed util should correlate with high DTD if beds are the issue
            if len(bed_util) == len(dtd_values) and len(bed_util) >= 10:
                correlation = np.corrcoef(bed_util, dtd_values)[0, 1]
                
                if correlation > 0.6:
                    causes.append(RootCause(
                        level="underlying",
                        cause_type="resource",
                        description=f"Resource utilization bottleneck: Bed utilization strongly correlates with DTD (r={correlation:.2f}), indicating beds are constraining flow",
                        evidence={
                            "correlation": correlation,
                            "avg_bed_util": np.mean(bed_util),
                            "avg_dtd": np.mean(dtd_values)
                        },
                        confidence=0.8,
                        impact_magnitude=0.75,
                        related_factors=["Bed capacity", "Discharge process", "Admission delays"]
                    ))
        
        # 3. Arrival pattern mismatch
        if len(events) >= 50:
            # Analyze arrival patterns by hour
            arrivals_by_hour = defaultdict(int)
            for event in events:
                if event.get("event_type") == "arrival":
                    hour = event["timestamp"].hour
                    arrivals_by_hour[hour] += 1
            
            # Find peak hours
            if arrivals_by_hour:
                peak_hour = max(arrivals_by_hour.items(), key=lambda x: x[1])[0]
                peak_arrivals = arrivals_by_hour[peak_hour]
                avg_arrivals = np.mean(list(arrivals_by_hour.values()))
                
                if peak_arrivals > avg_arrivals * 1.5:
                    # Check if resources match peak
                    peak_events = [e for e in events 
                                  if e.get("event_type") in ["doctor_visit", "triage"] 
                                  and e["timestamp"].hour == peak_hour]
                    
                    if len(peak_events) < peak_arrivals * 0.5:  # <50% processed during peak
                        causes.append(RootCause(
                            level="underlying",
                            cause_type="temporal",
                            description=f"Peak hour mismatch: {peak_arrivals:.0f} arrivals at {peak_hour}:00 vs {avg_arrivals:.1f} average, but only {len(peak_events)} processed (resource allocation doesn't match demand)",
                            evidence={
                                "peak_hour": peak_hour,
                                "peak_arrivals": peak_arrivals,
                                "avg_arrivals": avg_arrivals,
                                "processed_during_peak": len(peak_events)
                            },
                            confidence=0.75,
                            impact_magnitude=0.65,
                            related_factors=["Staffing schedule", "Demand forecasting", "Resource allocation"]
                        ))
        
        # 4. ESI-acuity mismatch
        esi_events = [e for e in events if e.get("esi")]
        if len(esi_events) >= 30:
            esi_dist = defaultdict(int)
            for e in esi_events:
                esi_dist[e["esi"]] += 1
            
            total = len(esi_events)
            high_acuity_ratio = (esi_dist[1] + esi_dist[2]) / total
            low_acuity_ratio = (esi_dist[4] + esi_dist[5]) / total
            
            # If high ratio of low-acuity but they're waiting long, it's a routing issue
            if low_acuity_ratio > 0.4:  # >40% low acuity
                low_acuity_events = [e for e in events if e.get("esi") in [4, 5]]
                avg_wait_low = np.mean([e.get("wait_time_minutes", 0) for e in low_acuity_events if e.get("wait_time_minutes")]) if low_acuity_events else 0
                
                if avg_wait_low > 60:  # Low-acuity waiting >1 hour
                    causes.append(RootCause(
                        level="underlying",
                        cause_type="process",
                        description=f"Acuity-routing mismatch: {low_acuity_ratio:.1%} low-acuity patients (ESI 4-5) waiting {avg_wait_low:.0f} min average (should be fast-tracked)",
                        evidence={
                            "low_acuity_ratio": low_acuity_ratio,
                            "high_acuity_ratio": high_acuity_ratio,
                            "avg_wait_low_acuity": avg_wait_low
                        },
                        confidence=0.8,
                        impact_magnitude=0.6,
                        related_factors=["Fast-track", "Priority routing", "ESI triage"]
                    ))
        
        return causes
    
    async def _analyze_systemic_causes(
        self,
        bottleneck: Bottleneck,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]]
    ) -> List[RootCause]:
        """Analyze systemic/structural issues."""
        causes = []
        
        # 1. Capacity planning issues
        if len(kpis) >= 20:
            # Check if bottleneck is persistent (not just a spike)
            dtd_values = [k.get("dtd", 0) for k in kpis]
            los_values = [k.get("los", 0) for k in kpis]
            
            # Calculate trend
            if len(dtd_values) >= 10:
                x = np.arange(len(dtd_values))
                dtd_trend = np.polyfit(x, dtd_values, 1)[0]  # Slope
                
                if dtd_trend > 0.5:  # Increasing trend
                    causes.append(RootCause(
                        level="systemic",
                        cause_type="structural",
                        description=f"Persistent capacity issue: DTD increasing by {dtd_trend:.2f} min/hour over analysis period, indicating structural capacity mismatch",
                        evidence={
                            "trend_slope": dtd_trend,
                            "current_dtd": dtd_values[-1],
                            "baseline_dtd": dtd_values[0],
                            "period_hours": len(dtd_values)
                        },
                        confidence=0.75,
                        impact_magnitude=0.8,
                        related_factors=["Capacity planning", "Long-term staffing", "Infrastructure"]
                    ))
        
        # 2. Process design issues
        if len(events) >= 100:
            # Analyze handoff delays
            handoff_delays = []
            for i, event in enumerate(events):
                if event.get("event_type") == "triage":
                    # Find next doctor visit for same patient
                    patient_id = event.get("patient_id")
                    if patient_id:
                        doctor_visit = next((e for e in events[i+1:i+50] 
                                            if e.get("patient_id") == patient_id 
                                            and e.get("event_type") == "doctor_visit"), None)
                        if doctor_visit:
                            delay = (doctor_visit["timestamp"] - event["timestamp"]).total_seconds() / 60
                            if delay > 0:
                                handoff_delays.append(delay)
            
            if len(handoff_delays) >= 20:
                avg_handoff = np.mean(handoff_delays)
                if avg_handoff > 30:  # >30 min handoff
                    causes.append(RootCause(
                        level="systemic",
                        cause_type="process",
                        description=f"Process handoff delays: {avg_handoff:.1f} min average delay between triage and doctor visit (target: <15 min), indicating process design inefficiency",
                        evidence={
                            "avg_handoff_delay": avg_handoff,
                            "sample_size": len(handoff_delays),
                            "target": 15
                        },
                        confidence=0.8,
                        impact_magnitude=0.7,
                        related_factors=["Process design", "Workflow", "Communication"]
                    ))
        
        # 3. Policy/structural constraints
        if bottleneck.stage == "bed":
            # Check if bed bottleneck is due to discharge delays
            discharge_events = [e for e in events if e.get("event_type") == "discharge"]
            bed_events = [e for e in events if e.get("event_type") == "bed_assign"]
            
            if len(discharge_events) >= 20 and len(bed_events) >= 20:
                # Calculate bed turnover time
                bed_turnover_times = []
                for bed_event in bed_events:
                    patient_id = bed_event.get("patient_id")
                    if patient_id:
                        discharge = next((e for e in events 
                                         if e.get("patient_id") == patient_id 
                                         and e.get("event_type") == "discharge"), None)
                        if discharge:
                            turnover = (discharge["timestamp"] - bed_event["timestamp"]).total_seconds() / 60
                            if turnover > 0:
                                bed_turnover_times.append(turnover)
                
                if bed_turnover_times:
                    avg_turnover = np.mean(bed_turnover_times)
                    if avg_turnover > 240:  # >4 hours
                        causes.append(RootCause(
                            level="systemic",
                            cause_type="structural",
                            description=f"Bed turnover inefficiency: {avg_turnover:.0f} min average bed occupancy (target: <180 min), indicating discharge process or admission delays",
                            evidence={
                                "avg_turnover": avg_turnover,
                                "target": 180,
                                "sample_size": len(bed_turnover_times)
                            },
                            confidence=0.85,
                            impact_magnitude=0.75,
                            related_factors=["Discharge process", "Admission delays", "Bed management"]
                        ))
        
        return causes
    
    def _build_causal_chain(
        self,
        immediate: List[RootCause],
        underlying: List[RootCause],
        systemic: List[RootCause],
        operational_mechanics: Optional[Any] = None
    ) -> List[str]:
        """Build a causal chain showing how causes connect."""
        chain = []
        
        # Start with systemic causes (deepest)
        if systemic:
            chain.append(f"SYSTEMIC: {systemic[0].description}")
        
        # Then underlying causes
        if underlying:
            chain.append(f"→ UNDERLYING: {underlying[0].description}")
        
        # Finally immediate causes
        if immediate:
            chain.append(f"→ IMMEDIATE: {immediate[0].description}")
        
        # Add operational mechanics if available
        if operational_mechanics and operational_mechanics.data_driven_examples:
            example = operational_mechanics.data_driven_examples[0]
            if example.get("type") == "operational_sequence":
                chain.append(f"→ OPERATIONAL: {example['data']['bottleneck_point']} - {example['data']['analysis']['bottleneck_wait_contribution']:.0f}% of total wait")
        
        # Add bottleneck symptom
        chain.append("→ BOTTLENECK: Current wait times and delays")
        
        return chain
    
    def _calculate_contributing_factors(
        self,
        immediate: List[RootCause],
        underlying: List[RootCause],
        systemic: List[RootCause]
    ) -> Dict[str, float]:
        """Calculate how much each factor contributes."""
        factors = {}
        
        all_causes = immediate + underlying + systemic
        
        # Normalize impact magnitudes
        total_impact = sum(c.impact_magnitude for c in all_causes)
        if total_impact > 0:
            for cause in all_causes:
                for factor in cause.related_factors:
                    if factor not in factors:
                        factors[factor] = 0.0
                    factors[factor] += cause.impact_magnitude / total_impact
        
        # Normalize to percentages
        total = sum(factors.values())
        if total > 0:
            factors = {k: (v / total) * 100 for k, v in factors.items()}
        
        return factors
    
    def _generate_root_cause_recommendations(
        self,
        immediate: List[RootCause],
        underlying: List[RootCause],
        systemic: List[RootCause],
        operational_mechanics: Optional[Any] = None
    ) -> List[str]:
        """Generate recommendations based on root causes."""
        recommendations = []
        
        # Immediate fixes
        for cause in immediate:
            if cause.cause_type == "resource":
                recommendations.append(f"IMMEDIATE: Address {cause.description.lower()}")
            elif cause.cause_type == "temporal":
                recommendations.append(f"IMMEDIATE: Activate surge protocol for current period")
        
        # Underlying fixes
        for cause in underlying:
            if cause.cause_type == "process":
                recommendations.append(f"UNDERLYING: Review and optimize {cause.description.lower()}")
            elif cause.cause_type == "temporal":
                recommendations.append(f"UNDERLYING: Adjust staffing schedule to match demand patterns")
        
        # Systemic fixes
        for cause in systemic:
            if cause.cause_type == "structural":
                recommendations.append(f"SYSTEMIC: Long-term capacity planning review needed for {cause.description.lower()}")
            elif cause.cause_type == "process":
                recommendations.append(f"SYSTEMIC: Process redesign needed for {cause.description.lower()}")
        
        # Operational fixes based on mechanics
        if operational_mechanics:
            if operational_mechanics.throughput_analysis:
                throughput = operational_mechanics.throughput_analysis
                if throughput.get("throughput_efficiency", 1) < 0.8:
                    recommendations.append(
                        f"OPERATIONAL: Throughput efficiency only {throughput['throughput_efficiency']:.0%} - "
                        f"Only {throughput['average_processed_per_hour']:.1f} patients/hour processed vs "
                        f"{throughput['average_arrivals_per_hour']:.1f} arrivals/hour, creating backlog"
                    )
            
            if operational_mechanics.utilization_analysis:
                util = operational_mechanics.utilization_analysis
                if util.get("peak_utilization", 0) > 0.9:
                    recommendations.append(
                        f"OPERATIONAL: Peak utilization {util['peak_utilization']:.0%} at {util['peak_window']} - "
                        f"Resources overloaded with {util['peak_events_per_resource']:.1f} events/resource"
                    )
        
        return recommendations
    
    def _calculate_confidence(
        self,
        immediate: List[RootCause],
        underlying: List[RootCause],
        systemic: List[RootCause]
    ) -> float:
        """Calculate overall confidence in analysis."""
        all_causes = immediate + underlying + systemic
        if not all_causes:
            return 0.5
        
        # Weighted average by impact
        total_impact = sum(c.impact_magnitude for c in all_causes)
        if total_impact == 0:
            return 0.5
        
        weighted_confidence = sum(
            c.confidence * c.impact_magnitude for c in all_causes
        ) / total_impact
        
        return weighted_confidence
    
    def _event_matches_stage(self, event: Dict[str, Any], stage: str) -> bool:
        """Check if event matches bottleneck stage."""
        if stage == "triage":
            return event.get("event_type") == "triage"
        elif stage == "doctor":
            return event.get("event_type") == "doctor_visit"
        elif stage == "bed":
            return event.get("event_type") == "bed_assign"
        elif stage == "imaging":
            return event.get("event_type") == "imaging"
        elif stage == "labs":
            return event.get("event_type") == "labs"
        return False

