"""
Unified Deep Insight Engine for ED Operations.

Provides unique, actionable insights for ALL metrics and query types:
- DTD (Door-to-Doctor)
- LOS (Length of Stay)
- LWBS (Left Without Being Seen)
- Throughput
- Bed Utilization
- Resource Efficiency
- Patient Flow
- And more...

Focus: Unearth unmet needs and non-obvious insights that ED directors need.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy import stats
from app.data.storage import get_events, get_kpis, cache_get, cache_set
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """A unique insight with evidence and actionability."""
    insight_type: str  # "wait_time_cliff", "temporal_pattern", "resource_mismatch", etc.
    title: str
    description: str
    evidence: Dict[str, Any]
    impact_score: float  # 0-1
    confidence: float  # 0-1
    actionable: bool
    recommendation: Optional[str] = None
    unmet_need: Optional[str] = None  # What unmet need does this reveal?


@dataclass
class DeepAnalysis:
    """Comprehensive analysis with unique insights."""
    metric_name: str
    current_value: float
    benchmark_value: float
    insights: List[Insight]
    patterns: Dict[str, Any]
    root_causes: List[str]
    unmet_needs: List[str]  # High-level unmet needs revealed
    predictive_signals: Dict[str, Any]
    economic_impact: Optional[Dict[str, Any]] = None


class InsightEngine:
    """
    Unified engine for deriving unique insights across all ED metrics.
    """
    
    def __init__(self):
        self.benchmarks = {
            "dtd": 30.0,  # minutes (2025 target)
            "los": 240.0,  # minutes (2025 target)
            "lwbs": 0.015,  # 1.5% (2025 target)
            "bed_utilization": 0.85,  # 85% (optimal)
            "throughput": 2.5,  # patients/hour per bed
        }
    
    async def analyze(
        self,
        metric_name: str,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int = 48
    ) -> DeepAnalysis:
        """
        Perform deep analysis for any metric.
        Uses caching to improve performance.
        """
        # Create cache key based on metric name, data characteristics, and window
        data_hash = hashlib.md5(
            json.dumps({
                "event_count": len(events),
                "kpi_count": len(kpis),
                "window_hours": window_hours,
                "latest_timestamp": str(kpis[-1].get("timestamp")) if kpis else ""
            }, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        
        cache_key = f"deep_analysis_{metric_name.lower()}_{data_hash}_{window_hours}h"
        
        # Check cache first
        cached_result = await cache_get(cache_key)
        if cached_result:
            logger.info(f"Using cached deep analysis for {metric_name}")
            # Convert dict back to DeepAnalysis dataclass
            return DeepAnalysis(**cached_result)
        
        logger.info(f"Performing deep analysis for {metric_name}")
        
        # Route to specific analyzer
        result = None
        if metric_name.lower() in ["lwbs", "left without being seen"]:
            result = await self._analyze_lwbs(events, kpis, window_hours)
        elif metric_name.lower() in ["dtd", "door to doctor", "door-to-doctor"]:
            result = await self._analyze_dtd(events, kpis, window_hours)
        elif metric_name.lower() in ["los", "length of stay"]:
            result = await self._analyze_los(events, kpis, window_hours)
        elif metric_name.lower() in ["throughput", "patient flow"]:
            result = await self._analyze_throughput(events, kpis, window_hours)
        elif metric_name.lower() in ["bed", "bed utilization", "bed occupancy"]:
            result = await self._analyze_bed_utilization(events, kpis, window_hours)
        else:
            # Generic analysis
            result = await self._analyze_generic(metric_name, events, kpis, window_hours)
        
        # Cache result for 5 minutes (300 seconds)
        # Convert DeepAnalysis to dict for caching
        if result:
            result_dict = {
                "metric_name": result.metric_name,
                "current_value": result.current_value,
                "benchmark_value": result.benchmark_value,
                "insights": [
                    {
                        "insight_type": i.insight_type,
                        "title": i.title,
                        "description": i.description,
                        "evidence": i.evidence,
                        "impact_score": i.impact_score,
                        "confidence": i.confidence,
                        "actionable": i.actionable,
                        "recommendation": i.recommendation,
                        "unmet_need": i.unmet_need
                    }
                    for i in result.insights
                ],
                "patterns": result.patterns,
                "root_causes": result.root_causes,
                "unmet_needs": result.unmet_needs,
                "predictive_signals": result.predictive_signals,
                "economic_impact": result.economic_impact
            }
            await cache_set(cache_key, result_dict, ttl=300)
        
        return result
    
    async def _analyze_lwbs(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int
    ) -> DeepAnalysis:
        """Deep LWBS analysis with unique insights."""
        from app.core.lwbs_analysis import LWBSAnalyzer
        
        analyzer = LWBSAnalyzer()
        lwbs_analysis = await analyzer.analyze_lwbs(events, kpis, window_hours)
        
        # Convert to DeepAnalysis format
        insights = []
        unmet_needs = []
        
        # Wait time cliff insight
        if lwbs_analysis.wait_time_thresholds.get("total_wait_p75"):
            threshold = lwbs_analysis.wait_time_thresholds["total_wait_p75"]
            insights.append(Insight(
                insight_type="wait_time_cliff",
                title="Wait Time Cliff Effect",
                description=f"75% of LWBS patients leave after {threshold:.0f} minutes - this is a critical threshold where patience runs out",
                evidence={"threshold": threshold, "p75_wait": threshold},
                impact_score=0.95,
                confidence=0.9,
                actionable=True,
                recommendation=f"Implement real-time alerts when any patient waits >{threshold:.0f} min",
                unmet_need="Lack of proactive patient retention system"
            ))
            unmet_needs.append("Proactive patient retention system")
        
        # Primary exit point
        exit_stages = defaultdict(int)
        for patient in lwbs_analysis.lwbs_patients:
            stage = patient.get("stage_at_lwbs")
            if stage:
                exit_stages[stage] += 1
        
        if exit_stages:
            top_exit = max(exit_stages.items(), key=lambda x: x[1])
            pct = top_exit[1] / len(lwbs_analysis.lwbs_patients) if lwbs_analysis.lwbs_patients else 0
            insights.append(Insight(
                insight_type="exit_point_analysis",
                title="Primary Exit Point",
                description=f"{pct:.0%} of LWBS patients leave at {top_exit[0]} stage - this is where intervention is most critical",
                evidence={"stage": top_exit[0], "percentage": pct, "count": top_exit[1]},
                impact_score=0.9,
                confidence=0.95,
                actionable=True,
                recommendation=f"Focus resources on reducing {top_exit[0]} wait times - 10 min reduction could prevent {int(top_exit[1] * 0.3)} LWBS cases",
                unmet_need="Stage-specific intervention protocols"
            ))
            unmet_needs.append("Stage-specific intervention protocols")
        
        # ESI-specific risk
        esi_dist = defaultdict(int)
        for patient in lwbs_analysis.lwbs_patients:
            if patient.get("esi") is not None:
                esi_dist[patient["esi"]] += 1
        
        if esi_dist:
            high_risk_esi = max(esi_dist.items(), key=lambda x: x[1])
            if high_risk_esi[0] in [1, 2, 3]:  # High-acuity patients leaving
                insights.append(Insight(
                    insight_type="acuity_mismatch",
                    title="High-Acuity Patients Leaving",
                    description=f"ESI {high_risk_esi[0]} patients represent {high_risk_esi[1]/len(lwbs_analysis.lwbs_patients):.0%} of LWBS - counterintuitive finding",
                    evidence={"esi": high_risk_esi[0], "count": high_risk_esi[1]},
                    impact_score=0.85,
                    confidence=0.9,
                    actionable=True,
                    recommendation="Review triage protocols - high-acuity patients should not wait this long",
                    unmet_need="Acuity-based prioritization system"
                ))
                unmet_needs.append("Acuity-based prioritization system")
        
        # Enhanced root causes with step-function detail
        root_causes = []
        for rf in lwbs_analysis.risk_factors[:5]:
            if rf.impact_score > 0.7:
                root_causes.append(f"{rf.factor_name}: {rf.description} (Impact: {rf.impact_score*100:.0f}%, Confidence: {rf.confidence*100:.0f}%)")
        
        # Enhanced patterns with actionable insights
        enhanced_patterns = lwbs_analysis.temporal_patterns.copy() if lwbs_analysis.temporal_patterns else {}
        if lwbs_analysis.wait_time_thresholds:
            enhanced_patterns["critical_thresholds"] = {
                "p50": lwbs_analysis.wait_time_thresholds.get("total_wait_p50"),
                "p75": lwbs_analysis.wait_time_thresholds.get("total_wait_p75"),
                "p90": lwbs_analysis.wait_time_thresholds.get("total_wait_p90"),
                "interpretation": f"50% of LWBS patients leave after {lwbs_analysis.wait_time_thresholds.get('total_wait_p50', 0):.0f} min, 75% after {lwbs_analysis.wait_time_thresholds.get('total_wait_p75', 0):.0f} min"
            }
        
        # Enhanced predictive signals
        enhanced_predictive = lwbs_analysis.predictive_insights.copy() if lwbs_analysis.predictive_insights else {}
        if lwbs_analysis.current_lwbs_rate > lwbs_analysis.benchmark_lwbs_rate:
            excess_rate = lwbs_analysis.current_lwbs_rate - lwbs_analysis.benchmark_lwbs_rate
            enhanced_predictive["risk_forecast"] = {
                "current_trend": "elevated",
                "excess_rate": float(excess_rate),
                "next_24h_forecast": float(lwbs_analysis.current_lwbs_rate * 1.1) if excess_rate > 0.005 else float(lwbs_analysis.current_lwbs_rate),
                "intervention_urgency": "high" if excess_rate > 0.01 else "medium" if excess_rate > 0.005 else "low"
            }
        
        return DeepAnalysis(
            metric_name="LWBS",
            current_value=lwbs_analysis.current_lwbs_rate,
            benchmark_value=lwbs_analysis.benchmark_lwbs_rate,
            insights=insights,
            patterns=enhanced_patterns,
            root_causes=root_causes if root_causes else ["Insufficient data for root cause analysis"],
            unmet_needs=list(set(unmet_needs)),
            predictive_signals=enhanced_predictive,
            economic_impact=lwbs_analysis.economic_impact
        )
    
    async def _analyze_dtd(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int
    ) -> DeepAnalysis:
        """Deep DTD analysis with unique insights."""
        insights = []
        unmet_needs = []
        
        # Calculate current DTD
        arrivals = [e for e in events if e.get("event_type") == "arrival"]
        dtds = []
        
        for arrival in arrivals:
            patient_id = arrival.get("patient_id")
            if not patient_id:
                continue
            
            doctor_visit = next(
                (e for e in events
                 if e.get("patient_id") == patient_id
                 and e.get("event_type") == "doctor_visit"),
                None
            )
            
            if doctor_visit:
                dtd = (doctor_visit.get("timestamp") - arrival.get("timestamp")).total_seconds() / 60
                dtds.append(dtd)
        
        if not dtds:
            return DeepAnalysis(
                metric_name="DTD",
                current_value=0,
                benchmark_value=self.benchmarks["dtd"],
                insights=[],
                patterns={},
                root_causes=[],
                unmet_needs=[],
                predictive_signals={}
            )
        
        current_dtd = np.median(dtds)
        p95_dtd = np.percentile(dtds, 95)
        
        # Insight 1: DTD distribution reveals hidden bottlenecks
        if p95_dtd > current_dtd * 2:
            insights.append(Insight(
                insight_type="distribution_skew",
                title="Hidden Long-Tail Problem",
                description=f"P95 DTD ({p95_dtd:.0f} min) is {p95_dtd/current_dtd:.1f}x median ({current_dtd:.0f} min) - small subset of patients experience extreme waits",
                evidence={"median": current_dtd, "p95": p95_dtd, "ratio": p95_dtd/current_dtd},
                impact_score=0.8,
                confidence=0.9,
                actionable=True,
                recommendation="Investigate what causes the 5% tail - likely specific patient types or time periods",
                unmet_need="Tail-end patient tracking system"
            ))
            unmet_needs.append("Tail-end patient tracking system")
        
        # Insight 2: Temporal patterns
        dtd_by_hour = defaultdict(list)
        for arrival in arrivals:
            hour = arrival.get("timestamp").hour
            patient_id = arrival.get("patient_id")
            doctor_visit = next(
                (e for e in events
                 if e.get("patient_id") == patient_id
                 and e.get("event_type") == "doctor_visit"),
                None
            )
            if doctor_visit:
                dtd = (doctor_visit.get("timestamp") - arrival.get("timestamp")).total_seconds() / 60
                dtd_by_hour[hour].append(dtd)
        
        if dtd_by_hour:
            avg_dtd_by_hour = {h: np.mean(dtds) for h, dtds in dtd_by_hour.items() if dtds}
            if avg_dtd_by_hour:
                peak_hour = max(avg_dtd_by_hour.items(), key=lambda x: x[1])
                if peak_hour[1] > current_dtd * 1.3:
                    insights.append(Insight(
                        insight_type="temporal_mismatch",
                        title="Resource-Timing Mismatch",
                        description=f"{peak_hour[0]}:00 hour has {peak_hour[1]:.0f} min DTD ({peak_hour[1]/current_dtd:.1f}x average) - resources not aligned with demand",
                        evidence={"peak_hour": peak_hour[0], "peak_dtd": peak_hour[1], "avg_dtd": current_dtd},
                        impact_score=0.85,
                        confidence=0.9,
                        actionable=True,
                        recommendation=f"Shift resources to {peak_hour[0]}:00 hour or implement surge protocols",
                        unmet_need="Dynamic resource allocation system"
                    ))
                    unmet_needs.append("Dynamic resource allocation system")
        
        # Insight 3: ESI-based DTD
        dtd_by_esi = defaultdict(list)
        for arrival in arrivals:
            esi = arrival.get("esi")
            if esi is None:
                continue
            patient_id = arrival.get("patient_id")
            doctor_visit = next(
                (e for e in events
                 if e.get("patient_id") == patient_id
                 and e.get("event_type") == "doctor_visit"),
                None
            )
            if doctor_visit:
                dtd = (doctor_visit.get("timestamp") - arrival.get("timestamp")).total_seconds() / 60
                dtd_by_esi[esi].append(dtd)
        
        if dtd_by_esi:
            avg_dtd_by_esi = {esi: np.mean(dtds) for esi, dtds in dtd_by_esi.items() if dtds}
            # Check if high-acuity patients wait longer than low-acuity (problem!)
            if 1 in avg_dtd_by_esi and 5 in avg_dtd_by_esi:
                if avg_dtd_by_esi[1] > avg_dtd_by_esi[5]:
                    insights.append(Insight(
                        insight_type="acuity_inversion",
                        title="Acuity Priority Inversion",
                        description=f"ESI 1 patients wait {avg_dtd_by_esi[1]:.0f} min vs ESI 5 wait {avg_dtd_by_esi[5]:.0f} min - critical patients waiting longer!",
                        evidence={"esi1_dtd": avg_dtd_by_esi[1], "esi5_dtd": avg_dtd_by_esi[5]},
                        impact_score=0.95,
                        confidence=0.9,
                        actionable=True,
                        recommendation="URGENT: Review triage and prioritization - critical patients must be seen first",
                        unmet_need="Real-time acuity-based prioritization"
                    ))
                    unmet_needs.append("Real-time acuity-based prioritization")
        
        # Enhanced root causes with step-function analysis
        root_causes = []
        if p95_dtd > current_dtd * 2:
            root_causes.append("Long-tail distribution indicates systemic bottlenecks affecting 5% of patients disproportionately")
        if avg_dtd_by_hour:
            peak_hour = max(avg_dtd_by_hour.items(), key=lambda x: x[1])
            if peak_hour[1] > current_dtd * 1.3:
                root_causes.append(f"Resource allocation mismatch: {peak_hour[0]}:00 hour experiences {peak_hour[1]/current_dtd:.1f}x average wait times")
        if avg_dtd_by_esi:
            high_esi_dtd = avg_dtd_by_esi.get(4, []) or avg_dtd_by_esi.get(5, [])
            if high_esi_dtd and isinstance(high_esi_dtd, (int, float)) and high_esi_dtd > current_dtd * 1.2:
                root_causes.append("Low-acuity patients (ESI 4-5) experiencing longer waits than expected - triage prioritization may be ineffective")
        
        # Enhanced patterns with actionable data
        patterns = {
            "by_hour": avg_dtd_by_hour,
            "by_esi": avg_dtd_by_esi,
            "distribution": {
                "median": float(current_dtd),
                "p75": float(np.percentile(dtds, 75)),
                "p95": float(p95_dtd),
                "p99": float(np.percentile(dtds, 99)) if len(dtds) > 1 else float(current_dtd),
                "skew": float(stats.skew(dtds)) if len(dtds) > 2 else 0.0
            }
        }
        
        if avg_dtd_by_hour:
            peak_hour = max(avg_dtd_by_hour.items(), key=lambda x: x[1])
            patterns["peak_period"] = {
                "hour": int(peak_hour[0]),
                "dtd": float(peak_hour[1]),
                "multiplier": float(peak_hour[1]/current_dtd) if current_dtd > 0 else 1.0
            }
        
        # Enhanced predictive signals
        predictive_signals = {}
        if len(dtds) >= 10:
            recent_dtds = dtds[-min(20, len(dtds)):]
            trend = np.polyfit(range(len(recent_dtds)), recent_dtds, 1)[0] if len(recent_dtds) > 1 else 0.0
            predictive_signals = {
                "trend": "increasing" if trend > 0.5 else "decreasing" if trend < -0.5 else "stable",
                "trend_magnitude": float(abs(trend)),
                "forecast_next_hour": float(current_dtd + trend),
                "risk_level": "high" if trend > 1.0 else "medium" if trend > 0.5 else "low"
            }
        
        return DeepAnalysis(
            metric_name="DTD",
            current_value=current_dtd,
            benchmark_value=self.benchmarks["dtd"],
            insights=insights,
            patterns=patterns,
            root_causes=root_causes if root_causes else ["Insufficient data for root cause analysis"],
            unmet_needs=list(set(unmet_needs)),
            predictive_signals=predictive_signals
        )
    
    async def _analyze_los(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int
    ) -> DeepAnalysis:
        """Deep LOS analysis with unique insights."""
        insights = []
        unmet_needs = []
        
        # Calculate LOS for discharged patients
        arrivals = [e for e in events if e.get("event_type") == "arrival"]
        los_values = []
        
        for arrival in arrivals:
            patient_id = arrival.get("patient_id")
            if not patient_id:
                continue
            
            discharge = next(
                (e for e in events
                 if e.get("patient_id") == patient_id
                 and e.get("event_type") in ["discharge", "lwbs"]),
                None
            )
            
            if discharge:
                los = (discharge.get("timestamp") - arrival.get("timestamp")).total_seconds() / 60
                los_values.append(los)
        
        if not los_values:
            return DeepAnalysis(
                metric_name="LOS",
                current_value=0,
                benchmark_value=self.benchmarks["los"],
                insights=[],
                patterns={},
                root_causes=[],
                unmet_needs=[],
                predictive_signals={}
            )
        
        current_los = np.median(los_values)
        p95_los = np.percentile(los_values, 95)
        
        # Insight 1: LOS by disposition
        los_by_disposition = defaultdict(list)
        for arrival in arrivals:
            patient_id = arrival.get("patient_id")
            if not patient_id:
                continue
            
            discharge = next(
                (e for e in events
                 if e.get("patient_id") == patient_id
                 and e.get("event_type") in ["discharge", "lwbs", "admission"]),
                None
            )
            
            if discharge:
                los = (discharge.get("timestamp") - arrival.get("timestamp")).total_seconds() / 60
                disposition = discharge.get("event_type", "discharge")
                los_by_disposition[disposition].append(los)
        
        if los_by_disposition:
            avg_los_by_disp = {d: np.mean(los) for d, los in los_by_disposition.items() if los}
            if "admission" in avg_los_by_disp:
                admission_los = avg_los_by_disp["admission"]
                if admission_los > current_los * 1.5:
                    insights.append(Insight(
                        insight_type="admission_bottleneck",
                        title="Admission Process Bottleneck",
                        description=f"Admitted patients have {admission_los:.0f} min LOS ({admission_los/current_los:.1f}x average) - admission process is the bottleneck",
                        evidence={"admission_los": admission_los, "avg_los": current_los},
                        impact_score=0.85,
                        confidence=0.9,
                        actionable=True,
                        recommendation="Streamline admission process or create holding area for admitted patients",
                        unmet_need="Admission process optimization system"
                    ))
                    unmet_needs.append("Admission process optimization system")
        
        # Insight 2: LOS by time of day
        los_by_hour = defaultdict(list)
        for arrival in arrivals:
            hour = arrival.get("timestamp").hour
            patient_id = arrival.get("patient_id")
            if not patient_id:
                continue
            
            discharge = next(
                (e for e in events
                 if e.get("patient_id") == patient_id
                 and e.get("event_type") in ["discharge", "lwbs"]),
                None
            )
            
            if discharge:
                los = (discharge.get("timestamp") - arrival.get("timestamp")).total_seconds() / 60
                los_by_hour[hour].append(los)
        
        if los_by_hour:
            avg_los_by_hour = {h: np.mean(los) for h, los in los_by_hour.items() if los}
            if avg_los_by_hour:
                peak_hour = max(avg_los_by_hour.items(), key=lambda x: x[1])
                if peak_hour[1] > current_los * 1.3:
                    insights.append(Insight(
                        insight_type="temporal_los_spike",
                        title="Time-of-Day LOS Spike",
                        description=f"Patients arriving at {peak_hour[0]}:00 have {peak_hour[1]:.0f} min LOS ({peak_hour[1]/current_los:.1f}x average) - system overloads during this hour",
                        evidence={"peak_hour": peak_hour[0], "peak_los": peak_hour[1], "avg_los": current_los},
                        impact_score=0.8,
                        confidence=0.85,
                        actionable=True,
                        recommendation=f"Implement surge capacity protocols for {peak_hour[0]}:00 hour arrivals",
                        unmet_need="Time-based surge capacity management"
                    ))
                    unmet_needs.append("Time-based surge capacity management")
        
        # Enhanced root causes
        root_causes = []
        if p95_los > current_los * 2:
            root_causes.append(f"P95 LOS ({p95_los:.0f} min) is {p95_los/current_los:.1f}x median - long-tail problem affecting 5% of patients")
        if "admission" in avg_los_by_disp:
            admission_los = avg_los_by_disp["admission"]
            if admission_los > current_los * 1.5:
                root_causes.append(f"Admission process bottleneck: Admitted patients have {admission_los:.0f} min LOS ({admission_los/current_los:.1f}x average)")
        if avg_los_by_hour:
            peak_hour = max(avg_los_by_hour.items(), key=lambda x: x[1])
            if peak_hour[1] > current_los * 1.3:
                root_causes.append(f"Time-based overload: {peak_hour[0]}:00 arrivals experience {peak_hour[1]:.0f} min LOS ({peak_hour[1]/current_los:.1f}x average)")
        
        # Enhanced patterns
        enhanced_patterns = {
            "by_hour": avg_los_by_hour,
            "by_disposition": avg_los_by_disp,
            "distribution": {
                "median": float(current_los),
                "p75": float(np.percentile(los_values, 75)),
                "p95": float(p95_los),
                "p99": float(np.percentile(los_values, 99)) if len(los_values) > 1 else float(current_los)
            }
        }
        
        # Enhanced predictive signals
        enhanced_predictive = {}
        if len(los_values) >= 10:
            recent_los = los_values[-min(20, len(los_values)):]
            trend = np.polyfit(range(len(recent_los)), recent_los, 1)[0] if len(recent_los) > 1 else 0.0
            enhanced_predictive = {
                "trend": "increasing" if trend > 1.0 else "decreasing" if trend < -1.0 else "stable",
                "trend_magnitude": float(abs(trend)),
                "forecast_next_hour": float(current_los + trend),
                "risk_level": "high" if trend > 2.0 else "medium" if trend > 1.0 else "low"
            }
        
        return DeepAnalysis(
            metric_name="LOS",
            current_value=current_los,
            benchmark_value=self.benchmarks["los"],
            insights=insights,
            patterns=enhanced_patterns,
            root_causes=root_causes if root_causes else ["Insufficient data for root cause analysis"],
            unmet_needs=list(set(unmet_needs)),
            predictive_signals=enhanced_predictive
        )
    
    async def _analyze_throughput(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int
    ) -> DeepAnalysis:
        """Deep throughput analysis with unique insights."""
        insights = []
        unmet_needs = []
        
        # Calculate throughput
        arrivals = [e for e in events if e.get("event_type") == "arrival"]
        discharges = [e for e in events if e.get("event_type") in ["discharge", "lwbs"]]
        
        total_arrivals = len(arrivals)
        total_discharges = len(discharges)
        
        throughput_rate = total_discharges / window_hours if window_hours > 0 else 0
        
        # Insight 1: Throughput efficiency
        if total_arrivals > 0:
            efficiency = total_discharges / total_arrivals
            if efficiency < 0.9:
                insights.append(Insight(
                    insight_type="throughput_deficit",
                    title="Patient Flow Deficit",
                    description=f"Only {efficiency:.0%} of arrivals are discharged - {total_arrivals - total_discharges} patients accumulating in system",
                    evidence={"arrivals": total_arrivals, "discharges": total_discharges, "efficiency": efficiency},
                    impact_score=0.9,
                    confidence=0.95,
                    actionable=True,
                    recommendation=f"Address backlog of {total_arrivals - total_discharges} patients - system is not keeping up",
                    unmet_need="Real-time backlog management system"
                ))
                unmet_needs.append("Real-time backlog management system")
        
        # Insight 2: Throughput by hour
        arrivals_by_hour = defaultdict(int)
        discharges_by_hour = defaultdict(int)
        
        for event in events:
            hour = event.get("timestamp").hour
            if event.get("event_type") == "arrival":
                arrivals_by_hour[hour] += 1
            elif event.get("event_type") in ["discharge", "lwbs"]:
                discharges_by_hour[hour] += 1
        
        if arrivals_by_hour and discharges_by_hour:
            # Find hours where arrivals > discharges (accumulation)
            accumulation_hours = []
            for hour in range(24):
                arrivals = arrivals_by_hour.get(hour, 0)
                discharges = discharges_by_hour.get(hour, 0)
                if arrivals > discharges * 1.2:  # 20% more arrivals than discharges
                    accumulation_hours.append((hour, arrivals - discharges))
            
            if accumulation_hours:
                worst_hour = max(accumulation_hours, key=lambda x: x[1])
                insights.append(Insight(
                    insight_type="hourly_accumulation",
                    title="Hourly Patient Accumulation",
                    description=f"{worst_hour[0]}:00 hour has {worst_hour[1]} more arrivals than discharges - patients accumulating during this hour",
                    evidence={"hour": worst_hour[0], "accumulation": worst_hour[1]},
                    impact_score=0.85,
                    confidence=0.9,
                    actionable=True,
                    recommendation=f"Increase discharge capacity during {worst_hour[0]}:00 hour or shift arrivals",
                    unmet_need="Hourly capacity balancing system"
                ))
                unmet_needs.append("Hourly capacity balancing system")
        
        return DeepAnalysis(
            metric_name="Throughput",
            current_value=throughput_rate,
            benchmark_value=self.benchmarks["throughput"],
            insights=insights,
            patterns={"arrivals_by_hour": dict(arrivals_by_hour), "discharges_by_hour": dict(discharges_by_hour)},
            root_causes=[],
            unmet_needs=list(set(unmet_needs)),
            predictive_signals={}
        )
    
    async def _analyze_bed_utilization(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int
    ) -> DeepAnalysis:
        """Deep bed utilization analysis."""
        insights = []
        unmet_needs = []
        
        # Calculate bed utilization from events
        bed_assignments = [e for e in events if e.get("event_type") == "bed_assign"]
        bed_releases = [e for e in events if e.get("event_type") == "discharge"]
        
        # Estimate utilization
        if kpis:
            latest_kpi = kpis[-1] if isinstance(kpis[-1], dict) else kpis[-1].dict() if hasattr(kpis[-1], 'dict') else {}
            current_util = latest_kpi.get("bed_utilization", 0)
        else:
            current_util = len(bed_assignments) / max(1, len(bed_releases)) if bed_releases else 0
        
        # Insight: Over-utilization
        if current_util > 0.9:
            insights.append(Insight(
                insight_type="over_utilization",
                title="Bed Over-Utilization",
                description=f"Bed utilization at {current_util:.0%} - system operating at capacity with no buffer",
                evidence={"utilization": current_util},
                impact_score=0.9,
                confidence=0.95,
                actionable=True,
                recommendation="Add bed capacity or improve turnover time - system has no surge capacity",
                unmet_need="Surge capacity management system"
            ))
            unmet_needs.append("Surge capacity management system")
        
        return DeepAnalysis(
            metric_name="Bed Utilization",
            current_value=current_util,
            benchmark_value=self.benchmarks["bed_utilization"],
            insights=insights,
            patterns={},
            root_causes=[],
            unmet_needs=list(set(unmet_needs)),
            predictive_signals={}
        )
    
    async def _analyze_generic(
        self,
        metric_name: str,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int
    ) -> DeepAnalysis:
        """Generic analysis for unknown metrics - generates insights from available data."""
        insights = []
        unmet_needs = []
        root_causes = []
        patterns = {}
        predictive_signals = {}
        
        if not events or len(events) == 0:
            return DeepAnalysis(
                metric_name=metric_name or "ED Operations",
                current_value=0,
                benchmark_value=0,
                insights=[Insight(
                    insight_type="data_availability",
                    title="No Data Available",
                    description="No event data found for analysis. Please upload data to generate insights.",
                    evidence={"event_count": 0},
                    impact_score=0.0,
                    confidence=1.0,
                    actionable=True,
                    recommendation="Upload CSV data or generate sample data to begin analysis",
                    unmet_need="Data ingestion system"
                )],
                patterns={},
                root_causes=["No data available for analysis"],
                unmet_needs=["Data ingestion"],
                predictive_signals={}
            )
        
        # Analyze overall ED performance from events
        arrivals = [e for e in events if e.get("event_type") == "arrival"]
        discharges = [e for e in events if e.get("event_type") == "discharge"]
        lwbs_events = [e for e in events if e.get("event_type") == "lwbs"]
        
        total_arrivals = len(arrivals)
        total_discharges = len(discharges)
        total_lwbs = len(lwbs_events)
        
        if total_arrivals > 0:
            lwbs_rate = total_lwbs / total_arrivals
            discharge_rate = total_discharges / total_arrivals
            
            # Insight 1: Overall performance
            if lwbs_rate > 0.02:  # > 2%
                insights.append(Insight(
                    insight_type="elevated_lwbs",
                    title="Elevated LWBS Rate",
                    description=f"LWBS rate is {lwbs_rate*100:.1f}% ({total_lwbs} patients) - above 2025 target of 1.5%",
                    evidence={"lwbs_rate": lwbs_rate, "lwbs_count": total_lwbs, "total_arrivals": total_arrivals},
                    impact_score=min(lwbs_rate / 0.05, 1.0),  # Normalize to 5% max
                    confidence=0.95,
                    actionable=True,
                    recommendation="Investigate wait times at triage and doctor stages - implement real-time alerts for patients waiting >45 min",
                    unmet_need="Proactive patient retention system"
                ))
                unmet_needs.append("Proactive patient retention system")
                root_causes.append(f"LWBS rate elevated at {lwbs_rate*100:.1f}% - {total_lwbs} patients left without being seen")
            
            # Insight 2: Discharge efficiency
            if discharge_rate < 0.85:
                insights.append(Insight(
                    insight_type="discharge_efficiency",
                    title="Discharge Efficiency Below Target",
                    description=f"Only {discharge_rate:.0%} of arrivals are discharged ({total_discharges}/{total_arrivals}) - patients accumulating in system",
                    evidence={"discharge_rate": discharge_rate, "discharges": total_discharges, "arrivals": total_arrivals},
                    impact_score=1.0 - discharge_rate,
                    confidence=0.9,
                    actionable=True,
                    recommendation="Review discharge processes and bed turnover - consider fast-track for low-acuity patients",
                    unmet_need="Discharge process optimization"
                ))
                unmet_needs.append("Discharge process optimization")
                root_causes.append(f"Discharge efficiency at {discharge_rate:.0%} - {total_arrivals - total_discharges} patients still in system")
            
            # Calculate basic metrics from events
            dtds = []
            los_values = []
            
            for arrival in arrivals[:100]:  # Limit to first 100 for performance
                patient_id = arrival.get("patient_id")
                if not patient_id:
                    continue
                
                arrival_time = arrival.get("timestamp")
                if not arrival_time:
                    continue
                
                # Find doctor visit
                doctor_visit = next(
                    (e for e in events
                     if e.get("patient_id") == patient_id
                     and e.get("event_type") == "doctor_visit"),
                    None
                )
                
                if doctor_visit:
                    dtd = (doctor_visit.get("timestamp") - arrival_time).total_seconds() / 60
                    if dtd > 0 and dtd < 300:  # Reasonable range
                        dtds.append(dtd)
                
                # Find discharge
                discharge = next(
                    (e for e in events
                     if e.get("patient_id") == patient_id
                     and e.get("event_type") in ["discharge", "lwbs"]),
                    None
                )
                
                if discharge:
                    los = (discharge.get("timestamp") - arrival_time).total_seconds() / 60
                    if los > 0 and los < 1000:  # Reasonable range
                        los_values.append(los)
            
            # Calculate current values
            current_dtd = np.median(dtds) if dtds else 0
            current_los = np.median(los_values) if los_values else 0
            
            # Insight 3: DTD performance
            if current_dtd > 0:
                if current_dtd > self.benchmarks["dtd"]:
                    insights.append(Insight(
                        insight_type="dtd_above_target",
                        title="Door-to-Doctor Above Target",
                        description=f"Median DTD is {current_dtd:.0f} min (target: {self.benchmarks['dtd']:.0f} min) - {current_dtd - self.benchmarks['dtd']:.0f} min above target",
                        evidence={"current_dtd": current_dtd, "benchmark": self.benchmarks["dtd"], "sample_size": len(dtds)},
                        impact_score=min((current_dtd - self.benchmarks["dtd"]) / 30, 1.0),
                        confidence=0.9 if len(dtds) >= 10 else 0.7,
                        actionable=True,
                        recommendation=f"Focus on reducing triage and doctor wait times - target {self.benchmarks['dtd']:.0f} min median",
                        unmet_need="Real-time wait time monitoring"
                    ))
                    root_causes.append(f"DTD at {current_dtd:.0f} min exceeds {self.benchmarks['dtd']:.0f} min target")
            
            # Insight 4: LOS performance
            if current_los > 0:
                if current_los > self.benchmarks["los"]:
                    insights.append(Insight(
                        insight_type="los_above_target",
                        title="Length of Stay Above Target",
                        description=f"Median LOS is {current_los:.0f} min (target: {self.benchmarks['los']:.0f} min) - {current_los - self.benchmarks['los']:.0f} min above target",
                        evidence={"current_los": current_los, "benchmark": self.benchmarks["los"], "sample_size": len(los_values)},
                        impact_score=min((current_los - self.benchmarks["los"]) / 120, 1.0),
                        confidence=0.9 if len(los_values) >= 10 else 0.7,
                        actionable=True,
                        recommendation="Review patient flow bottlenecks - focus on labs, imaging, and bed assignment stages",
                        unmet_need="End-to-end patient flow optimization"
                    ))
                    root_causes.append(f"LOS at {current_los:.0f} min exceeds {self.benchmarks['los']:.0f} min target")
            
            # Patterns
            if dtds:
                patterns["dtd_distribution"] = {
                    "median": float(np.median(dtds)),
                    "p75": float(np.percentile(dtds, 75)) if len(dtds) > 1 else float(np.median(dtds)),
                    "p95": float(np.percentile(dtds, 95)) if len(dtds) > 1 else float(np.median(dtds)),
                    "sample_size": len(dtds)
                }
            
            if los_values:
                patterns["los_distribution"] = {
                    "median": float(np.median(los_values)),
                    "p75": float(np.percentile(los_values, 75)) if len(los_values) > 1 else float(np.median(los_values)),
                    "p95": float(np.percentile(los_values, 95)) if len(los_values) > 1 else float(np.median(los_values)),
                    "sample_size": len(los_values)
                }
            
            # Temporal patterns
            arrivals_by_hour = defaultdict(int)
            for arrival in arrivals:
                hour = arrival.get("timestamp").hour if arrival.get("timestamp") else None
                if hour is not None:
                    arrivals_by_hour[hour] += 1
            
            if arrivals_by_hour:
                peak_hour = max(arrivals_by_hour.items(), key=lambda x: x[1])
                patterns["arrival_patterns"] = {
                    "peak_hour": int(peak_hour[0]),
                    "peak_arrivals": int(peak_hour[1]),
                    "total_arrivals": total_arrivals
                }
            
            # Predictive signals
            if len(dtds) >= 10:
                recent_dtds = dtds[-min(20, len(dtds)):]
                trend = np.polyfit(range(len(recent_dtds)), recent_dtds, 1)[0] if len(recent_dtds) > 1 else 0.0
                predictive_signals["dtd_trend"] = {
                    "trend": "increasing" if trend > 0.5 else "decreasing" if trend < -0.5 else "stable",
                    "trend_magnitude": float(abs(trend)),
                    "forecast": float(np.median(dtds) + trend)
                }
        
        # If no insights generated, provide data summary
        if not insights:
            insights.append(Insight(
                insight_type="data_summary",
                title="Data Available for Analysis",
                description=f"Analyzed {total_arrivals} arrivals, {total_discharges} discharges, {total_lwbs} LWBS events over {window_hours} hours",
                evidence={"arrivals": total_arrivals, "discharges": total_discharges, "lwbs": total_lwbs, "window_hours": window_hours},
                impact_score=0.5,
                confidence=1.0,
                actionable=True,
                recommendation="Ask specific questions like 'What are my bottlenecks?' or 'Analyze LWBS' for detailed insights",
                unmet_need="Specific metric analysis"
            ))
        
        # Calculate overall current value (average of available metrics)
        current_value = 0.0
        if current_dtd > 0 and current_los > 0:
            current_value = (current_dtd + current_los) / 2
        elif current_dtd > 0:
            current_value = current_dtd
        elif current_los > 0:
            current_value = current_los
        elif lwbs_rate > 0:
            current_value = lwbs_rate
        
        return DeepAnalysis(
            metric_name=metric_name or "ED Operations",
            current_value=current_value,
            benchmark_value=self.benchmarks.get("dtd", 30.0),
            insights=insights,
            patterns=patterns,
            root_causes=root_causes if root_causes else ["Analysis completed - see insights for details"],
            unmet_needs=list(set(unmet_needs)) if unmet_needs else [],
            predictive_signals=predictive_signals
        )

