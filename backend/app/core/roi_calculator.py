"""
ROI Calculator for ED Bottleneck Engine.
Provides board-ready ROI calculations for optimization suggestions.

Per 2025 research:
- $5k LWBS aversion per case
- $200-500/visit savings from efficiency gains
- $500k-$2M/year savings per 50k-visit ED
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.data.schemas import OptimizationSuggestion, SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class ROICalculation:
    """ROI calculation result."""
    suggestion: OptimizationSuggestion
    cost_per_shift: float
    cost_per_year: float
    
    # Savings
    lwbs_aversion_savings: float  # $5k per LWBS case averted
    dtd_reduction_savings: float   # $50 per minute reduction (readmit prevention)
    los_reduction_savings: float   # $100 per hour reduction (throughput)
    total_annual_savings: float
    
    # ROI metrics
    roi_percentage: float
    payback_period_days: float
    net_present_value: float  # 3-year NPV
    
    # Confidence
    confidence: float
    assumptions: List[str]


class ROICalculator:
    """
    Calculate ROI for optimization suggestions.
    
    Based on 2025 benchmarks:
    - LWBS aversion: $5,000 per case
    - DTD reduction: $50 per minute (readmit prevention)
    - LOS reduction: $100 per hour (throughput improvement)
    - Annual ED visits: 50,000 (medium ED)
    """
    
    # Cost assumptions (per shift, 2025 AHA benchmarks)
    RESOURCE_COSTS = {
        "nurse": 50.0,      # $50/shift
        "doctor": 200.0,    # $200/shift
        "tech": 75.0,       # $75/shift
        "bed": 100.0        # $100/day
    }
    
    # Savings assumptions (2025 research)
    LWBS_AVERSION_VALUE = 5000.0  # $5k per LWBS case averted
    DTD_REDUCTION_VALUE = 50.0    # $50 per minute reduction
    LOS_REDUCTION_VALUE = 100.0   # $100 per hour reduction
    
    # Annual assumptions
    SHIFTS_PER_YEAR = 365 * 3  # 3 shifts/day
    ANNUAL_VISITS = 50000       # Medium ED
    
    def __init__(self):
        self.assumptions_log = []
    
    def calculate_roi(
        self,
        suggestion: OptimizationSuggestion,
        simulation_result: Optional[SimulationResult] = None,
        annual_visits: int = None
    ) -> ROICalculation:
        """
        Calculate ROI for an optimization suggestion.
        
        Args:
            suggestion: Optimization suggestion
            simulation_result: Optional simulation results for accurate deltas
            annual_visits: Annual ED visit volume (default: 50,000)
            
        Returns:
            ROI calculation with savings and payback period
        """
        annual_visits = annual_visits or self.ANNUAL_VISITS
        
        # Cost calculation
        cost_per_shift = suggestion.cost
        shifts_per_year = self.SHIFTS_PER_YEAR
        if suggestion.resource_type == "bed":
            shifts_per_year = 365  # Beds are per-day, not per-shift
        
        cost_per_year = cost_per_shift * shifts_per_year
        
        # Get impact from simulation or suggestion
        if simulation_result and simulation_result.deltas:
            dtd_reduction_pct = abs(simulation_result.deltas.get("dtd_reduction", 0))
            los_reduction_pct = abs(simulation_result.deltas.get("los_reduction", 0))
            lwbs_drop_pct = abs(simulation_result.deltas.get("lwbs_drop", 0))
            confidence = simulation_result.confidence
        else:
            # Use expected impact from suggestion
            dtd_reduction_pct = abs(suggestion.expected_impact.get("dtd_reduction", 0)) / 100.0
            los_reduction_pct = abs(suggestion.expected_impact.get("los_reduction", 0)) / 100.0
            lwbs_drop_pct = abs(suggestion.expected_impact.get("lwbs_drop", 0)) / 100.0
            confidence = suggestion.confidence
        
        # Baseline assumptions (realistic ED metrics)
        baseline_dtd = 35.0  # minutes
        baseline_los = 270.0  # minutes (4.5 hours)
        baseline_lwbs_rate = 0.015  # 1.5%
        
        # Calculate absolute reductions
        dtd_reduction_min = baseline_dtd * dtd_reduction_pct
        los_reduction_min = baseline_los * los_reduction_pct
        lwbs_reduction_rate = baseline_lwbs_rate * lwbs_drop_pct
        
        # Savings calculations
        # 1. LWBS aversion: $5k per case averted
        lwbs_cases_averted = annual_visits * lwbs_reduction_rate
        lwbs_aversion_savings = lwbs_cases_averted * self.LWBS_AVERSION_VALUE
        
        # 2. DTD reduction: $50 per minute (readmit prevention)
        dtd_reduction_savings = dtd_reduction_min * annual_visits * (self.DTD_REDUCTION_VALUE / 60.0)
        
        # 3. LOS reduction: $100 per hour (throughput improvement)
        los_reduction_savings = (los_reduction_min / 60.0) * annual_visits * (self.LOS_REDUCTION_VALUE / 60.0)
        
        # Total annual savings
        total_annual_savings = (
            lwbs_aversion_savings +
            dtd_reduction_savings +
            los_reduction_savings
        )
        
        # ROI metrics
        roi_percentage = ((total_annual_savings - cost_per_year) / cost_per_year) * 100 if cost_per_year > 0 else 0
        payback_period_days = (cost_per_year / (total_annual_savings / 365)) if total_annual_savings > 0 else float('inf')
        
        # 3-year NPV (10% discount rate)
        discount_rate = 0.10
        npv = sum(
            (total_annual_savings - cost_per_year) / ((1 + discount_rate) ** year)
            for year in range(1, 4)
        )
        
        # Assumptions
        assumptions = [
            f"Annual visits: {annual_visits:,}",
            f"Baseline DTD: {baseline_dtd:.1f} min",
            f"Baseline LOS: {baseline_los:.1f} min ({baseline_los/60:.1f}h)",
            f"Baseline LWBS: {baseline_lwbs_rate:.1%}",
            f"LWBS aversion value: ${self.LWBS_AVERSION_VALUE:,.0f}/case",
            f"DTD reduction value: ${self.DTD_REDUCTION_VALUE}/min",
            f"LOS reduction value: ${self.LOS_REDUCTION_VALUE}/hour"
        ]
        
        return ROICalculation(
            suggestion=suggestion,
            cost_per_shift=cost_per_shift,
            cost_per_year=cost_per_year,
            lwbs_aversion_savings=lwbs_aversion_savings,
            dtd_reduction_savings=dtd_reduction_savings,
            los_reduction_savings=los_reduction_savings,
            total_annual_savings=total_annual_savings,
            roi_percentage=roi_percentage,
            payback_period_days=payback_period_days,
            net_present_value=npv,
            confidence=confidence,
            assumptions=assumptions
        )
    
    def format_roi_report(self, roi: ROICalculation) -> str:
        """
        Format ROI calculation as board-ready report.
        """
        report = f"""
OPTIMIZATION ROI ANALYSIS
=========================

Action: {roi.suggestion.action.title()} {roi.suggestion.quantity} {roi.suggestion.resource_type}(s)
Cost: ${roi.cost_per_shift:,.2f}/shift | ${roi.cost_per_year:,.2f}/year

EXPECTED IMPACT
---------------
DTD Reduction: {abs(roi.suggestion.expected_impact.get('dtd_reduction', 0)):.1f}%
LOS Reduction: {abs(roi.suggestion.expected_impact.get('los_reduction', 0)):.1f}%
LWBS Drop: {abs(roi.suggestion.expected_impact.get('lwbs_drop', 0)):.1f}%

ANNUAL SAVINGS
--------------
LWBS Aversion: ${roi.lwbs_aversion_savings:,.2f}
DTD Reduction: ${roi.dtd_reduction_savings:,.2f}
LOS Reduction: ${roi.los_reduction_savings:,.2f}
─────────────────────────────
Total Savings: ${roi.total_annual_savings:,.2f}

ROI METRICS
-----------
ROI: {roi.roi_percentage:.1f}%
Payback Period: {roi.payback_period_days:.0f} days
3-Year NPV: ${roi.net_present_value:,.2f}
Confidence: {roi.confidence:.1%}

ASSUMPTIONS
-----------
{chr(10).join(f'  • {a}' for a in roi.assumptions)}
"""
        return report.strip()
    
    def compare_scenarios(
        self,
        suggestions: List[OptimizationSuggestion],
        simulation_results: Optional[List[SimulationResult]] = None
    ) -> List[ROICalculation]:
        """Compare ROI across multiple scenarios."""
        rois = []
        for i, suggestion in enumerate(suggestions):
            sim_result = simulation_results[i] if simulation_results and i < len(simulation_results) else None
            roi = self.calculate_roi(suggestion, sim_result)
            rois.append(roi)
        
        # Sort by total annual savings
        rois.sort(key=lambda x: x.total_annual_savings, reverse=True)
        
        return rois


