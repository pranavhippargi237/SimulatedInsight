"""
Conversational Advisor: Generates detailed action plans from "what should I do" queries.
Combines bottleneck detection, optimization, and ROI analysis into actionable recommendations.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.data.schemas import Bottleneck, OptimizationSuggestion
from app.core.detection import BottleneckDetector
from app.core.optimization import Optimizer

# ROI calculator (optional)
try:
    from app.core.roi_calculator import ROICalculator
    ROI_CALCULATOR_AVAILABLE = True
except ImportError:
    ROI_CALCULATOR_AVAILABLE = False
    ROICalculator = None

# Advanced optimization (optional)
try:
    from app.core.advanced_optimization import AdvancedOptimizer
    ADVANCED_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ADVANCED_OPTIMIZATION_AVAILABLE = False
    AdvancedOptimizer = None

logger = logging.getLogger(__name__)


class EDAdvisor:
    """
    Conversational advisor that generates detailed action plans.
    
    Analyzes current state, detects bottlenecks, optimizes resources,
    and provides prioritized recommendations with ROI analysis.
    """
    
    def __init__(self):
        self.detector = BottleneckDetector()
        self.optimizer = Optimizer()
        self.advanced_optimizer = AdvancedOptimizer() if ADVANCED_OPTIMIZATION_AVAILABLE else None
        self.roi_calculator = ROICalculator() if ROICalculator else None
    
    async def generate_action_plan(
        self,
        window_hours: int = 48,
        top_n: int = 5,
        include_roi: bool = True,
        equity_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive action plan based on current ED state.
        
        Args:
            window_hours: Time window for analysis (default 48h)
            top_n: Number of top recommendations (default 5)
            include_roi: Include ROI calculations (default True)
            equity_mode: Enable equity-aware recommendations (default True)
            
        Returns:
            Detailed action plan with priorities, ROI, and implementation steps
        """
        start_time = datetime.utcnow()
        
        # Step 1: Detect current bottlenecks
        logger.info("Step 1: Detecting bottlenecks...")
        bottlenecks = await self.detector.detect_bottlenecks(window_hours=window_hours, top_n=10)
        
        if not bottlenecks:
            return {
                "status": "no_bottlenecks",
                "message": "No significant bottlenecks detected. Your ED is operating efficiently!",
                "recommendations": [
                    {
                        "priority": 1,
                        "action": "Monitor",
                        "description": "Continue monitoring key metrics (DTD, LOS, LWBS) to maintain current performance.",
                        "expected_impact": "Maintain current performance levels",
                        "cost": 0.0,
                        "timeline": "Ongoing"
                    }
                ],
                "summary": "No immediate action required. System is operating within normal parameters."
            }
        
        # Step 2: Generate optimization suggestions
        logger.info(f"Step 2: Generating optimization suggestions for {len(bottlenecks)} bottlenecks...")
        
        # Use advanced optimization if available
        if self.advanced_optimizer:
            from app.data.schemas import OptimizationRequest
            opt_request = OptimizationRequest(
                constraints={"budget": 5000.0, "staff_max": 20},
                target_metrics=["dtd", "lwbs", "los"]
            )
            suggestions, forecast, metadata = await self.advanced_optimizer.optimize_advanced(
                request=opt_request,
                bottlenecks=bottlenecks,
                historical_sims=None,
                equity_mode=equity_mode,
                forecast_horizon=72
            )
        else:
            from app.data.schemas import OptimizationRequest
            opt_request = OptimizationRequest(
                constraints={"budget": 5000.0, "staff_max": 20},
                target_metrics=["dtd", "lwbs", "los"]
            )
            suggestions = await self.optimizer.optimize(
                request=opt_request,
                bottlenecks=bottlenecks,
                historical_sims=None
            )
            forecast = None
            metadata = {}
        
        # Step 3: Calculate ROI for top suggestions
        roi_analyses = {}
        if include_roi and self.roi_calculator:
            logger.info("Step 3: Calculating ROI for recommendations...")
            for suggestion in suggestions[:top_n]:
                roi = self.roi_calculator.calculate_roi(suggestion)
                roi_analyses[f"suggestion_{suggestion.priority}"] = {
                    "cost_per_shift": roi.cost_per_shift,
                    "cost_per_year": roi.cost_per_year,
                    "total_annual_savings": roi.total_annual_savings,
                    "roi_percentage": roi.roi_percentage,
                    "payback_period_days": roi.payback_period_days,
                    "net_present_value": roi.net_present_value,
                    "confidence": roi.confidence
                }
        
        # Step 4: Build detailed action plan
        logger.info("Step 4: Building detailed action plan...")
        
        action_plan = self._build_action_plan(
            bottlenecks=bottlenecks,
            suggestions=suggestions[:top_n],
            roi_analyses=roi_analyses,
            forecast=forecast,
            equity_mode=equity_mode
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        action_plan["execution_time_seconds"] = execution_time
        action_plan["analysis_window_hours"] = window_hours
        
        return action_plan
    
    def _build_action_plan(
        self,
        bottlenecks: List[Bottleneck],
        suggestions: List[OptimizationSuggestion],
        roi_analyses: Dict[str, Any],
        forecast: Optional[Any],
        equity_mode: bool
    ) -> Dict[str, Any]:
        """Build a detailed, human-readable action plan."""
        
        # Categorize bottlenecks by severity
        critical_bottlenecks = [b for b in bottlenecks if b.severity == "critical"]
        high_bottlenecks = [b for b in bottlenecks if b.severity == "high"]
        medium_bottlenecks = [b for b in bottlenecks if b.severity == "medium"]
        
        # Build recommendations with detailed explanations
        recommendations = []
        
        for i, suggestion in enumerate(suggestions, 1):
            # Find related bottleneck
            related_bottleneck = None
            for bottleneck in bottlenecks:
                if bottleneck.stage == suggestion.resource_type or \
                   bottleneck.bottleneck_name.lower() in suggestion.resource_type.lower():
                    related_bottleneck = bottleneck
                    break
            
            # Get ROI if available
            roi_key = f"suggestion_{suggestion.priority}"
            roi = roi_analyses.get(roi_key, {})
            
            # Build detailed recommendation
            recommendation = {
                "priority": i,
                "action": suggestion.action.title(),
                "resource_type": suggestion.resource_type,
                "quantity": suggestion.quantity,
                "description": self._build_description(suggestion, related_bottleneck),
                "expected_impact": {
                    "dtd_reduction": f"{abs(suggestion.expected_impact.get('dtd_reduction', 0)):.1f}%",
                    "los_reduction": f"{abs(suggestion.expected_impact.get('los_reduction', 0)):.1f}%",
                    "lwbs_drop": f"{abs(suggestion.expected_impact.get('lwbs_drop', 0)):.1f}%"
                },
                "cost": {
                    "per_shift": roi.get("cost_per_shift", suggestion.cost),
                    "per_year": roi.get("cost_per_year", suggestion.cost * 1095),
                    "roi_percentage": roi.get("roi_percentage", 0),
                    "payback_days": roi.get("payback_period_days", 0),
                    "annual_savings": roi.get("total_annual_savings", 0)
                },
                "confidence": suggestion.confidence,
                "implementation_steps": self._build_implementation_steps(suggestion, related_bottleneck),
                "timeline": self._estimate_timeline(suggestion),
                "risks": self._identify_risks(suggestion, related_bottleneck),
                "success_metrics": self._define_success_metrics(suggestion)
            }
            
            recommendations.append(recommendation)
        
        # Build summary
        summary = self._build_summary(
            bottlenecks=bottlenecks,
            suggestions=suggestions,
            forecast=forecast,
            equity_mode=equity_mode
        )
        
        return {
            "status": "ok",
            "summary": summary,
            "current_state": {
                "total_bottlenecks": len(bottlenecks),
                "critical": len(critical_bottlenecks),
                "high": len(high_bottlenecks),
                "medium": len(medium_bottlenecks),
                "bottlenecks": [b.dict() for b in bottlenecks[:5]]
            },
            "recommendations": recommendations,
            "forecast": {
                "predicted_dtd": forecast.predicted_dtd if forecast else None,
                "predicted_lwbs": forecast.predicted_lwbs if forecast else None,
                "forecast_hours": forecast.forecast_hours if forecast else None,
                "confidence": forecast.confidence if forecast else None
            } if forecast else None,
            "next_steps": self._build_next_steps(suggestions)
        }
    
    def _build_description(
        self,
        suggestion: OptimizationSuggestion,
        bottleneck: Optional[Bottleneck]
    ) -> str:
        """Build human-readable description of the recommendation."""
        
        action = suggestion.action
        resource = suggestion.resource_type
        quantity = suggestion.quantity
        
        if bottleneck:
            return (
                f"{action.title()} {quantity} {resource}(s) to address the {bottleneck.bottleneck_name} "
                f"bottleneck. This bottleneck is causing {bottleneck.current_wait_time_minutes:.1f}-minute wait times "
                f"and has a {bottleneck.impact_score:.0%} impact on overall ED flow. "
                f"Expected to reduce DTD by {abs(suggestion.expected_impact.get('dtd_reduction', 0)):.1f}% and "
                f"LWBS by {abs(suggestion.expected_impact.get('lwbs_drop', 0)):.1f}%."
            )
        else:
            return (
                f"{action.title()} {quantity} {resource}(s) to improve ED operations. "
                f"Expected to reduce DTD by {abs(suggestion.expected_impact.get('dtd_reduction', 0)):.1f}% and "
                f"LWBS by {abs(suggestion.expected_impact.get('lwbs_drop', 0)):.1f}%."
            )
    
    def _build_implementation_steps(
        self,
        suggestion: OptimizationSuggestion,
        bottleneck: Optional[Bottleneck]
    ) -> List[str]:
        """Build step-by-step implementation guide."""
        
        steps = []
        
        if suggestion.action == "add":
            steps.append(f"1. Identify available {suggestion.resource_type}(s) for deployment")
            steps.append(f"2. Coordinate with staffing office to schedule {suggestion.quantity} additional {suggestion.resource_type}(s)")
            steps.append(f"3. Notify current staff of resource addition")
            steps.append(f"4. Monitor metrics (DTD, LOS, LWBS) for 24-48 hours after implementation")
            steps.append(f"5. Adjust if needed based on observed impact")
        elif suggestion.action == "remove":
            steps.append(f"1. Assess impact of removing {suggestion.quantity} {suggestion.resource_type}(s)")
            steps.append(f"2. Ensure other resources can handle increased load")
            steps.append(f"3. Coordinate removal with staffing office")
            steps.append(f"4. Monitor metrics closely for 24-48 hours")
            steps.append(f"5. Have backup plan ready if metrics worsen")
        elif suggestion.action == "shift":
            steps.append(f"1. Identify {suggestion.resource_type}(s) available for shift")
            steps.append(f"2. Coordinate shift timing with affected staff")
            steps.append(f"3. Ensure coverage during transition period")
            steps.append(f"4. Monitor metrics during and after shift")
            steps.append(f"5. Evaluate effectiveness and adjust if needed")
        
        return steps
    
    def _estimate_timeline(self, suggestion: OptimizationSuggestion) -> str:
        """Estimate implementation timeline."""
        
        if suggestion.action == "add":
            if suggestion.resource_type == "doctor":
                return "2-4 weeks (credentialing, scheduling)"
            elif suggestion.resource_type == "nurse":
                return "1-2 weeks (scheduling, orientation)"
            elif suggestion.resource_type in ["lab_tech", "imaging_tech", "tech"]:
                return "1-2 weeks (scheduling, training)"
            else:
                return "1 week (scheduling)"
        elif suggestion.action == "remove":
            return "Immediate (coordinate with staffing)"
        elif suggestion.action == "shift":
            return "1-3 days (coordinate with affected staff)"
        else:
            return "1-2 weeks"
    
    def _identify_risks(
        self,
        suggestion: OptimizationSuggestion,
        bottleneck: Optional[Bottleneck]
    ) -> List[str]:
        """Identify potential risks of the recommendation."""
        
        risks = []
        
        if suggestion.action == "add":
            risks.append(f"Cost: ${suggestion.cost:.0f}/shift may exceed budget constraints")
            risks.append("Staff availability: May be difficult to find qualified staff on short notice")
            if suggestion.resource_type == "doctor":
                risks.append("Credentialing: New doctors require credentialing (2-4 weeks)")
        elif suggestion.action == "remove":
            risks.append("Capacity: Removing resources may worsen other bottlenecks")
            risks.append("Staff morale: May impact staff satisfaction")
        elif suggestion.action == "shift":
            risks.append("Coverage gaps: Transition period may have coverage issues")
            risks.append("Staff resistance: Staff may resist schedule changes")
        
        if bottleneck and bottleneck.severity == "critical":
            risks.append("Urgency: Critical bottleneck requires immediate action")
        
        return risks
    
    def _define_success_metrics(self, suggestion: OptimizationSuggestion) -> Dict[str, Any]:
        """Define success metrics for the recommendation."""
        
        dtd_reduction = abs(suggestion.expected_impact.get('dtd_reduction', 0))
        los_reduction = abs(suggestion.expected_impact.get('los_reduction', 0))
        lwbs_drop = abs(suggestion.expected_impact.get('lwbs_drop', 0))
        
        return {
            "primary": {
                "metric": "DTD Reduction",
                "target": f"{dtd_reduction:.1f}%",
                "measurement": "Compare DTD before and after implementation"
            },
            "secondary": [
                {
                    "metric": "LOS Reduction",
                    "target": f"{los_reduction:.1f}%",
                    "measurement": "Track median LOS over 1 week"
                },
                {
                    "metric": "LWBS Drop",
                    "target": f"{lwbs_drop:.1f}%",
                    "measurement": "Compare LWBS rate before and after"
                }
            ],
            "timeline": "Measure after 1 week of implementation",
            "success_criteria": f"DTD reduction ≥{dtd_reduction * 0.8:.1f}% (80% of expected)"
        }
    
    def _build_summary(
        self,
        bottlenecks: List[Bottleneck],
        suggestions: List[OptimizationSuggestion],
        forecast: Optional[Any],
        equity_mode: bool
    ) -> str:
        """Build executive summary of the action plan."""
        
        critical_count = len([b for b in bottlenecks if b.severity == "critical"])
        total_savings = sum(
            s.cost * 1095 * -1  # Negative because we're calculating potential savings
            for s in suggestions
        )
        
        summary_parts = []
        
        if critical_count > 0:
            summary_parts.append(
                f"🚨 URGENT: {critical_count} critical bottleneck(s) detected requiring immediate action."
            )
        
        summary_parts.append(
            f"Found {len(bottlenecks)} bottleneck(s) affecting ED operations. "
            f"Top {len(suggestions)} recommendations provided with detailed implementation plans."
        )
        
        if suggestions:
            top_suggestion = suggestions[0]
            summary_parts.append(
                f"Priority #1: {top_suggestion.action.title()} {top_suggestion.quantity} "
                f"{top_suggestion.resource_type}(s) - Expected {abs(top_suggestion.expected_impact.get('dtd_reduction', 0)):.1f}% DTD reduction."
            )
        
        if forecast:
            summary_parts.append(
                f"Forecast: Predicted DTD of {forecast.predicted_dtd:.1f} min in next {forecast.forecast_hours}h "
                f"if no action taken (confidence: {forecast.confidence:.0%})."
            )
        
        return " ".join(summary_parts)
    
    def _build_next_steps(self, suggestions: List[OptimizationSuggestion]) -> List[str]:
        """Build next steps for the director."""
        
        next_steps = [
            "1. Review the prioritized recommendations above",
            "2. Assess budget and staffing availability for top 3 recommendations",
            "3. Coordinate with staffing office to implement Priority #1 recommendation",
            "4. Set up monitoring dashboard to track metrics after implementation",
            "5. Schedule follow-up review in 1 week to assess impact"
        ]
        
        if suggestions:
            top_suggestion = suggestions[0]
            next_steps.insert(0, f"🚨 IMMEDIATE: Implement Priority #1 - {top_suggestion.action.title()} {top_suggestion.quantity} {top_suggestion.resource_type}(s)")
        
        return next_steps

