"""
Optimization endpoints.
Enhanced with RL-driven optimization, equity-aware constraints, and predictive forecasting.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from app.data.schemas import OptimizationRequest, OptimizationSuggestion
from app.core.optimization import Optimizer
from app.core.detection import BottleneckDetector

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

router = APIRouter()
optimizer = Optimizer()
detector = BottleneckDetector()
advanced_optimizer = AdvancedOptimizer() if ADVANCED_OPTIMIZATION_AVAILABLE else None
roi_calculator = ROICalculator() if ROI_CALCULATOR_AVAILABLE else None


@router.post("/optimize")
async def optimize_operations(request: OptimizationRequest):
    """
    Generate optimization suggestions based on bottlenecks and constraints.
    
    Args:
        request: Optimization request with bottlenecks and constraints
        
    Returns:
        Ranked list of optimization suggestions
    """
    try:
        # Detect bottlenecks (always detect, no longer part of request)
        bottlenecks = await detector.detect_bottlenecks(top_n=10)
        
        # Run optimization
        suggestions = await optimizer.optimize(
            request=request,
            bottlenecks=bottlenecks,
            historical_sims=None
        )
        
        return {
            "status": "ok",
            "suggestions": [s.dict() for s in suggestions],
            "count": len(suggestions)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/optimize/advanced")
async def optimize_advanced(
    request: OptimizationRequest,
    equity_mode: bool = True,
    forecast_horizon: int = 72
):
    """
    Advanced RL-driven optimization with equity-aware constraints and predictive forecasting.
    
    Per 2025 research:
    - 31% faster decisions vs static LP
    - 25% better throughput
    - 10% equity improvements
    - 72h predictive horizon (MAE <5 min)
    
    Args:
        request: Optimization request
        equity_mode: Enable equity-aware optimization (SDOH integration)
        forecast_horizon: Hours to forecast ahead (default 72h)
        historical_sims: Historical simulation results for RL learning
    """
    if not advanced_optimizer:
        raise HTTPException(
            status_code=501,
            detail="Advanced optimization not available."
        )
    
    try:
        # Detect bottlenecks (always detect, no longer part of request)
        bottlenecks = await detector.detect_bottlenecks(top_n=10)
        
        # Get historical sims from request if available
        historical_sims = getattr(request, 'historical_sims', None)
        
        # Run advanced optimization
        suggestions, forecast, metadata = await advanced_optimizer.optimize_advanced(
            request=request,
            bottlenecks=bottlenecks,
            historical_sims=historical_sims,
            equity_mode=equity_mode,
            forecast_horizon=forecast_horizon
        )
        
        # Calculate equity metrics
        equity_metrics = None
        if equity_mode:
            equity_metrics = advanced_optimizer.calculate_equity_metrics(suggestions)
        
        # Generate explanations for top suggestions
        explanations = {}
        for suggestion in suggestions[:3]:  # Top 3
            explanations[f"suggestion_{suggestion.priority}"] = advanced_optimizer.explain_suggestion(
                suggestion, bottlenecks
            )
        
        # Calculate ROI for top suggestions
        roi_analyses = {}
        if roi_calculator:
            for suggestion in suggestions[:3]:  # Top 3
                roi = roi_calculator.calculate_roi(suggestion)
                roi_analyses[f"suggestion_{suggestion.priority}"] = {
                    "cost_per_shift": roi.cost_per_shift,
                    "cost_per_year": roi.cost_per_year,
                    "total_annual_savings": roi.total_annual_savings,
                    "roi_percentage": roi.roi_percentage,
                    "payback_period_days": roi.payback_period_days,
                    "net_present_value": roi.net_present_value,
                    "confidence": roi.confidence,
                    "breakdown": {
                        "lwbs_aversion_savings": roi.lwbs_aversion_savings,
                        "dtd_reduction_savings": roi.dtd_reduction_savings,
                        "los_reduction_savings": roi.los_reduction_savings
                    },
                    "report": roi_calculator.format_roi_report(roi)
                }
        
        response = {
            "status": "ok",
            "suggestions": [s.dict() for s in suggestions],
            "count": len(suggestions),
            "forecast": {
                "forecast_hours": forecast.forecast_hours,
                "predicted_dtd": forecast.predicted_dtd,
                "predicted_lwbs": forecast.predicted_lwbs,
                "confidence": forecast.confidence,
                "optimal_allocation": forecast.optimal_allocation,
                "expected_impact": forecast.expected_impact
            } if forecast else None,
            "equity_metrics": {
                "lwbs_disparity": equity_metrics.lwbs_disparity,
                "access_score": equity_metrics.access_score,
                "sdh_penalty": equity_metrics.sdh_penalty,
                "rural_penalty": equity_metrics.rural_penalty
            } if equity_metrics else None,
            "explanations": explanations,
            "roi_analyses": roi_analyses,
            "metadata": metadata
        }
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Advanced optimization failed: {str(e)}")

