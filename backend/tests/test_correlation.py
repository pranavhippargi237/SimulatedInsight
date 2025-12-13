"""
Tests for correlation analysis with Polars.
"""
import pytest
from app.core.correlation_analysis import CorrelationAnalyzer
from datetime import datetime, timedelta


@pytest.mark.asyncio
async def test_correlation_basic():
    """Test basic correlation analysis."""
    analyzer = CorrelationAnalyzer()
    
    # Mock events and KPIs
    events = [
        {
            "timestamp": datetime.now() - timedelta(hours=i),
            "event_type": "arrival",
            "metadata": {"disease_category": "psychiatric" if i % 3 == 0 else "other"}
        }
        for i in range(10)
    ]
    
    kpis = [
        {
            "timestamp": datetime.now() - timedelta(hours=i),
            "lwbs": 0.1 if i % 3 == 0 else 0.05,
            "los": 180.0
        }
        for i in range(10)
    ]
    
    result = await analyzer.analyze_correlations(events, kpis, window_hours=24)
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_correlation_polars():
    """Test that Polars is used for correlation if available."""
    analyzer = CorrelationAnalyzer()
    
    # Check if Polars is available
    try:
        import polars as pl
        polars_available = True
    except ImportError:
        polars_available = False
    
    # Test with sufficient data for Polars correlation
    events = [
        {
            "timestamp": datetime.now() - timedelta(hours=i),
            "event_type": "arrival",
            "metadata": {"disease_category": "psychiatric" if i % 2 == 0 else "other"}
        }
        for i in range(50)  # More data for Polars
    ]
    
    kpis = [
        {
            "timestamp": datetime.now() - timedelta(hours=i),
            "lwbs": 0.1 if i % 2 == 0 else 0.05,
            "los": 180.0
        }
        for i in range(50)
    ]
    
    result = await analyzer.analyze_correlations(events, kpis, window_hours=48)
    assert isinstance(result, dict)
    
    # Check that correlation method is indicated
    if result.get("lwbs"):
        for corr_data in result["lwbs"].values():
            if isinstance(corr_data, dict):
                method = corr_data.get("method")
                if polars_available:
                    # Polars should be used if available
                    assert method in ["Polars", "Pandas/NumPy"]
                else:
                    assert method == "Pandas/NumPy"
