"""
LLM-powered narrative generation for causal analysis.

Converts causal graphs, ATE estimates, and probabilities into
clinician-friendly narratives with counterfactuals and equity analysis.
"""
import logging
import os
from typing import Dict, Any
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)

client = None
if settings.OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
    except Exception as e:
        logger.warning(f"OpenAI client initialization failed: {e}")


async def generate_causal_narrative(
    causal_analysis: Dict[str, Any],
    bottleneck: Dict[str, Any]
) -> str:
    """
    Generate human-readable narrative from causal analysis using GPT-4o-mini.
    """
    # Convert to dict if Pydantic model
    if hasattr(bottleneck, 'dict'):
        bottleneck_dict = bottleneck.dict()
    elif hasattr(bottleneck, '__dict__'):
        bottleneck_dict = bottleneck.__dict__
    else:
        bottleneck_dict = bottleneck
    
    if not client:
        return _fallback_narrative(causal_analysis, bottleneck_dict)
    
    try:
        stage = bottleneck_dict.get("stage", "unknown") if isinstance(bottleneck_dict, dict) else getattr(bottleneck, 'stage', 'unknown')
        bottleneck_name = bottleneck_dict.get("bottleneck_name", "Bottleneck") if isinstance(bottleneck_dict, dict) else getattr(bottleneck, 'bottleneck_name', 'Bottleneck')
        
        # Extract key insights
        ate = causal_analysis.get('ate_estimates', {})
        probabilities = causal_analysis.get('probabilistic_insights', {}).get('probabilities', {})
        attributions = causal_analysis.get('feature_attributions', {}).get('attributions', {})
        counterfactuals = causal_analysis.get('counterfactuals', [])
        confounders = causal_analysis.get('confounders', [])
        interactions = causal_analysis.get('interactions', [])
        equity = causal_analysis.get('equity_analysis', {})
        variance_explained = causal_analysis.get('variance_explained', {})
        confidence_scores = causal_analysis.get('confidence_scores', {})
        
        # Extract values for prompt (with safe defaults)
        ate_value = 0
        ci_lower = None
        ci_upper = None
        treatment = 'intervention'
        
        if isinstance(ate, dict):
            try:
                ate_value = float(ate.get('value', 0)) if ate.get('value') is not None else 0
            except (ValueError, TypeError):
                ate_value = 0
            
            try:
                ci_lower = float(ate.get('ci_lower')) if ate.get('ci_lower') is not None else None
            except (ValueError, TypeError):
                ci_lower = None
            
            try:
                ci_upper = float(ate.get('ci_upper')) if ate.get('ci_upper') is not None else None
            except (ValueError, TypeError):
                ci_upper = None
            
            treatment = str(ate.get('treatment', 'intervention'))
        
        # CI formatting with safe defaults
        ci_str = ""
        if ci_lower is not None and ci_upper is not None:
            try:
                ci_str = f", 95% CI [{ci_lower:.1f}, {ci_upper:.1f}]"
            except (ValueError, TypeError):
                ci_str = ""
        
        # Variance explained summary
        variance_summary = {}
        if variance_explained:
            for var, data in list(variance_explained.items())[:2]:
                if isinstance(data, dict):
                    variance_summary[var] = data.get('percentage', 0)
        
        # Confidence scores
        overall_conf = confidence_scores.get('overall_confidence', 0.5) if isinstance(confidence_scores, dict) else 0.5
        bayesian_conf = confidence_scores.get('bayesian_confidence', 0.5) if isinstance(confidence_scores, dict) else 0.5
        
        # Interaction strength
        interaction_strength = 0
        if interactions and len(interactions) > 0:
            interaction_strength = interactions[0].get('strength', 0) if isinstance(interactions[0], dict) else 0
        
        # Equity disparity
        disparity_pct = equity.get('disparity_pct', 0) if isinstance(equity, dict) else 0
        high_acuity_wait = equity.get('high_acuity_wait', 0) if isinstance(equity, dict) else 0
        low_acuity_wait = equity.get('low_acuity_wait', 0) if isinstance(equity, dict) else 0
        concern_level = equity.get('concern', 'unknown') if isinstance(equity, dict) else 'unknown'
        
        # Counterfactual ROI
        roi_pct = 0
        payback_days = 0
        cost = 0
        savings = 0
        if counterfactuals and len(counterfactuals) > 0:
            cf_roi = counterfactuals[0].get('roi', {})
            if isinstance(cf_roi, dict):
                roi_pct = cf_roi.get('roi_percentage', 0)
                payback_days = cf_roi.get('payback_days', 0)
                cost = cf_roi.get('daily_cost', 0)
                savings = cf_roi.get('daily_savings', 0)
        
        # Probability
        prob_val = 0
        if probabilities:
            prob_items = list(probabilities.items())
            if prob_items:
                prob_val = prob_items[0][1] if isinstance(prob_items[0][1], (int, float)) else 0
        
        # Build prompt with enhanced data
        prompt = f"""You are an ED operations expert analyzing a {bottleneck_name} bottleneck at the {stage} stage.

CAUSAL ANALYSIS RESULTS:
- ATE Estimate: {ate} (with 95% CI if available)
- Probabilistic Insights: {probabilities}
- Feature Attributions: {attributions} (SHAP percentages)
- Variance Explained: {variance_explained} (R² decomposition)
- Confidence Scores: {confidence_scores} (overall: {confidence_scores.get('overall_confidence', 0.5):.0%})
- Counterfactuals: {counterfactuals} (with ROI if available)
- Confounders: {confounders}
- Interactions: {interactions} (multivariate effects)
- Equity Analysis: {equity} (disparity percentages)

Generate a structured root cause analysis narrative with:

1. **Immediate Causes** (what's happening now):
   - Use ATE estimates with confidence intervals (e.g., "{treatment} causes {ate_value:.1f} min wait{ci_str}")
   - Reference variance explained: Show which factors explain what percentage (use variance_explained data)
   - Include confidence: "Analysis confidence: {overall_conf:.0%}"

2. **Underlying Causes** (process issues):
   - Explain confounders: "After controlling for {confounders}, {treatment} effect remains {ate_value:.1f} min"
   - Describe interactions with quantified strength: "Staff shortage × surge amplifies impact by {interaction_strength*100:.0f}% (multiplicative)"
   - Show variance decomposition from variance_explained data

3. **Systemic Causes** (structural problems):
   - Reference probabilistic insights: "P(wait spike | staff_short) = {prob_val:.0%} (Bayesian inference)"
   - Explain causal chain with probabilities from the analysis
   - Show confidence: "Network fit confidence: {bayesian_conf:.0%}"

4. **Counterfactual Analysis with ROI**:
   - For each counterfactual: "If we [intervention from counterfactual]:"
     - Expected wait: [predicted_outcome] min ([improvement_pct]% improvement)
     - ROI: {roi_pct:.0f}% (Daily cost: ${cost:.0f}, Savings: ${savings:.0f}, Payback: {payback_days:.1f} days)
     - Use confidence from confidence_scores
   - Rank by ROI and feasibility

5. **Equity Implications** (CRITICAL):
   - If equity analysis shows disparities: "High-acuity patients wait {disparity_pct:.0f}% longer ({high_acuity_wait:.0f} min vs {low_acuity_wait:.0f} min) - {concern_level} equity concern"
   - Stratify by demographics if available
   - Recommend equity-aware interventions

6. **Prioritized Recommendations** (2-3, ranked by ROI + impact):
   - Format: "[Intervention]: [impact] (ATE: {ate_value:.1f} min{ci_str}) → ROI: {roi_pct:.0f}%, Payback: {payback_days:.1f} days"
   - Include variance explained from variance_explained data
   - Equity-aware: "Reduces disparity by [calculated reduction]%"

7. **Confidence & Limitations**:
   - Overall confidence: {overall_conf:.0%}
   - Data quality: Based on available data
   - Limitations: Mention any data gaps or assumptions

Keep it clinical, data-driven, and actionable. Use SPECIFIC NUMBERS with confidence intervals. 
Avoid generic statements like "insufficient staffing" - quantify with ATE, variance explained, and ROI.
Format as structured markdown with clear sections."""

        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert ED operations analyst. Generate clear, data-driven narratives from causal analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low for factual accuracy
            max_tokens=800
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.warning(f"Narrative generation failed: {e}")
        return _fallback_narrative(causal_analysis, bottleneck_dict)


def _fallback_narrative(causal_analysis: Dict[str, Any], bottleneck: Dict[str, Any]) -> str:
    """Fallback narrative when LLM is unavailable."""
    # Handle both dict and Pydantic model
    if hasattr(bottleneck, 'dict'):
        bottleneck = bottleneck.dict()
    elif hasattr(bottleneck, '__dict__'):
        bottleneck = bottleneck.__dict__
    
    stage = bottleneck.get("stage", "unknown") if isinstance(bottleneck, dict) else getattr(bottleneck, 'stage', 'unknown')
    ate = causal_analysis.get('ate_estimates', {})
    counterfactuals = causal_analysis.get('counterfactuals', [])
    
    bottleneck_name = bottleneck.get('bottleneck_name', 'Bottleneck') if isinstance(bottleneck, dict) else getattr(bottleneck, 'bottleneck_name', 'Bottleneck')
    narrative = f"### Root Cause Analysis: {bottleneck_name}\n\n"
    
    if ate:
        treatment = ate.get('treatment', '')
        value = ate.get('value', 0)
        narrative += f"**Immediate Cause**: {treatment} has an average treatment effect (ATE) of {value:.1f} minutes on wait time.\n\n"
    
    if counterfactuals:
        narrative += "**Counterfactual Scenarios**:\n"
        for cf in counterfactuals[:2]:
            narrative += f"- {cf.get('scenario', 'Intervention')}: Expected {cf.get('improvement_pct', 0):.0f}% improvement\n"
    
    return narrative

