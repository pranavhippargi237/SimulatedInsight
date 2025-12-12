"""
Correlation Analysis Engine for ED Metrics.

Identifies correlations between patient types, conditions, and outcomes.
Example: "17% psych + 23% abdominal weekend → +12% LWBS correlation"
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from app.data.storage import get_events, get_kpis

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyzes correlations between patient characteristics and outcomes."""
    
    def __init__(self):
        self.patient_type_keywords = {
            "psych": ["psych", "mental", "behavioral", "psychiatric", "suicide", "overdose"],
            "abdominal": ["abdominal", "stomach", "abdomen", "appendicitis", "gallbladder"],
            "cardiac": ["chest", "cardiac", "heart", "mi", "angina"],
            "respiratory": ["respiratory", "breathing", "asthma", "copd", "pneumonia"],
            "trauma": ["trauma", "injury", "fracture", "laceration"]
        }
    
    async def analyze_correlations(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int = 48
    ) -> Dict[str, Any]:
        """
        Analyze correlations between patient types and outcomes.
        
        Returns correlations like:
        - "psych + abdominal → LWBS": OR 1.12, p<0.01
        - "psych weekend → LOS": +7% correlation
        """
        if not events or not kpis:
            return {}
        
        # Convert to DataFrames
        events_df = pd.DataFrame(events)
        kpis_df = pd.DataFrame(kpis)
        
        if events_df.empty or kpis_df.empty:
            return {}
        
        # Ensure timestamps are datetime
        if 'timestamp' in events_df.columns:
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        if 'timestamp' in kpis_df.columns:
            kpis_df['timestamp'] = pd.to_datetime(kpis_df['timestamp'])
        
        # Aggregate events by hour
        events_df['hour'] = events_df['timestamp'].dt.floor('H')
        kpis_df['hour'] = kpis_df['timestamp'].dt.floor('H')
        
        # Classify patient types from events
        hourly_patient_types = self._classify_patient_types(events_df)
        
        # Merge with KPIs
        merged = hourly_patient_types.merge(kpis_df, on='hour', how='inner')
        
        if merged.empty:
            return {}
        
        # Calculate correlations
        correlations = {}
        
        # 1. Patient type → LWBS correlation
        lwbs_corrs = self._calculate_outcome_correlations(
            merged, 'lwbs', ['psych_pct', 'abdominal_pct', 'cardiac_pct']
        )
        correlations['lwbs'] = lwbs_corrs
        
        # 2. Patient type → LOS correlation
        los_corrs = self._calculate_outcome_correlations(
            merged, 'los', ['psych_pct', 'abdominal_pct', 'cardiac_pct']
        )
        correlations['los'] = los_corrs
        
        # 3. Combined effects (psych + abdominal)
        combined_effects = self._analyze_combined_effects(merged)
        correlations['combined'] = combined_effects
        
        # 4. Temporal patterns (weekend effects)
        temporal_corrs = self._analyze_temporal_correlations(merged)
        correlations['temporal'] = temporal_corrs
        
        return correlations
    
    def _classify_patient_types(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Classify patients by type based on event metadata or stage patterns."""
        hourly_data = []
        
        for hour, hour_events in events_df.groupby('hour'):
            arrivals = hour_events[hour_events['event_type'] == 'arrival']
            total_arrivals = len(arrivals)
            
            if total_arrivals == 0:
                continue
            
            # Count by patient type (using metadata or ESI as proxy)
            type_counts = defaultdict(int)
            for _, event in arrivals.iterrows():
                metadata = event.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # Check metadata for patient type hints
                metadata_str = str(metadata).lower()
                for ptype, keywords in self.patient_type_keywords.items():
                    if any(kw in metadata_str for kw in keywords):
                        type_counts[ptype] += 1
                        break
                
                # Also use ESI as proxy (ESI 1-2 = high acuity, might indicate specific types)
                esi = event.get('esi')
                if esi and esi <= 2:
                    type_counts['high_acuity'] = type_counts.get('high_acuity', 0) + 1
            
            hourly_data.append({
                'hour': hour,
                'total_arrivals': total_arrivals,
                'psych_pct': (type_counts.get('psych', 0) / total_arrivals) * 100 if total_arrivals > 0 else 0,
                'abdominal_pct': (type_counts.get('abdominal', 0) / total_arrivals) * 100 if total_arrivals > 0 else 0,
                'cardiac_pct': (type_counts.get('cardiac', 0) / total_arrivals) * 100 if total_arrivals > 0 else 0,
                'high_acuity_pct': (type_counts.get('high_acuity', 0) / total_arrivals) * 100 if total_arrivals > 0 else 0
            })
        
        return pd.DataFrame(hourly_data)
    
    def _calculate_outcome_correlations(
        self,
        merged_df: pd.DataFrame,
        outcome_col: str,
        predictor_cols: List[str]
    ) -> Dict[str, Any]:
        """Calculate correlations between predictors and outcome."""
        if outcome_col not in merged_df.columns:
            return {}
        
        correlations = {}
        
        for pred_col in predictor_cols:
            if pred_col not in merged_df.columns:
                continue
            
            # Calculate Pearson correlation
            valid_data = merged_df[[pred_col, outcome_col]].dropna()
            if len(valid_data) < 5:
                continue
            
            corr, p_value = stats.pearsonr(valid_data[pred_col], valid_data[outcome_col])
            
            if not np.isnan(corr) and not np.isnan(p_value):
                correlations[pred_col] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'effect_size': abs(corr),
                    'interpretation': self._interpret_correlation(corr, p_value, pred_col, outcome_col)
                }
        
        return correlations
    
    def _analyze_combined_effects(self, merged_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze combined effects (e.g., psych + abdominal → LWBS)."""
        combined_effects = {}
        
        # Create combined variable
        if 'psych_pct' in merged_df.columns and 'abdominal_pct' in merged_df.columns:
            merged_df['psych_abdominal_pct'] = merged_df['psych_pct'] + merged_df['abdominal_pct']
            
            # Correlation with LWBS
            if 'lwbs' in merged_df.columns:
                valid_data = merged_df[['psych_abdominal_pct', 'lwbs']].dropna()
                if len(valid_data) >= 5:
                    corr, p_value = stats.pearsonr(valid_data['psych_abdominal_pct'], valid_data['lwbs'])
                    if not np.isnan(corr):
                        # Calculate odds ratio approximation
                        high_combo = valid_data[valid_data['psych_abdominal_pct'] > valid_data['psych_abdominal_pct'].median()]
                        low_combo = valid_data[valid_data['psych_abdominal_pct'] <= valid_data['psych_abdominal_pct'].median()]
                        
                        if len(high_combo) > 0 and len(low_combo) > 0:
                            high_lwbs = (high_combo['lwbs'] > high_combo['lwbs'].median()).sum()
                            low_lwbs = (low_combo['lwbs'] > low_combo['lwbs'].median()).sum()
                            
                            if low_lwbs > 0:
                                or_approx = (high_lwbs / len(high_combo)) / (low_lwbs / len(low_combo))
                            else:
                                or_approx = float('inf') if high_lwbs > 0 else 1.0
                            
                            combined_effects['psych_abdominal_lwbs'] = {
                                'correlation': float(corr),
                                'p_value': float(p_value),
                                'odds_ratio_approx': float(or_approx) if not np.isinf(or_approx) else None,
                                'interpretation': f"Psych + abdominal combo: {corr*100:.1f}% correlation with LWBS (OR: {or_approx:.2f}, p={p_value:.3f})" if not np.isinf(or_approx) else f"Psych + abdominal combo: {corr*100:.1f}% correlation with LWBS"
                            }
        
        return combined_effects
    
    def _analyze_temporal_correlations(self, merged_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal correlations (e.g., weekend effects)."""
        temporal = {}
        
        if 'hour' not in merged_df.columns:
            return temporal
        
        # Extract day of week and hour
        merged_df['day_of_week'] = merged_df['hour'].dt.dayofweek
        merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6]).astype(int)
        merged_df['hour_of_day'] = merged_df['hour'].dt.hour
        
        # Weekend effects
        if 'psych_pct' in merged_df.columns and 'lwbs' in merged_df.columns:
            weekend_data = merged_df[merged_df['is_weekend'] == 1]
            weekday_data = merged_df[merged_df['is_weekend'] == 0]
            
            if len(weekend_data) > 0 and len(weekday_data) > 0:
                weekend_psych = weekend_data['psych_pct'].mean()
                weekday_psych = weekday_data['psych_pct'].mean()
                weekend_lwbs = weekend_data['lwbs'].mean()
                weekday_lwbs = weekday_data['lwbs'].mean()
                
                if weekday_lwbs > 0:
                    lwbs_increase = ((weekend_lwbs - weekday_lwbs) / weekday_lwbs) * 100
                else:
                    lwbs_increase = 0
                
                temporal['weekend_psych_lwbs'] = {
                    'weekend_psych_pct': float(weekend_psych),
                    'weekday_psych_pct': float(weekday_psych),
                    'weekend_lwbs': float(weekend_lwbs),
                    'weekday_lwbs': float(weekday_lwbs),
                    'lwbs_increase_pct': float(lwbs_increase),
                    'interpretation': f"Weekend psych surge ({weekend_psych:.1f}% vs {weekday_psych:.1f}%) → {lwbs_increase:+.1f}% LWBS increase"
                }
        
        return temporal
    
    def _interpret_correlation(self, corr: float, p_value: float, pred: str, outcome: str) -> str:
        """Generate human-readable interpretation."""
        pred_name = pred.replace('_pct', '').replace('_', ' ').title()
        outcome_name = outcome.upper()
        
        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
        direction = "increases" if corr > 0 else "decreases"
        sig = "significant" if p_value < 0.05 else "not significant"
        
        return f"{pred_name} {strength}ly {direction} {outcome_name} (r={corr:.2f}, p={p_value:.3f}, {sig})"
