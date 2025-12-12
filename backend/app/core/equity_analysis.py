"""
Equity Analysis Engine for ED Operations.

Stratifies metrics by SES proxies and ESI to identify disparities.
Example: "LWBS spike: 2x in low-SES proxies?"
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


class EquityAnalyzer:
    """Analyzes equity in ED operations by stratifying metrics."""
    
    def __init__(self):
        # SES proxies (can be enhanced with actual SDOH data)
        self.ses_proxies = {
            'esi': 'Emergency Severity Index (1-5, lower = higher acuity)',
            'arrival_mode': 'Transport mode (ambulance = higher acuity, walk-in = lower SES proxy)',
            'time_of_day': 'Off-hours arrival (proxy for access barriers)'
        }
    
    async def analyze_equity(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int = 48
    ) -> Dict[str, Any]:
        """
        Analyze equity by stratifying metrics.
        
        Returns:
        - Disparities by ESI level
        - Disparities by arrival mode (SES proxy)
        - Disparities by time of day (access proxy)
        - Equity scores and recommendations
        """
        if not events:
            return {}
        
        events_df = pd.DataFrame(events)
        if events_df.empty:
            return {}
        
        # Ensure timestamp is datetime
        if 'timestamp' in events_df.columns:
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
        
        equity_results = {}
        
        # 1. Stratify by ESI
        esi_stratification = self._stratify_by_esi(events_df, kpis)
        equity_results['esi_stratification'] = esi_stratification
        
        # 2. Stratify by arrival mode (SES proxy)
        arrival_stratification = self._stratify_by_arrival_mode(events_df, kpis)
        equity_results['arrival_stratification'] = arrival_stratification
        
        # 3. Stratify by time of day (access proxy)
        temporal_stratification = self._stratify_by_time(events_df, kpis)
        equity_results['temporal_stratification'] = temporal_stratification
        
        # 4. Calculate overall equity scores
        equity_scores = self._calculate_equity_scores(equity_results)
        equity_results['equity_scores'] = equity_scores
        
        # 5. Generate equity recommendations
        recommendations = self._generate_equity_recommendations(equity_results)
        equity_results['recommendations'] = recommendations
        
        return equity_results
    
    def _stratify_by_esi(
        self,
        events_df: pd.DataFrame,
        kpis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Stratify metrics by ESI level."""
        if 'esi' not in events_df.columns:
            return {}
        
        # Group events by ESI
        esi_groups = {}
        for esi in [1, 2, 3, 4, 5]:
            esi_events = events_df[events_df['esi'] == esi]
            if len(esi_events) == 0:
                continue
            
            # Calculate LWBS rate for this ESI
            lwbs_count = len(esi_events[esi_events['event_type'] == 'lwbs'])
            total_arrivals = len(esi_events[esi_events['event_type'] == 'arrival'])
            lwbs_rate = lwbs_count / total_arrivals if total_arrivals > 0 else 0
            
            # Calculate average wait times
            wait_times = []
            patients = {}
            for _, event in esi_events.iterrows():
                patient_id = event.get('patient_id')
                if not patient_id:
                    continue
                if patient_id not in patients:
                    patients[patient_id] = {'arrival': None, 'doctor_visit': None}
                
                if event.get('event_type') == 'arrival':
                    patients[patient_id]['arrival'] = event.get('timestamp')
                elif event.get('event_type') == 'doctor_visit':
                    patients[patient_id]['doctor_visit'] = event.get('timestamp')
            
            for patient_id, journey in patients.items():
                if journey['arrival'] and journey['doctor_visit']:
                    wait = (journey['doctor_visit'] - journey['arrival']).total_seconds() / 60
                    if wait > 0 and wait < 300:  # Reasonable cap
                        wait_times.append(wait)
            
            avg_wait = np.mean(wait_times) if wait_times else 0
            
            esi_groups[esi] = {
                'count': len(esi_events),
                'lwbs_rate': float(lwbs_rate),
                'avg_wait_minutes': float(avg_wait),
                'acuity_level': 'critical' if esi == 1 else 'high' if esi == 2 else 'moderate' if esi == 3 else 'low'
            }
        
        # Calculate disparities
        if len(esi_groups) >= 2:
            high_acuity = esi_groups.get(1, {}) or esi_groups.get(2, {})
            low_acuity = esi_groups.get(4, {}) or esi_groups.get(5, {})
            
            if high_acuity and low_acuity:
                lwbs_disparity = low_acuity.get('lwbs_rate', 0) / high_acuity.get('lwbs_rate', 0.001) if high_acuity.get('lwbs_rate', 0) > 0 else 0
                wait_disparity = low_acuity.get('avg_wait_minutes', 0) / high_acuity.get('avg_wait_minutes', 0.001) if high_acuity.get('avg_wait_minutes', 0) > 0 else 0
                
                esi_groups['disparities'] = {
                    'lwbs_ratio': float(lwbs_disparity),
                    'wait_ratio': float(wait_disparity),
                    'interpretation': f"Low-acuity (ESI 4-5) patients have {lwbs_disparity:.1f}x LWBS rate and {wait_disparity:.1f}x wait time vs high-acuity (ESI 1-2)"
                }
        
        return esi_groups
    
    def _stratify_by_arrival_mode(
        self,
        events_df: pd.DataFrame,
        kpis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Stratify by arrival mode (ambulance vs walk-in as SES proxy)."""
        # This is a simplified proxy - in real implementation, would use actual SDOH data
        # For now, use metadata or infer from event patterns
        
        arrival_modes = defaultdict(list)
        
        for _, event in events_df.iterrows():
            if event.get('event_type') != 'arrival':
                continue
            
            metadata = event.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    import json
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            # Infer arrival mode (simplified)
            # In real system, would have explicit arrival_mode field
            mode = 'walk_in'  # Default
            if 'ambulance' in str(metadata).lower() or 'ems' in str(metadata).lower():
                mode = 'ambulance'
            
            arrival_modes[mode].append(event)
        
        stratification = {}
        for mode, mode_events in arrival_modes.items():
            if len(mode_events) == 0:
                continue
            
            mode_df = pd.DataFrame(mode_events)
            lwbs_count = len(mode_df[mode_df['event_type'] == 'lwbs'])
            total = len(mode_df[mode_df['event_type'] == 'arrival'])
            lwbs_rate = lwbs_count / total if total > 0 else 0
            
            stratification[mode] = {
                'count': len(mode_events),
                'lwbs_rate': float(lwbs_rate),
                'ses_proxy': 'higher' if mode == 'ambulance' else 'lower'
            }
        
        # Calculate disparity
        if 'walk_in' in stratification and 'ambulance' in stratification:
            walk_in_lwbs = stratification['walk_in']['lwbs_rate']
            ambulance_lwbs = stratification['ambulance']['lwbs_rate']
            
            if ambulance_lwbs > 0:
                disparity_ratio = walk_in_lwbs / ambulance_lwbs
                stratification['disparity'] = {
                    'ratio': float(disparity_ratio),
                    'interpretation': f"Walk-in patients (lower SES proxy) have {disparity_ratio:.1f}x LWBS rate vs ambulance arrivals"
                }
        
        return stratification
    
    def _stratify_by_time(
        self,
        events_df: pd.DataFrame,
        kpis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Stratify by time of day (off-hours as access barrier proxy)."""
        if 'timestamp' not in events_df.columns:
            return {}
        
        events_df['hour'] = events_df['timestamp'].dt.hour
        events_df['is_off_hours'] = ((events_df['hour'] >= 22) | (events_df['hour'] < 6)).astype(int)
        
        stratification = {}
        
        for is_off_hours in [0, 1]:
            subset = events_df[events_df['is_off_hours'] == is_off_hours]
            if len(subset) == 0:
                continue
            
            lwbs_count = len(subset[subset['event_type'] == 'lwbs'])
            total = len(subset[subset['event_type'] == 'arrival'])
            lwbs_rate = lwbs_count / total if total > 0 else 0
            
            time_label = 'off_hours' if is_off_hours else 'regular_hours'
            stratification[time_label] = {
                'count': len(subset),
                'lwbs_rate': float(lwbs_rate),
                'access_barrier': 'higher' if is_off_hours else 'lower'
            }
        
        # Calculate disparity
        if 'off_hours' in stratification and 'regular_hours' in stratification:
            off_hours_lwbs = stratification['off_hours']['lwbs_rate']
            regular_lwbs = stratification['regular_hours']['lwbs_rate']
            
            if regular_lwbs > 0:
                disparity_ratio = off_hours_lwbs / regular_lwbs
                stratification['disparity'] = {
                    'ratio': float(disparity_ratio),
                    'interpretation': f"Off-hours arrivals have {disparity_ratio:.1f}x LWBS rate vs regular hours"
                }
        
        return stratification
    
    def _calculate_equity_scores(self, equity_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall equity scores."""
        scores = {}
        
        # ESI equity score (lower disparity = higher score)
        esi_strat = equity_results.get('esi_stratification', {})
        if 'disparities' in esi_strat:
            lwbs_ratio = esi_strat['disparities'].get('lwbs_ratio', 1.0)
            # Score: 100 if ratio = 1.0 (no disparity), decreases as ratio increases
            esi_score = max(0, 100 - (abs(lwbs_ratio - 1.0) * 50))
            scores['esi_equity'] = float(esi_score)
        
        # Overall equity score (average)
        if scores:
            scores['overall_equity'] = float(np.mean(list(scores.values())))
        
        return scores
    
    def _generate_equity_recommendations(self, equity_results: Dict[str, Any]) -> List[str]:
        """Generate equity-focused recommendations."""
        recommendations = []
        
        esi_strat = equity_results.get('esi_stratification', {})
        if 'disparities' in esi_strat:
            lwbs_ratio = esi_strat['disparities'].get('lwbs_ratio', 1.0)
            if lwbs_ratio > 1.5:
                recommendations.append(f"Address LWBS disparity: Low-acuity patients have {lwbs_ratio:.1f}x higher LWBS rate - consider fast-track for ESI 4-5")
        
        arrival_strat = equity_results.get('arrival_stratification', {})
        if 'disparity' in arrival_strat:
            ratio = arrival_strat['disparity'].get('ratio', 1.0)
            if ratio > 1.5:
                recommendations.append(f"Address access barriers: Walk-in patients (lower SES proxy) have {ratio:.1f}x higher LWBS - improve triage efficiency")
        
        temporal_strat = equity_results.get('temporal_stratification', {})
        if 'disparity' in temporal_strat:
            ratio = temporal_strat['disparity'].get('ratio', 1.0)
            if ratio > 1.5:
                recommendations.append(f"Address off-hours access: Off-hours arrivals have {ratio:.1f}x higher LWBS - consider extended hours or telemedicine")
        
        if not recommendations:
            recommendations.append("No significant equity disparities detected")
        
        return recommendations
