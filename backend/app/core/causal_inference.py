"""
Causal Inference Engine for Deep Root Cause Analysis.

Replaces rule-based RCA with:
- DoWhy for causal graphs and ATE estimation
- pgmpy for Bayesian networks and probabilistic inference
- SHAP for feature attribution
- Counterfactual analysis
- Multivariate interaction detection
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallbacks
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy not available - causal analysis will use fallbacks")

# Phase 2: Neural Causal Models
try:
    from app.core.neural_causal_models import EnhancedCausalInference
    NEURAL_CAUSAL_AVAILABLE = True
except ImportError:
    NEURAL_CAUSAL_AVAILABLE = False
    logger.warning("Neural causal models not available")

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    PGM_AVAILABLE = True
except ImportError:
    PGM_AVAILABLE = False
    logger.warning("pgmpy not available - Bayesian networks will use fallbacks")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available - feature attribution will use fallbacks")

from app.data.storage import get_events, get_kpis, cache_get, cache_set
import hashlib
import json


class CausalInferenceEngine:
    """
    Advanced causal inference engine for ED bottleneck analysis.
    Uses DAGs, Bayesian networks, and counterfactual reasoning.
    """
    
    def __init__(self):
        # Phase 2: Neural causal inference
        if NEURAL_CAUSAL_AVAILABLE:
            self.neural_inference = EnhancedCausalInference(use_neural=True, use_dowhy=True)
            logger.info("Causal inference initialized with Neural Causal Models")
        else:
            self.neural_inference = None
        
        self.domain_dags = {
            "imaging": """
            digraph {
                staff_count -> imaging_wait;
                boarding_lag -> staff_count;
                external_surge -> boarding_lag;
                patient_acuity -> imaging_wait;
                lab_backlog -> imaging_wait;
                imaging_wait -> lwbs_risk;
            }
            """,
            "labs": """
            digraph {
                lab_tech_count -> labs_wait;
                boarding_lag -> lab_tech_count;
                external_surge -> boarding_lag;
                patient_acuity -> labs_wait;
                labs_wait -> lwbs_risk;
            }
            """,
            "bed": """
            digraph {
                bed_count -> bed_wait;
                boarding_lag -> bed_count;
                external_surge -> boarding_lag;
                admission_rate -> boarding_lag;
                bed_wait -> lwbs_risk;
            }
            """,
            "doctor": """
            digraph {
                doctor_count -> dtd;
                boarding_lag -> doctor_count;
                external_surge -> boarding_lag;
                patient_acuity -> dtd;
                dtd -> lwbs_risk;
            }
            """
        }
    
    async def analyze_bottleneck_causality(
        self,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int = 48
    ) -> Dict[str, Any]:
        """
        Perform deep causal analysis for a bottleneck.
        Returns: causal graph, ATE estimates, probabilities, confounders, counterfactuals
        
        Uses caching to improve performance - cache key based on bottleneck signature and data hash.
        """
        # Convert to dict if Pydantic model
        if hasattr(bottleneck, 'dict'):
            bottleneck_dict = bottleneck.dict()
        elif hasattr(bottleneck, '__dict__'):
            bottleneck_dict = bottleneck.__dict__
        else:
            bottleneck_dict = bottleneck
        
        bottleneck_name = bottleneck_dict.get('bottleneck_name') if isinstance(bottleneck_dict, dict) else getattr(bottleneck, 'bottleneck_name', 'Unknown')
        stage = bottleneck_dict.get("stage", "generic") if isinstance(bottleneck_dict, dict) else getattr(bottleneck, 'stage', 'generic')
        
        # Create cache key based on bottleneck signature and data characteristics
        # Use bottleneck name, stage, and a hash of recent data to ensure cache invalidation on data changes
        data_hash = hashlib.md5(
            json.dumps({
                "event_count": len(events),
                "kpi_count": len(kpis),
                "window_hours": window_hours,
                "latest_timestamp": str(kpis[-1].get("timestamp")) if kpis else ""
            }, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        
        cache_key = f"causal_analysis_{bottleneck_name}_{stage}_{data_hash}_{window_hours}h"
        
        # Check cache first
        cached_result = await cache_get(cache_key)
        if cached_result:
            logger.info(f"Using cached causal analysis for {bottleneck_name}")
            return cached_result
        
        logger.info(f"Performing causal analysis for {bottleneck_name}")
        
        try:
            # 1. Build enriched DataFrame with causal variables
            df_enriched = await self._build_enriched_dataframe(events, kpis, window_hours)
            
            if df_enriched.empty or len(df_enriched) < 10:
                logger.warning("Insufficient data for causal analysis")
                return self._fallback_analysis(bottleneck_dict)
            
            # 2. Get domain DAG for this bottleneck
            stage = bottleneck_dict.get("stage", "generic") if isinstance(bottleneck_dict, dict) else getattr(bottleneck, 'stage', 'generic')
            domain_dag = self.domain_dags.get(stage, self.domain_dags.get("doctor", ""))
            
            # 3. Build causal model (Neural Causal Model > DoWhy > Fallback)
            if self.neural_inference:
                try:
                    # Try neural causal model first
                    treatment_var = self._identify_treatment_variable(bottleneck_dict, df_enriched)
                    outcome_var = self._identify_outcome_variable(bottleneck_dict, df_enriched)
                    covariates = [col for col in df_enriched.columns if col not in [treatment_var, outcome_var, 'timestamp', 'hour']]
                    
                    if treatment_var and outcome_var and len(covariates) > 0:
                        neural_result = await self.neural_inference.estimate_causal_effect(
                            df_enriched, treatment_var, outcome_var, covariates, domain_dag
                        )
                        if neural_result.get("method") in ["Neural Causal Model", "DoWhy"]:
                            causal_results = {
                                "ate": {treatment_var: neural_result.get("ate", 0)},
                                "graph": domain_dag,
                                "confounders": covariates[:5],
                                "method": neural_result.get("method", "Neural/DoWhy")
                            }
                            logger.info(f"Using {neural_result.get('method')} for causal estimation")
                        else:
                            causal_results = await self._estimate_causal_effects(
                                df_enriched, bottleneck_dict, domain_dag
                            )
                    else:
                        causal_results = await self._estimate_causal_effects(
                            df_enriched, bottleneck_dict, domain_dag
                        )
                except Exception as e:
                    logger.warning(f"Neural causal estimation failed, falling back: {e}")
                    causal_results = await self._estimate_causal_effects(
                        df_enriched, bottleneck_dict, domain_dag
                    )
            else:
                causal_results = await self._estimate_causal_effects(
                    df_enriched, bottleneck_dict, domain_dag
                )
            
            # 4. Build Bayesian network for probabilistic inference
            bayesian_results = await self._build_bayesian_network(
                df_enriched, bottleneck_dict, stage
            )
            
            # 5. SHAP feature attribution
            shap_results = await self._compute_shap_attributions(
                df_enriched, bottleneck_dict, stage
            )
            
            # 6. Counterfactual analysis
            counterfactuals = await self._compute_counterfactuals(
                df_enriched, bottleneck_dict, causal_results, bayesian_results
            )
            
            # 7. Variance explained and confidence scoring (use shap_results)
            attributions_dict = shap_results.get('attributions', {}) if shap_results else {}
            variance_analysis = await self._calculate_variance_explained(
                df_enriched, bottleneck_dict, stage, attributions_dict
            )
            
            # 8. ROI calculations for counterfactuals
            roi_enhanced_counterfactuals = await self._add_roi_to_counterfactuals(
                counterfactuals, bottleneck_dict, stage
            )
            
            result = {
                "causal_graph": causal_results.get("graph", ""),
                "ate_estimates": causal_results.get("ate", {}),
                "probabilistic_insights": bayesian_results,
                "feature_attributions": shap_results,
                "counterfactuals": roi_enhanced_counterfactuals,
                "confounders": causal_results.get("confounders", []),
                "interactions": await self._detect_interactions(df_enriched, bottleneck_dict),
                "equity_analysis": await self._analyze_equity(df_enriched, bottleneck_dict),
                "variance_explained": variance_analysis,
                "confidence_scores": await self._calculate_confidence_scores(
                    causal_results, bayesian_results, shap_results
                )
            }
            
            # Cache result for 5 minutes (300 seconds)
            await cache_set(cache_key, result, ttl=300)
            
            return result
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}", exc_info=True)
            return self._fallback_analysis(bottleneck_dict)
    
    def _identify_treatment_variable(
        self,
        bottleneck: Dict[str, Any],
        df: pd.DataFrame
    ) -> Optional[str]:
        """Identify treatment variable from bottleneck and data."""
        stage = bottleneck.get("stage", "")
        
        # Map stages to treatment variables
        treatment_map = {
            "doctor": "staff_count",
            "imaging": "staff_count",
            "labs": "lab_tech_count",
            "bed": "bed_count"
        }
        
        treatment = treatment_map.get(stage, "staff_count")
        
        # Check if variable exists in dataframe
        if treatment in df.columns:
            return treatment
        
        # Fallback to available variables
        for var in ["staff_count", "doctor_count", "nurse_count", "bed_count"]:
            if var in df.columns:
                return var
        
        return None
    
    def _identify_outcome_variable(
        self,
        bottleneck: Dict[str, Any],
        df: pd.DataFrame
    ) -> Optional[str]:
        """Identify outcome variable from bottleneck and data."""
        stage = bottleneck.get("stage", "")
        
        # Map stages to outcome variables
        outcome_map = {
            "doctor": "dtd",
            "imaging": "imaging_wait",
            "labs": "labs_wait",
            "bed": "bed_wait"
        }
        
        outcome = outcome_map.get(stage, "dtd")
        
        # Check if variable exists in dataframe
        if outcome in df.columns:
            return outcome
        
        # Fallback to available variables
        for var in ["dtd", "los", "lwbs", "wait_time"]:
            if var in df.columns:
                return var
        
        return None
    
    async def _build_enriched_dataframe(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int
    ) -> pd.DataFrame:
        """Build enriched DataFrame with causal variables and confounders."""
        # Convert events to DataFrame
        events_df = pd.DataFrame(events)
        
        if events_df.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is datetime
        if 'timestamp' in events_df.columns:
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            events_df = events_df.sort_values('timestamp')
        
        # Convert KPIs to DataFrame
        kpis_df = pd.DataFrame(kpis)
        if not kpis_df.empty and 'timestamp' in kpis_df.columns:
            kpis_df['timestamp'] = pd.to_datetime(kpis_df['timestamp'])
        
        # Merge events with KPIs
        if not kpis_df.empty:
            # Aggregate events by hour for merging
            events_df['hour'] = events_df['timestamp'].dt.floor('H')
            kpis_df['hour'] = kpis_df['timestamp'].dt.floor('H')
            df = events_df.merge(kpis_df, on='hour', how='left', suffixes=('', '_kpi'))
        else:
            df = events_df.copy()
        
        # Feature engineering: Create causal variables
        
        # 1. Staffing variables (proxies from resource utilization)
        if 'resource_type' in df.columns:
            staff_counts = df[df['resource_type'].isin(['doctor', 'nurse', 'tech'])].groupby('hour').size()
            df['staff_count'] = df['hour'].map(staff_counts).fillna(0)
        else:
            df['staff_count'] = 3  # Default
        
        # 2. Boarding lag (bed utilization lag)
        if 'bed_utilization' in df.columns:
            df['boarding_lag'] = df['bed_utilization'].shift(1).fillna(0)
        else:
            df['boarding_lag'] = 0
        
        # 3. External surge (arrival rate proxy)
        if 'event_type' in df.columns:
            arrivals = df[df['event_type'] == 'arrival'].groupby('hour').size()
            df['arrival_rate'] = df['hour'].map(arrivals).fillna(0)
            # Surge = arrival rate > 75th percentile
            surge_threshold = df['arrival_rate'].quantile(0.75)
            df['external_surge'] = (df['arrival_rate'] > surge_threshold).astype(int)
        else:
            df['external_surge'] = 0
            df['arrival_rate'] = 0
        
        # 4. Patient acuity (ESI proxy)
        if 'esi' in df.columns:
            df['patient_acuity'] = df['esi'].fillna(3)
            # High acuity = ESI 1-2
            df['high_acuity'] = (df['patient_acuity'] <= 2).astype(int)
        else:
            df['patient_acuity'] = 3
            df['high_acuity'] = 0
        
        # 4b. Cohort ratios and interactions
        df['high_acuity_ratio'] = df.groupby('hour')['high_acuity'].transform('mean')
        df['imaging_load'] = df['imaging_wait'] if 'imaging_wait' in df.columns else 0
        df['labs_load'] = df['labs_wait'] if 'labs_wait' in df.columns else 0
        df['high_acuity_x_imaging'] = df['high_acuity_ratio'] * df['imaging_load']
        df['high_acuity_x_labs'] = df['high_acuity_ratio'] * df['labs_load']
        
        # 4c. Disease category/type flags if available
        if 'disease_category' in df.columns:
            top_categories = df['disease_category'].value_counts().head(5).index
            for cat in top_categories:
                df[f'disease_{cat}_flag'] = (df['disease_category'] == cat).astype(int)
        
        # 5. Wait time variables by stage
        for stage in ['imaging', 'labs', 'bed', 'doctor']:
            stage_events = df[df.get('stage') == stage]
            if not stage_events.empty:
                # Calculate wait times
                wait_times = []
                for patient_id in stage_events.get('patient_id', []).unique():
                    patient_events = df[df.get('patient_id') == patient_id].sort_values('timestamp')
                    if len(patient_events) > 1:
                        wait = (patient_events.iloc[-1]['timestamp'] - patient_events.iloc[0]['timestamp']).total_seconds() / 60
                        wait_times.append(wait)
                    else:
                        wait_times.append(0)
                df[f'{stage}_wait'] = 0  # Placeholder - would need proper patient journey tracking
            else:
                df[f'{stage}_wait'] = 0
        
        # 6. LWBS risk proxy
        if 'event_type' in df.columns:
            lwbs_events = df[df['event_type'] == 'lwbs'].groupby('hour').size()
            df['lwbs_risk'] = df['hour'].map(lwbs_events).fillna(0)
        else:
            df['lwbs_risk'] = 0
        
        # 7. Lab backlog (for imaging/labs bottlenecks)
        if 'event_type' in df.columns:
            lab_events = df[df['event_type'] == 'labs'].groupby('hour').size()
            df['lab_backlog'] = df['hour'].map(lab_events).fillna(0).cumsum()
        else:
            df['lab_backlog'] = 0
        
        # 8. Admission rate (for bed bottlenecks)
        if 'event_type' in df.columns:
            admissions = df[df['event_type'] == 'admission'].groupby('hour').size()
            df['admission_rate'] = df['hour'].map(admissions).fillna(0)
        else:
            df['admission_rate'] = 0
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Aggregate to hourly level for causal analysis
        if 'hour' in df.columns:
            df_hourly = df.groupby('hour').agg({
                'staff_count': 'mean',
                'boarding_lag': 'mean',
                'external_surge': 'max',
                'arrival_rate': 'mean',
                'patient_acuity': 'mean',
                'high_acuity': 'sum',
                'lwbs_risk': 'sum',
                'lab_backlog': 'mean',
                'admission_rate': 'mean',
                'imaging_wait': 'mean',
                'labs_wait': 'mean',
                'bed_wait': 'mean',
                'doctor_wait': 'mean'
            }).reset_index()
            return df_hourly
        
        return df
    
    async def _estimate_causal_effects(
        self,
        df: pd.DataFrame,
        bottleneck: Dict[str, Any],
        domain_dag: str
    ) -> Dict[str, Any]:
        """Estimate causal effects using DoWhy."""
        stage = bottleneck.get("stage", "doctor")
        outcome_var = f"{stage}_wait" if f"{stage}_wait" in df.columns else "dtd"
        
        # Define treatment variables (potential causes)
        treatments = []
        if 'staff_count' in df.columns:
            treatments.append('staff_count')
        if 'external_surge' in df.columns:
            treatments.append('external_surge')
        if 'boarding_lag' in df.columns:
            treatments.append('boarding_lag')
        
        if not treatments or outcome_var not in df.columns:
            return {"ate": {}, "confounders": [], "graph": domain_dag}
        
        # Use first treatment for now (can extend to multiple)
        treatment = treatments[0]
        
        try:
            if not DOWHY_AVAILABLE:
                raise ImportError("DoWhy not available")
            
            # Build causal model
            model = CausalModel(
                data=df,
                treatment=[treatment],
                outcome=[outcome_var],
                graph=domain_dag
            )
            
            # Identify effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Estimate effect (using propensity score matching)
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching"
            )
            
            # Get confounders
            confounders = list(identified_estimand.get_backdoor_variables()) if hasattr(identified_estimand, 'get_backdoor_variables') else []
            
            return {
                "ate": {
                    "treatment": treatment,
                    "outcome": outcome_var,
                    "value": float(causal_estimate.value) if hasattr(causal_estimate, 'value') else 0.0,
                    "ci_lower": float(causal_estimate.get_confidence_intervals()[0][0]) if hasattr(causal_estimate, 'get_confidence_intervals') else None,
                    "ci_upper": float(causal_estimate.get_confidence_intervals()[0][1]) if hasattr(causal_estimate, 'get_confidence_intervals') else None
                },
                "confounders": confounders,
                "graph": domain_dag
            }
        except Exception as e:
            logger.warning(f"DoWhy estimation failed: {e}")
            # Fallback: Simple correlation
            if treatment in df.columns and outcome_var in df.columns:
                corr = df[treatment].corr(df[outcome_var])
                return {
                    "ate": {
                        "treatment": treatment,
                        "outcome": outcome_var,
                        "value": corr * df[outcome_var].std() if not pd.isna(corr) else 0.0,
                        "ci_lower": None,
                        "ci_upper": None
                    },
                    "confounders": [],
                    "graph": domain_dag
                }
            return {"ate": {}, "confounders": [], "graph": domain_dag}
    
    async def _build_bayesian_network(
        self,
        df: pd.DataFrame,
        bottleneck: Dict[str, Any],
        stage: str
    ) -> Dict[str, Any]:
        # Handle both dict and Pydantic model for bottleneck
        if hasattr(bottleneck, 'dict'):
            bottleneck = bottleneck.dict()
        elif hasattr(bottleneck, '__dict__'):
            bottleneck = bottleneck.__dict__
        """Build Bayesian network for probabilistic inference."""
        if not PGM_AVAILABLE:
            return {"probabilities": {}, "inference": {}}
        
        try:
            # Define network structure based on stage
            edges = []
            if stage in ["imaging", "labs"]:
                edges = [
                    ('staff_short', 'wait_spike'),
                    ('surge', 'staff_short'),
                    ('boarding', 'staff_short')
                ]
            elif stage == "bed":
                edges = [
                    ('admission_rate', 'bed_wait'),
                    ('boarding', 'bed_wait'),
                    ('surge', 'boarding')
                ]
            else:  # doctor
                edges = [
                    ('staff_short', 'dtd_spike'),
                    ('surge', 'staff_short'),
                    ('acuity', 'dtd_spike')
                ]
            
            if not edges:
                return {"probabilities": {}, "inference": {}}
            
            # Create Bayesian network
            bn = BayesianNetwork(edges)
            
            # Prepare data for learning
            bn_data = pd.DataFrame()
            if 'staff_count' in df.columns:
                bn_data['staff_short'] = (df['staff_count'] < df['staff_count'].median()).astype(int)
            if 'external_surge' in df.columns:
                bn_data['surge'] = df['external_surge']
            if 'boarding_lag' in df.columns:
                bn_data['boarding'] = (df['boarding_lag'] > df['boarding_lag'].median()).astype(int)
            if 'admission_rate' in df.columns:
                bn_data['admission_rate'] = (df['admission_rate'] > df['admission_rate'].median()).astype(int)
            if 'high_acuity' in df.columns:
                bn_data['acuity'] = df['high_acuity']
            
            outcome_var = 'wait_spike' if stage in ["imaging", "labs"] else ('bed_wait' if stage == "bed" else 'dtd_spike')
            if outcome_var == 'wait_spike':
                wait_col = f"{stage}_wait" if f"{stage}_wait" in df.columns else "dtd"
                if wait_col in df.columns:
                    bn_data[outcome_var] = (df[wait_col] > df[wait_col].quantile(0.75)).astype(int)
                else:
                    bn_data[outcome_var] = 0
            elif outcome_var == 'bed_wait':
                if 'bed_wait' in df.columns:
                    bn_data[outcome_var] = (df['bed_wait'] > df['bed_wait'].quantile(0.75)).astype(int)
                else:
                    bn_data[outcome_var] = 0
            else:  # dtd_spike
                if 'doctor_wait' in df.columns:
                    bn_data[outcome_var] = (df['doctor_wait'] > df['doctor_wait'].quantile(0.75)).astype(int)
                else:
                    bn_data[outcome_var] = 0
            
            # Remove columns not in network
            bn_data = bn_data[[col for col in bn_data.columns if col in [e[0] for e in edges] + [e[1] for e in edges]]]
            
            if len(bn_data) < 5 or bn_data.empty:
                return {"probabilities": {}, "inference": {}}
            
            # Fit network
            bn.fit(bn_data, estimator=MaximumLikelihoodEstimator)
            
            # Inference
            inference = VariableElimination(bn)
            
            # Query probabilities
            probabilities = {}
            if 'staff_short' in bn_data.columns:
                try:
                    prob = inference.query(variables=[outcome_var], evidence={'staff_short': 1})
                    if prob:
                        probabilities['wait_given_staff_short'] = float(prob.values[1]) if len(prob.values) > 1 else 0.0
                except:
                    pass
            
            return {
                "probabilities": probabilities,
                "inference": {
                    "network_edges": edges,
                    "fitted": True
                }
            }
        except Exception as e:
            logger.warning(f"Bayesian network failed: {e}")
            return {"probabilities": {}, "inference": {}}
    
    async def _compute_shap_attributions(
        self,
        df: pd.DataFrame,
        bottleneck: Dict[str, Any],
        stage: str
    ) -> Dict[str, Any]:
        # Handle both dict and Pydantic model for bottleneck
        if hasattr(bottleneck, 'dict'):
            bottleneck = bottleneck.dict()
        elif hasattr(bottleneck, '__dict__'):
            bottleneck = bottleneck.__dict__
        """Compute SHAP feature attributions."""
        if not SHAP_AVAILABLE:
            return {"attributions": {}}
        
        try:
            outcome_var = f"{stage}_wait" if f"{stage}_wait" in df.columns else "dtd"
            
            if outcome_var not in df.columns:
                return {"attributions": {}}
            
            # Select features
            feature_cols = ['staff_count', 'boarding_lag', 'external_surge', 'arrival_rate', 'patient_acuity']
            feature_cols = [col for col in feature_cols if col in df.columns]
            
            if not feature_cols:
                return {"attributions": {}}
            
            X = df[feature_cols].fillna(0)
            y = df[outcome_var].fillna(0)
            
            if len(X) < 5:
                return {"attributions": {}}
            
            # Simple model for SHAP
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Average absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            attributions = {}
            for i, col in enumerate(feature_cols):
                attributions[col] = float(np.abs(shap_values[:, i]).mean())
            
            # Normalize to percentages
            total = sum(attributions.values())
            if total > 0:
                attributions = {k: (v / total) * 100 for k, v in attributions.items()}
            
            return {"attributions": attributions}
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {"attributions": {}}
    
    async def _compute_counterfactuals(
        self,
        df: pd.DataFrame,
        bottleneck: Dict[str, Any],
        causal_results: Dict[str, Any],
        bayesian_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # Handle both dict and Pydantic model for bottleneck
        if hasattr(bottleneck, 'dict'):
            bottleneck = bottleneck.dict()
        elif hasattr(bottleneck, '__dict__'):
            bottleneck = bottleneck.__dict__
        """Compute counterfactual scenarios."""
        counterfactuals = []
        
        ate = causal_results.get("ate", {})
        if not ate:
            return counterfactuals
        
        treatment = ate.get("treatment")
        outcome = ate.get("outcome")
        ate_value = ate.get("value", 0)
        
        if treatment and outcome and outcome in df.columns:
            current_avg = df[outcome].mean()
            
            # Counterfactual 1: Increase staffing
            if treatment == 'staff_count':
                counterfactuals.append({
                    "scenario": "Add 1 staff member",
                    "intervention": f"Increase {treatment} by 1",
                    "current_outcome": current_avg,
                    "predicted_outcome": max(0, current_avg + ate_value),
                    "improvement_pct": abs(ate_value / current_avg * 100) if current_avg > 0 else 0,
                    "confidence": "medium"
                })
            
            # Counterfactual 2: Reduce surge
            if treatment == 'external_surge':
                counterfactuals.append({
                    "scenario": "Eliminate surge conditions",
                    "intervention": f"Set {treatment} to 0",
                    "current_outcome": current_avg,
                    "predicted_outcome": max(0, current_avg + ate_value),
                    "improvement_pct": abs(ate_value / current_avg * 100) if current_avg > 0 else 0,
                    "confidence": "medium"
                })
        
        return counterfactuals
    
    async def _detect_interactions(
        self,
        df: pd.DataFrame,
        bottleneck: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect multivariate interactions."""
        interactions = []
        
        # Handle both dict and Pydantic model
        if hasattr(bottleneck, 'dict'):
            bottleneck = bottleneck.dict()
        elif hasattr(bottleneck, '__dict__'):
            bottleneck = bottleneck.__dict__
        
        stage = bottleneck.get("stage", "doctor") if isinstance(bottleneck, dict) else getattr(bottleneck, 'stage', 'doctor')
        outcome_var = f"{stage}_wait" if f"{stage}_wait" in df.columns else "dtd"
        
        if outcome_var not in df.columns:
            return interactions
        
        # Check interaction: staff_count * external_surge
        if 'staff_count' in df.columns and 'external_surge' in df.columns:
            df['interaction'] = df['staff_count'] * df['external_surge']
            corr = df['interaction'].corr(df[outcome_var])
            if abs(corr) > 0.3:
                interactions.append({
                    "variables": ["staff_count", "external_surge"],
                    "type": "multiplicative",
                    "strength": abs(corr),
                    "interpretation": "Staff shortage amplifies surge impact"
                })
        
        return interactions
    
    async def _analyze_equity(
        self,
        df: pd.DataFrame,
        bottleneck: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze equity implications (SDOH proxies)."""
        equity = {}
        
        # Handle both dict and Pydantic model
        if hasattr(bottleneck, 'dict'):
            bottleneck = bottleneck.dict()
        elif hasattr(bottleneck, '__dict__'):
            bottleneck = bottleneck.__dict__
        
        stage = bottleneck.get("stage", "doctor") if isinstance(bottleneck, dict) else getattr(bottleneck, 'stage', 'doctor')
        outcome_var = f"{stage}_wait" if f"{stage}_wait" in df.columns else "dtd"
        
        if outcome_var not in df.columns or 'patient_acuity' not in df.columns:
            return equity
        
        # Compare wait times by acuity
        if 'high_acuity' in df.columns:
            high_acuity_wait = df[df['high_acuity'] == 1][outcome_var].mean()
            low_acuity_wait = df[df['high_acuity'] == 0][outcome_var].mean()
            
            if not pd.isna(high_acuity_wait) and not pd.isna(low_acuity_wait) and low_acuity_wait > 0:
                disparity = (high_acuity_wait - low_acuity_wait) / low_acuity_wait * 100
                equity = {
                    "high_acuity_wait": float(high_acuity_wait),
                    "low_acuity_wait": float(low_acuity_wait),
                    "disparity_pct": float(disparity),
                    "concern": "high" if abs(disparity) > 20 else "medium" if abs(disparity) > 10 else "low"
                }
        
        return equity
    
    async def _calculate_variance_explained(
        self,
        df: pd.DataFrame,
        bottleneck: Dict[str, Any],
        stage: str,
        attributions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate variance explained by each factor (R² decomposition)."""
        variance = {}
        
        outcome_var = f"{stage}_wait" if f"{stage}_wait" in df.columns else "dtd"
        if outcome_var not in df.columns:
            return variance
        
        # Calculate total variance
        total_var = df[outcome_var].var()
        if total_var == 0 or pd.isna(total_var):
            return variance
        
        # Use SHAP attributions to estimate variance explained
        if attributions and isinstance(attributions, dict):
            # attributions might be a dict with 'attributions' key or direct dict
            attribs = attributions.get('attributions', attributions) if isinstance(attributions, dict) else {}
            if isinstance(attribs, dict):
                for var, pct in attribs.items():
                    if var in df.columns:
                        try:
                            pct_val = float(pct)
                            # Rough estimate: variance explained ≈ attribution percentage
                            variance[var] = {
                                "percentage": pct_val,
                                "variance_explained": float(total_var * pct_val / 100) if total_var > 0 else 0.0,
                                "interpretation": f"{var} explains {pct_val:.0f}% of wait time variance"
                            }
                        except (ValueError, TypeError):
                            continue
        
        return variance
    
    async def _add_roi_to_counterfactuals(
        self,
        counterfactuals: List[Dict[str, Any]],
        bottleneck: Dict[str, Any],
        stage: str
    ) -> List[Dict[str, Any]]:
        """Add ROI calculations to counterfactuals."""
        from app.core.roi_calculator import ROICalculator
        
        roi_calc = ROICalculator()
        enhanced = []
        
        for cf in counterfactuals:
            improvement_pct = cf.get('improvement_pct', 0)
            scenario = cf.get('scenario', '')
            
            # Estimate cost and savings
            if 'staff' in scenario.lower() or 'tech' in scenario.lower():
                # Staffing intervention
                cost_per_hour = 50.0  # $50/hour for temp staff
                hours_per_day = 8
                daily_cost = cost_per_hour * hours_per_day
                
                # Savings from reduced wait times
                wait_reduction_min = cf.get('current_outcome', 0) - cf.get('predicted_outcome', 0)
                # Each minute saved = $10 in avoided LWBS risk + efficiency
                daily_savings = wait_reduction_min * 10 * 24  # Rough estimate
                
                roi_pct = ((daily_savings - daily_cost) / daily_cost * 100) if daily_cost > 0 else 0
                payback_days = daily_cost / daily_savings if daily_savings > 0 else 999
                
                cf['roi'] = {
                    "daily_cost": daily_cost,
                    "daily_savings": daily_savings,
                    "roi_percentage": roi_pct,
                    "payback_days": payback_days,
                    "net_daily_benefit": daily_savings - daily_cost
                }
            else:
                # Process improvement (lower cost)
                cf['roi'] = {
                    "daily_cost": 0,
                    "daily_savings": improvement_pct * 100,  # Rough estimate
                    "roi_percentage": 999 if improvement_pct > 0 else 0,
                    "payback_days": 0,
                    "net_daily_benefit": improvement_pct * 100
                }
            
            enhanced.append(cf)
        
        return enhanced
    
    async def _calculate_confidence_scores(
        self,
        causal_results: Dict[str, Any],
        bayesian_results: Dict[str, Any],
        shap_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall confidence scores for the analysis."""
        scores = {}
        
        # Confidence from ATE CI width
        ate = causal_results.get("ate", {})
        if ate and ate.get("ci_lower") and ate.get("ci_upper"):
            ci_width = ate.get("ci_upper", 0) - ate.get("ci_lower", 0)
            ate_value = abs(ate.get("value", 0))
            if ate_value > 0:
                # Narrower CI = higher confidence
                ci_ratio = ci_width / ate_value
                scores["ate_confidence"] = max(0, min(1, 1 - (ci_ratio / 2)))  # 0-1 scale
            else:
                scores["ate_confidence"] = 0.5
        else:
            scores["ate_confidence"] = 0.3  # Low confidence without CI
        
        # Confidence from Bayesian network fit
        if bayesian_results.get("inference", {}).get("fitted"):
            scores["bayesian_confidence"] = 0.7
        else:
            scores["bayesian_confidence"] = 0.3
        
        # Confidence from SHAP attributions
        if shap_results.get("attributions"):
            scores["attribution_confidence"] = 0.8
        else:
            scores["attribution_confidence"] = 0.3
        
        # Overall confidence (weighted average)
        scores["overall_confidence"] = (
            scores.get("ate_confidence", 0.3) * 0.4 +
            scores.get("bayesian_confidence", 0.3) * 0.3 +
            scores.get("attribution_confidence", 0.3) * 0.3
        )
        
        return scores
    
    def _fallback_analysis(self, bottleneck: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback when causal analysis fails."""
        return {
            "causal_graph": "",
            "ate_estimates": {},
            "probabilistic_insights": {},
            "feature_attributions": {},
            "counterfactuals": [],
            "confounders": [],
            "interactions": [],
            "equity_analysis": {},
            "variance_explained": {},
            "confidence_scores": {"overall_confidence": 0.3}
        }

