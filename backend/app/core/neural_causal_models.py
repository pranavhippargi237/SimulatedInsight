"""
Neural Causal Models (NCM) for advanced causal inference.

Phase 2 Upgrade: Replaces/enhances DoWhy with neural causal models for:
- More accurate causal effect estimation
- Differentiable causal inference
- Neural causal discovery
- Better handling of complex confounders
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional neural causal model dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - neural causal models will use fallbacks")

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy not available - will use neural-only methods")


class NeuralCausalModel(nn.Module):
    """
    Neural network for causal effect estimation.
    Learns to predict outcomes under different treatments.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        treatment_dim: int = 1
    ):
        super(NeuralCausalModel, self).__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for neural causal models")
        
        layers = []
        prev_dim = input_dim + treatment_dim  # Concatenate features + treatment
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        """
        Predict outcome given features and treatment.
        
        Args:
            x: Covariates [batch_size, input_dim]
            treatment: Treatment assignment [batch_size, treatment_dim]
        Returns:
            Predicted outcome [batch_size, output_dim]
        """
        # Concatenate features and treatment
        x_t = torch.cat([x, treatment], dim=1)
        return self.network(x_t)


class NeuralCausalInference:
    """
    Neural Causal Inference engine.
    Uses neural networks to estimate causal effects more accurately than traditional methods.
    """
    
    def __init__(self, use_neural: bool = True):
        self.use_neural = use_neural and TORCH_AVAILABLE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}  # Cache trained models
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for neural causal model.
        
        Returns:
            x: Covariates tensor
            t: Treatment tensor
            y: Outcome tensor
        """
        # Extract features
        x = df[covariates].values.astype(np.float32)
        t = df[treatment].values.astype(np.float32).reshape(-1, 1)
        y = df[outcome].values.astype(np.float32).reshape(-1, 1)
        
        # Normalize
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0) + 1e-8
        x = (x - x_mean) / x_std
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
    
    def _train_neural_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100
    ) -> NeuralCausalModel:
        """Train neural causal model."""
        input_dim = x.shape[1]
        model = NeuralCausalModel(input_dim=input_dim).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        x = x.to(self.device)
        t = t.to(self.device)
        y = y.to(self.device)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(x, t)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        return model
    
    def estimate_ate_neural(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    ) -> Dict[str, Any]:
        """
        Estimate Average Treatment Effect (ATE) using neural causal model.
        
        Args:
            df: DataFrame with treatment, outcome, and covariates
            treatment: Treatment variable name
            outcome: Outcome variable name
            covariates: List of covariate names
        
        Returns:
            Dictionary with ATE estimate and confidence interval
        """
        if not self.use_neural:
            return self._estimate_ate_fallback(df, treatment, outcome, covariates)
        
        try:
            # Prepare data
            x, t, y = self._prepare_data(df, treatment, outcome, covariates)
            
            # Train model
            model = self._train_neural_model(x, t, y, epochs=50)
            
            # Estimate ATE: E[Y(1) - Y(0)]
            model.eval()
            with torch.no_grad():
                x = x.to(self.device)
                
                # Predict under treatment (t=1)
                t_treated = torch.ones_like(t).to(self.device)
                y_treated = model(x, t_treated).cpu().numpy()
                
                # Predict under control (t=0)
                t_control = torch.zeros_like(t).to(self.device)
                y_control = model(x, t_control).cpu().numpy()
                
                # Calculate ATE
                ate = float(np.mean(y_treated - y_control))
                
                # Bootstrap confidence interval
                n_bootstrap = 100
                ate_samples = []
                for _ in range(n_bootstrap):
                    indices = np.random.choice(len(df), len(df), replace=True)
                    x_boot = x[indices]
                    t_treated_boot = torch.ones(len(indices), 1).to(self.device)
                    t_control_boot = torch.zeros(len(indices), 1).to(self.device)
                    
                    y_treated_boot = model(x_boot, t_treated_boot).cpu().numpy()
                    y_control_boot = model(x_boot, t_control_boot).cpu().numpy()
                    ate_samples.append(np.mean(y_treated_boot - y_control_boot))
                
                ci_lower = float(np.percentile(ate_samples, 2.5))
                ci_upper = float(np.percentile(ate_samples, 97.5))
            
            return {
                "ate": ate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "method": "Neural Causal Model",
                "confidence": min(0.95, max(0.5, 1.0 - abs(ci_upper - ci_lower) / abs(ate) if ate != 0 else 0.5))
            }
            
        except Exception as e:
            logger.error(f"Neural ATE estimation failed: {e}")
            return self._estimate_ate_fallback(df, treatment, outcome, covariates)
    
    def _estimate_ate_fallback(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    ) -> Dict[str, Any]:
        """Fallback ATE estimation using simple difference-in-means."""
        treated = df[df[treatment] == 1][outcome].mean()
        control = df[df[treatment] == 0][outcome].mean()
        ate = float(treated - control)
        
        return {
            "ate": ate,
            "ci_lower": ate - 1.96 * df[outcome].std() / np.sqrt(len(df)),
            "ci_upper": ate + 1.96 * df[outcome].std() / np.sqrt(len(df)),
            "method": "Difference-in-Means (Fallback)",
            "confidence": 0.6
        }
    
    def estimate_ite_neural(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str],
        individual_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate Individual Treatment Effect (ITE) using neural causal model.
        
        Returns ITE for specific individual or average ITE.
        """
        if not self.use_neural:
            return {"ite": 0.0, "method": "Fallback", "confidence": 0.5}
        
        try:
            x, t, y = self._prepare_data(df, treatment, outcome, covariates)
            model = self._train_neural_model(x, t, y, epochs=50)
            
            model.eval()
            with torch.no_grad():
                x = x.to(self.device)
                
                if individual_idx is not None:
                    x_individual = x[individual_idx:individual_idx+1]
                    t_treated = torch.ones(1, 1).to(self.device)
                    t_control = torch.zeros(1, 1).to(self.device)
                    
                    y_treated = model(x_individual, t_treated).item()
                    y_control = model(x_individual, t_control).item()
                    ite = float(y_treated - y_control)
                else:
                    # Average ITE
                    t_treated = torch.ones_like(t).to(self.device)
                    t_control = torch.zeros_like(t).to(self.device)
                    
                    y_treated = model(x, t_treated).cpu().numpy()
                    y_control = model(x, t_control).cpu().numpy()
                    ite = float(np.mean(y_treated - y_control))
            
            return {
                "ite": ite,
                "method": "Neural Causal Model",
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Neural ITE estimation failed: {e}")
            return {"ite": 0.0, "method": "Fallback", "confidence": 0.5}
    
    async def discover_causal_structure_neural(
        self,
        df: pd.DataFrame,
        variables: List[str],
        max_parents: int = 3
    ) -> Dict[str, Any]:
        """
        Discover causal structure using neural methods.
        
        Uses neural networks to learn causal relationships.
        """
        if not self.use_neural:
            return {"graph": {}, "method": "Fallback"}
        
        try:
            # Simple neural causal discovery: learn pairwise relationships
            causal_graph = {}
            
            for var in variables:
                if var not in df.columns:
                    continue
                
                # Find variables that predict this one
                parents = []
                for parent_var in variables:
                    if parent_var == var or parent_var not in df.columns:
                        continue
                    
                    # Train simple model to predict var from parent_var
                    try:
                        x = df[[parent_var]].values.astype(np.float32)
                        y = df[var].values.astype(np.float32)
                        
                        if len(x) < 10:
                            continue
                        
                        # Simple linear model
                        x_tensor = torch.tensor(x, dtype=torch.float32)
                        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
                        
                        model = nn.Linear(1, 1)
                        optimizer = optim.Adam(model.parameters(), lr=0.01)
                        criterion = nn.MSELoss()
                        
                        for _ in range(50):
                            optimizer.zero_grad()
                            y_pred = model(x_tensor)
                            loss = criterion(y_pred, y_tensor)
                            loss.backward()
                            optimizer.step()
                        
                        # Check if relationship is significant
                        with torch.no_grad():
                            y_pred = model(x_tensor)
                            r2 = 1 - (criterion(y_pred, y_tensor).item() / y_tensor.var().item())
                            
                            if r2 > 0.1:  # Threshold for causal relationship
                                parents.append({
                                    "variable": parent_var,
                                    "strength": float(r2),
                                    "coefficient": float(model.weight.item())
                                })
                    except Exception:
                        continue
                
                # Sort by strength and take top max_parents
                parents.sort(key=lambda x: x["strength"], reverse=True)
                causal_graph[var] = parents[:max_parents]
            
            return {
                "graph": causal_graph,
                "method": "Neural Causal Discovery",
                "confidence": 0.7
            }
            
        except Exception as e:
            logger.error(f"Neural causal discovery failed: {e}")
            return {"graph": {}, "method": "Fallback"}


class EnhancedCausalInference:
    """
    Enhanced causal inference combining DoWhy and Neural Causal Models.
    Uses neural models when available, falls back to DoWhy, then to simple methods.
    """
    
    def __init__(self, use_neural: bool = True, use_dowhy: bool = True):
        self.use_neural = use_neural
        self.use_dowhy = use_dowhy and DOWHY_AVAILABLE
        self.neural_inference = NeuralCausalInference(use_neural=use_neural) if use_neural else None
    
    async def estimate_causal_effect(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str],
        causal_graph: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate causal effect using best available method.
        
        Priority: Neural Causal Model > DoWhy > Fallback
        """
        # Try neural first
        if self.use_neural and self.neural_inference:
            try:
                result = self.neural_inference.estimate_ate_neural(
                    df, treatment, outcome, covariates
                )
                if result.get("method") == "Neural Causal Model":
                    logger.info("Using Neural Causal Model for ATE estimation")
                    return result
            except Exception as e:
                logger.warning(f"Neural causal estimation failed: {e}")
        
        # Try DoWhy
        if self.use_dowhy and causal_graph:
            try:
                model = CausalModel(
                    data=df,
                    treatment=treatment,
                    outcome=outcome,
                    graph=causal_graph
                )
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.linear_regression"
                )
                
                return {
                    "ate": float(estimate.value),
                    "ci_lower": float(estimate.get_confidence_intervals()[0]),
                    "ci_upper": float(estimate.get_confidence_intervals()[1]),
                    "method": "DoWhy",
                    "confidence": 0.75
                }
            except Exception as e:
                logger.warning(f"DoWhy estimation failed: {e}")
        
        # Fallback
        if self.neural_inference:
            return self.neural_inference._estimate_ate_fallback(df, treatment, outcome, covariates)
        else:
            treated = df[df[treatment] == 1][outcome].mean()
            control = df[df[treatment] == 0][outcome].mean()
            return {
                "ate": float(treated - control),
                "method": "Simple Difference",
                "confidence": 0.5
            }

