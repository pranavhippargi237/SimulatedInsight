"""
RL Environment for ED Optimization using Gymnasium.

Phase 1 Upgrade: Full RL environment for Stable-Baselines3 integration.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

@dataclass
class EDObservation:
    """Observation space for ED RL environment."""
    dtd: float  # Door-to-doctor time
    los: float  # Length of stay
    lwbs: float  # Left without being seen rate
    bed_utilization: float
    queue_length: int
    current_staff_nurses: int
    current_staff_doctors: int
    current_staff_techs: int
    hour_of_day: int  # 0-23
    day_of_week: int  # 0-6
    bottleneck_severity: float  # 0-1


class EDOptimizationEnv(gym.Env):
    """
    Gymnasium environment for ED resource optimization.
    
    Phase 1 Upgrade: Full RL environment compatible with Stable-Baselines3.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        max_steps: int = 10
    ):
        super().__init__()
        
        self.constraints = constraints or {
            "budget": 1000.0,
            "max_doctors": 2,
            "max_nurses": 3,
            "max_techs": 2
        }
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: (resource_type, quantity)
        # resource_type: 0=nurse, 1=doctor, 2=tech
        # quantity: 0=none, 1=add 1, 2=add 2
        self.action_space = spaces.MultiDiscrete([3, 3])  # [resource_type, quantity]
        
        # Observation space: 11 features
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([200, 600, 1, 1, 100, 10, 5, 5, 23, 6, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = self._initialize_state(initial_state)
        
    def _initialize_state(self, initial_state: Optional[Dict[str, Any]]) -> np.ndarray:
        """Initialize environment state."""
        if initial_state:
            return np.array([
                initial_state.get("dtd", 35.0),
                initial_state.get("los", 180.0),
                initial_state.get("lwbs", 0.05),
                initial_state.get("bed_utilization", 0.7),
                initial_state.get("queue_length", 5),
                initial_state.get("nurses", 2),
                initial_state.get("doctors", 1),
                initial_state.get("techs", 1),
                initial_state.get("hour", 12),
                initial_state.get("day_of_week", 0),
                initial_state.get("bottleneck_severity", 0.5)
            ], dtype=np.float32)
        else:
            # Default state
            return np.array([35.0, 180.0, 0.05, 0.7, 5, 2, 1, 1, 12, 0, 0.5], dtype=np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        self.current_step = 0
        
        if options and "initial_state" in options:
            self.state = self._initialize_state(options["initial_state"])
        else:
            self.state = self._initialize_state(None)
        
        return self.state.copy(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: [resource_type, quantity]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        resource_type = int(action[0])  # 0=nurse, 1=doctor, 2=tech
        quantity = int(action[1])  # 0=none, 1=add 1, 2=add 2
        
        # Resource costs
        costs = {0: 50.0, 1: 200.0, 2: 75.0}  # nurse, doctor, tech
        resource_names = {0: "nurse", 1: "doctor", 2: "tech"}
        
        # Calculate cost
        cost = costs[resource_type] * quantity
        
        # Check constraints
        budget = self.constraints.get("budget", 1000.0)
        if cost > budget:
            # Invalid action - high penalty
            reward = -100.0
            terminated = False
            truncated = self.current_step >= self.max_steps
            self.current_step += 1
            return self.state.copy(), reward, terminated, truncated, {"invalid_action": True}
        
        # Apply action (add resources)
        if quantity > 0:
            if resource_type == 0:  # nurse
                self.state[5] = min(self.state[5] + quantity, self.constraints.get("max_nurses", 3))
            elif resource_type == 1:  # doctor
                self.state[6] = min(self.state[6] + quantity, self.constraints.get("max_doctors", 2))
            elif resource_type == 2:  # tech
                self.state[7] = min(self.state[7] + quantity, self.constraints.get("max_techs", 2))
        
        # Simulate impact on metrics
        # DTD reduction based on resources
        dtd_reduction = (
            -15.0 * self.state[5] * 0.1 +  # Nurses help
            -20.0 * self.state[6] * 0.15 +  # Doctors help more
            -8.0 * self.state[7] * 0.05  # Techs help less
        )
        self.state[0] = max(10.0, self.state[0] + dtd_reduction)
        
        # LWBS reduction
        lwbs_reduction = (
            -10.0 * self.state[5] * 0.1 +
            -15.0 * self.state[6] * 0.15
        )
        self.state[2] = max(0.0, min(1.0, self.state[2] + lwbs_reduction))
        
        # Calculate reward
        # Reward = improvement in metrics - cost penalty
        dtd_improvement = max(0, 45.0 - self.state[0])  # Reward for DTD < 45
        lwbs_improvement = max(0, 0.05 - self.state[2]) * 100  # Reward for LWBS < 5%
        cost_penalty = cost / 100.0  # Normalize cost
        
        reward = (
            dtd_improvement * 2.0 +  # DTD improvement is important
            lwbs_improvement * 10.0 +  # LWBS is very important
            -cost_penalty  # Cost penalty
        )
        
        # Update step
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            "dtd": float(self.state[0]),
            "lwbs": float(self.state[2]),
            "cost": cost,
            "resource_type": resource_names[resource_type],
            "quantity": quantity
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def render(self):
        """Render environment (optional)."""
        pass

