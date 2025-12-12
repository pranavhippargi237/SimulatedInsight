"""
Optimization layer using rule-based heuristics and linear programming.
"""
import logging
from typing import List, Dict, Any, Optional
import pulp
from app.data.schemas import OptimizationRequest, OptimizationSuggestion, Bottleneck

logger = logging.getLogger(__name__)


class Optimizer:
    """Optimization engine for ED operations."""
    
    def __init__(self):
        self.resource_costs = {
            "nurse": 50.0,  # per shift
            "doctor": 200.0,
            "tech": 75.0,
            "bed": 100.0  # per day
        }
    
    async def optimize(
        self,
        request: OptimizationRequest,
        bottlenecks: List[Bottleneck],
        historical_sims: Optional[List[Dict[str, Any]]] = None
    ) -> List[OptimizationSuggestion]:
        """
        Generate optimization suggestions.
        
        Args:
            request: Optimization request with constraints
            bottlenecks: List of detected bottlenecks
            historical_sims: Historical simulation results for learning
            
        Returns:
            Ranked list of optimization suggestions
        """
        suggestions = []
        
        # Rule-based suggestions
        rule_based = await self._generate_rule_based_suggestions(
            bottlenecks,
            request.constraints
        )
        suggestions.extend(rule_based)
        
        # LP-based suggestions
        lp_based = await self._generate_lp_suggestions(
            request,
            bottlenecks
        )
        suggestions.extend(lp_based)
        
        # Rank by expected impact
        suggestions.sort(key=lambda x: (
            x.expected_impact.get("dtd_reduction", 0) +
            x.expected_impact.get("lwbs_drop", 0) * 10
        ), reverse=True)
        
        # Assign priorities
        for i, suggestion in enumerate(suggestions[:10], 1):  # Top 10
            suggestion.priority = i
        
        return suggestions[:10]
    
    async def _generate_rule_based_suggestions(
        self,
        bottlenecks: List[Bottleneck],
        constraints: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Generate suggestions based on rules and heuristics with realistic limits."""
        suggestions = []
        
        staff_max = constraints.get("staff_max", 20)
        budget = constraints.get("budget", 1000.0)
        
        # Track what we've already suggested to avoid duplicates
        suggested_resources = set()
        
        for bottleneck in bottlenecks:
            # Rule 1: If bottleneck is in triage, add triage nurse (MAX 2 nurses total)
            if bottleneck.stage == "triage" and "nurse" not in suggested_resources:
                cost = self.resource_costs.get("nurse", 50.0)
                if cost <= budget:
                    suggestions.append(OptimizationSuggestion(
                        priority=0,  # Will be reassigned
                        action="add",
                        resource_type="nurse",
                        quantity=1,  # Always suggest 1 at a time (realistic)
                        expected_impact={
                            "dtd_reduction": -15.0,
                            "lwbs_drop": -10.0
                        },
                        cost=cost,
                        confidence=0.8
                    ))
                    suggested_resources.add("nurse")
            
            # Rule 2: If bottleneck is in doctor stage, add doctor (MAX 1-2 doctors)
            elif bottleneck.stage == "doctor" and "doctor" not in suggested_resources:
                cost = self.resource_costs.get("doctor", 200.0)
                if cost <= budget:
                    # Only suggest 1 doctor at a time (realistic - most EDs can't add 20 doctors!)
                    suggestions.append(OptimizationSuggestion(
                        priority=0,
                        action="add",
                        resource_type="doctor",
                        quantity=1,  # REALISTIC: 1 doctor at a time
                        expected_impact={
                            "dtd_reduction": -20.0,
                            "lwbs_drop": -15.0
                        },
                        cost=cost,
                        confidence=0.85
                    ))
                    suggested_resources.add("doctor")
            
            # Rule 3: If bottleneck is in bed stage, optimize bed turnover
            elif bottleneck.stage == "bed":
                suggestions.append(OptimizationSuggestion(
                    priority=0,
                    action="optimize",
                    resource_type="bed",
                    quantity=0,
                    expected_impact={
                        "los_reduction": -10.0,
                        "bed_utilization": -5.0
                    },
                    cost=0.0,
                    confidence=0.7
                ))
            
            # Rule 4: If bottleneck is in imaging, add tech or cross-train
            elif bottleneck.stage == "imaging":
                cost = self.resource_costs.get("tech", 75.0)
                if cost <= budget:
                    suggestions.append(OptimizationSuggestion(
                        priority=0,
                        action="add",
                        resource_type="tech",
                        quantity=1,
                        expected_impact={
                            "dtd_reduction": -8.0,
                            "los_reduction": -5.0
                        },
                        cost=cost,
                        confidence=0.75
                    ))
        
        return suggestions
    
    async def _generate_lp_suggestions(
        self,
        request: OptimizationRequest,
        bottlenecks: List[Bottleneck]
    ) -> List[OptimizationSuggestion]:
        """Generate suggestions using linear programming with realistic constraints."""
        suggestions = []
        
        constraints = request.constraints
        budget = constraints.get("budget", 1000.0)
        staff_max = constraints.get("staff_max", 20)
        
        # REALISTIC CONSTRAINTS based on typical ED sizes:
        # Small ED (10-20 beds): 2-3 doctors, 4-6 nurses, 1-2 techs
        # Medium ED (20-40 beds): 3-5 doctors, 6-10 nurses, 2-3 techs
        # Large ED (40+ beds): 5-8 doctors, 10-15 nurses, 3-4 techs
        # 
        # We assume medium ED (20 beds) as baseline, so realistic additions are:
        # - Doctors: MAX 2 additions (most EDs have 3-5 total, adding 1-2 is realistic)
        # - Nurses: MAX 3 additions (most EDs have 6-10 total)
        # - Techs: MAX 2 additions (most EDs have 2-3 total)
        
        # Create LP problem
        prob = pulp.LpProblem("ED_Optimization", pulp.LpMaximize)
        
        # Decision variables: number of each resource to add (REALISTIC LIMITS)
        nurses = pulp.LpVariable("nurses", lowBound=0, upBound=min(3, staff_max), cat="Integer")
        doctors = pulp.LpVariable("doctors", lowBound=0, upBound=min(2, staff_max), cat="Integer")  # MAX 2 doctors
        techs = pulp.LpVariable("techs", lowBound=0, upBound=min(2, staff_max), cat="Integer")  # MAX 2 techs
        
        # Objective: maximize DTD reduction (minimize DTD)
        # Simplified: maximize weighted sum of resources
        prob += (
            15 * nurses +  # Each nurse reduces DTD by ~15%
            20 * doctors +  # Each doctor reduces DTD by ~20%
            8 * techs
        )
        
        # Constraints
        prob += (
            self.resource_costs["nurse"] * nurses +
            self.resource_costs["doctor"] * doctors +
            self.resource_costs["tech"] * techs <= budget,
            "Budget"
        )
        
        prob += nurses + doctors + techs <= staff_max, "Staff_Max"
        
        # Solve
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                # Extract solution
                nurse_count = int(nurses.varValue) if nurses.varValue else 0
                doctor_count = int(doctors.varValue) if doctors.varValue else 0
                tech_count = int(techs.varValue) if techs.varValue else 0
                
                # Create suggestions from solution
                if nurse_count > 0:
                    suggestions.append(OptimizationSuggestion(
                        priority=0,
                        action="add",
                        resource_type="nurse",
                        quantity=nurse_count,
                        expected_impact={
                            "dtd_reduction": -15.0 * nurse_count,
                            "lwbs_drop": -10.0 * nurse_count
                        },
                        cost=self.resource_costs["nurse"] * nurse_count,
                        confidence=0.8
                    ))
                
                if doctor_count > 0:
                    suggestions.append(OptimizationSuggestion(
                        priority=0,
                        action="add",
                        resource_type="doctor",
                        quantity=doctor_count,
                        expected_impact={
                            "dtd_reduction": -20.0 * doctor_count,
                            "lwbs_drop": -15.0 * doctor_count
                        },
                        cost=self.resource_costs["doctor"] * doctor_count,
                        confidence=0.85
                    ))
                
                if tech_count > 0:
                    suggestions.append(OptimizationSuggestion(
                        priority=0,
                        action="add",
                        resource_type="tech",
                        quantity=tech_count,
                        expected_impact={
                            "dtd_reduction": -8.0 * tech_count,
                            "los_reduction": -5.0 * tech_count
                        },
                        cost=self.resource_costs["tech"] * tech_count,
                        confidence=0.75
                    ))
        
        except Exception as e:
            logger.warning(f"LP optimization failed: {e}")
        
        return suggestions

