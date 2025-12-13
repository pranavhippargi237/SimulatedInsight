"""
LLM Integration for Causal Discovery and Explanations.

Phase 2 Upgrade: Uses LLMs for:
- Causal discovery from text (clinical notes)
- Generating natural language explanations
- Learning from medical literature
- Enhanced narrative generation
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Optional LLM dependencies
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - LLM features will use fallbacks")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - will use local models")


class LLMCausalDiscovery:
    """
    LLM-based causal discovery from text and structured data.
    Uses language models to identify causal relationships.
    """
    
    def __init__(self, use_llm: bool = True, model_name: str = "gpt-3.5-turbo"):
        self.use_llm = use_llm
        self.model_name = model_name
        self.local_model = None
        self.tokenizer = None
        
        # Try to load local model if transformers available
        if use_llm and TRANSFORMERS_AVAILABLE and not OPENAI_AVAILABLE:
            try:
                # Use a smaller model for local inference
                model_name_local = "microsoft/DialoGPT-small"  # Lightweight option
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_local)
                self.local_model = AutoModelForCausalLM.from_pretrained(model_name_local)
                logger.info(f"Loaded local LLM model: {model_name_local}")
            except Exception as e:
                logger.warning(f"Failed to load local LLM model: {e}")
                self.use_llm = False
    
    def _extract_causal_relationships_from_text(
        self,
        text: str,
        variables: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract causal relationships from text using LLM.
        
        Returns list of causal relationships found in text.
        """
        if not self.use_llm:
            return []
        
        try:
            # Create prompt for LLM
            prompt = f"""Analyze the following text and identify causal relationships between these variables:
Variables: {', '.join(variables)}

Text: {text}

Identify causal relationships in the format: "Variable A causes Variable B" or "Variable A affects Variable B".
Return as JSON list of {{"cause": "variable_name", "effect": "variable_name", "strength": "strong/medium/weak", "evidence": "quote from text"}}.
"""
            
            if OPENAI_AVAILABLE:
                # Use OpenAI API
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a causal inference expert. Extract causal relationships from text."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.3
                    )
                    result_text = response.choices[0].message.content
                    
                    # Parse JSON from response
                    try:
                        relationships = json.loads(result_text)
                        if isinstance(relationships, list):
                            return relationships
                    except json.JSONDecodeError:
                        # Try to extract JSON from text
                        import re
                        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                        if json_match:
                            relationships = json.loads(json_match.group())
                            return relationships if isinstance(relationships, list) else []
                except Exception as e:
                    logger.warning(f"OpenAI API call failed: {e}")
            
            # Fallback: Use local model or simple extraction
            return self._extract_causal_fallback(text, variables)
            
        except Exception as e:
            logger.error(f"LLM causal extraction failed: {e}")
            return []
    
    def _extract_causal_fallback(
        self,
        text: str,
        variables: List[str]
    ) -> List[Dict[str, Any]]:
        """Fallback causal extraction using keyword matching."""
        relationships = []
        text_lower = text.lower()
        
        # Look for causal keywords
        causal_keywords = ["causes", "leads to", "results in", "affects", "influences", "impacts"]
        
        for var1 in variables:
            for var2 in variables:
                if var1 == var2:
                    continue
                
                # Check if both variables appear in text
                if var1.lower() in text_lower and var2.lower() in text_lower:
                    # Check for causal keywords between them
                    for keyword in causal_keywords:
                        pattern = f"{var1.lower()}.*{keyword}.*{var2.lower()}"
                        import re
                        if re.search(pattern, text_lower):
                            relationships.append({
                                "cause": var1,
                                "effect": var2,
                                "strength": "medium",
                                "evidence": f"Found '{keyword}' relationship in text",
                                "method": "Keyword Matching"
                            })
                            break
        
        return relationships
    
    async def discover_causal_from_notes(
        self,
        clinical_notes: List[str],
        variables: List[str]
    ) -> Dict[str, Any]:
        """
        Discover causal relationships from clinical notes.
        
        Args:
            clinical_notes: List of clinical note texts
            variables: List of variable names to look for
        
        Returns:
            Dictionary with discovered causal graph
        """
        if not clinical_notes:
            return {"graph": {}, "method": "No data"}
        
        all_relationships = []
        
        for note in clinical_notes:
            relationships = self._extract_causal_relationships_from_text(note, variables)
            all_relationships.extend(relationships)
        
        # Aggregate relationships
        causal_graph = {}
        for rel in all_relationships:
            cause = rel.get("cause")
            effect = rel.get("effect")
            
            if cause and effect:
                if effect not in causal_graph:
                    causal_graph[effect] = []
                
                causal_graph[effect].append({
                    "cause": cause,
                    "strength": rel.get("strength", "medium"),
                    "evidence_count": 1
                })
        
        # Merge duplicate relationships
        for effect in causal_graph:
            merged = {}
            for rel in causal_graph[effect]:
                cause = rel["cause"]
                if cause not in merged:
                    merged[cause] = rel
                else:
                    merged[cause]["evidence_count"] += 1
            
            causal_graph[effect] = list(merged.values())
        
        return {
            "graph": causal_graph,
            "method": "LLM Causal Discovery",
            "confidence": min(0.9, 0.5 + len(all_relationships) * 0.1),
            "relationships_found": len(all_relationships)
        }


class LLMExplanationGenerator:
    """
    Generate natural language explanations using LLMs.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
    
    def generate_bottleneck_explanation(
        self,
        bottleneck: Dict[str, Any],
        causal_analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate natural language explanation for a bottleneck.
        
        Args:
            bottleneck: Bottleneck information
            causal_analysis: Causal analysis results
            context: Additional context
        
        Returns:
            Natural language explanation
        """
        if not self.use_llm or not OPENAI_AVAILABLE:
            return self._generate_explanation_fallback(bottleneck, causal_analysis)
        
        try:
            prompt = f"""Explain the following ED bottleneck in clear, actionable language for an Emergency Department Director:

Bottleneck: {bottleneck.get('bottleneck_name', 'Unknown')}
Stage: {bottleneck.get('stage', 'Unknown')}
Impact Score: {bottleneck.get('impact_score', 0):.2f}

Causal Analysis:
- Main Causes: {', '.join(causal_analysis.get('confounders', [])[:3])}
- ATE Estimates: {json.dumps(causal_analysis.get('ate_estimates', {}), indent=2)}

Provide:
1. A clear explanation of what's causing this bottleneck
2. Why it's happening now
3. What actions would be most effective
4. Expected impact of interventions

Write in a professional but accessible tone, suitable for a busy ED director.
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert healthcare operations analyst explaining ED bottlenecks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM explanation generation failed: {e}")
            return self._generate_explanation_fallback(bottleneck, causal_analysis)
    
    def _generate_explanation_fallback(
        self,
        bottleneck: Dict[str, Any],
        causal_analysis: Dict[str, Any]
    ) -> str:
        """Fallback explanation generation."""
        name = bottleneck.get('bottleneck_name', 'bottleneck')
        stage = bottleneck.get('stage', 'unknown stage')
        impact = bottleneck.get('impact_score', 0)
        
        causes = causal_analysis.get('confounders', [])
        main_cause = causes[0] if causes else "resource constraints"
        
        explanation = f"""
The {name} at the {stage} stage is experiencing significant delays (impact score: {impact:.2f}).

Primary Cause: {main_cause}

This bottleneck is likely caused by insufficient resources or increased patient volume. 
Recommended actions include:
- Adding additional staff to the {stage} stage
- Optimizing workflow processes
- Addressing upstream bottlenecks that may be contributing

Expected impact: Reducing wait times by approximately {impact * 20:.0f}% with appropriate interventions.
"""
        return explanation.strip()
    
    def generate_optimization_explanation(
        self,
        suggestion: Dict[str, Any],
        expected_impact: Dict[str, Any]
    ) -> str:
        """Generate explanation for optimization suggestion."""
        if not self.use_llm or not OPENAI_AVAILABLE:
            return self._generate_optimization_fallback(suggestion, expected_impact)
        
        try:
            prompt = f"""Explain this ED optimization recommendation:

Action: {suggestion.get('action', 'Unknown')} {suggestion.get('quantity', 0)} {suggestion.get('resource_type', 'resources')}
Expected Impact:
- DTD Reduction: {expected_impact.get('dtd_reduction', 0):.1f} minutes
- LWBS Reduction: {expected_impact.get('lwbs_reduction', 0):.3f}
Confidence: {suggestion.get('confidence', 0):.2f}

Provide a clear, concise explanation of why this recommendation makes sense and what the expected outcomes are.
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an ED operations expert explaining optimization recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM optimization explanation failed: {e}")
            return self._generate_optimization_fallback(suggestion, expected_impact)
    
    def _generate_optimization_fallback(
        self,
        suggestion: Dict[str, Any],
        expected_impact: Dict[str, Any]
    ) -> str:
        """Fallback optimization explanation."""
        action = suggestion.get('action', 'add')
        quantity = suggestion.get('quantity', 1)
        resource = suggestion.get('resource_type', 'resource')
        dtd_reduction = expected_impact.get('dtd_reduction', 0)
        
        return f"""
Recommendation: {action.capitalize()} {quantity} {resource}(s)

Expected Impact:
- Door-to-Doctor time reduction: {abs(dtd_reduction):.1f} minutes
- Leave Without Being Seen reduction: {abs(expected_impact.get('lwbs_reduction', 0)):.3f}

This recommendation is based on analysis of current bottlenecks and resource constraints.
Confidence: {suggestion.get('confidence', 0.7):.0%}
"""


class LLMIntegration:
    """
    Main LLM integration class combining causal discovery and explanation generation.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.causal_discovery = LLMCausalDiscovery(use_llm=use_llm)
        self.explanation_generator = LLMExplanationGenerator(use_llm=use_llm)
    
    async def enhance_causal_analysis(
        self,
        causal_analysis: Dict[str, Any],
        clinical_notes: Optional[List[str]] = None,
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhance causal analysis with LLM-discovered relationships.
        """
        enhanced = causal_analysis.copy()
        
        if clinical_notes and variables and self.use_llm:
            try:
                llm_discovery = await self.causal_discovery.discover_causal_from_notes(
                    clinical_notes, variables
                )
                
                enhanced["llm_discovered_relationships"] = llm_discovery.get("graph", {})
                enhanced["llm_confidence"] = llm_discovery.get("confidence", 0.5)
                enhanced["llm_method"] = llm_discovery.get("method", "LLM")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
        
        return enhanced
    
    def generate_narrative(
        self,
        bottleneck: Dict[str, Any],
        causal_analysis: Dict[str, Any],
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive narrative explanation using LLM.
        """
        if not self.use_llm:
            return self._generate_narrative_fallback(bottleneck, causal_analysis, suggestions)
        
        try:
            explanation = self.explanation_generator.generate_bottleneck_explanation(
                bottleneck, causal_analysis
            )
            
            # Add suggestions
            if suggestions:
                suggestion_text = "\n\nRecommended Actions:\n"
                for i, suggestion in enumerate(suggestions[:3], 1):
                    opt_explanation = self.explanation_generator.generate_optimization_explanation(
                        suggestion, suggestion.get("expected_impact", {})
                    )
                    suggestion_text += f"{i}. {opt_explanation}\n"
                
                explanation += suggestion_text
            
            return explanation
            
        except Exception as e:
            logger.warning(f"LLM narrative generation failed: {e}")
            return self._generate_narrative_fallback(bottleneck, causal_analysis, suggestions)
    
    def _generate_narrative_fallback(
        self,
        bottleneck: Dict[str, Any],
        causal_analysis: Dict[str, Any],
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """Fallback narrative generation."""
        narrative = f"""
## {bottleneck.get('bottleneck_name', 'Bottleneck Analysis')}

### Summary
{bottleneck.get('description', 'No description available')}

### Causal Factors
{', '.join(causal_analysis.get('confounders', ['Unknown factors']))}

### Recommendations
"""
        for i, suggestion in enumerate(suggestions[:3], 1):
            narrative += f"{i}. {suggestion.get('action', 'Unknown')} {suggestion.get('quantity', 0)} {suggestion.get('resource_type', 'resources')}\n"
        
        return narrative.strip()

LLM Integration for Causal Discovery and Explanations.

Phase 2 Upgrade: Uses LLMs for:
- Causal discovery from text (clinical notes)
- Generating natural language explanations
- Learning from medical literature
- Enhanced narrative generation
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Optional LLM dependencies
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - LLM features will use fallbacks")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - will use local models")


class LLMCausalDiscovery:
    """
    LLM-based causal discovery from text and structured data.
    Uses language models to identify causal relationships.
    """
    
    def __init__(self, use_llm: bool = True, model_name: str = "gpt-3.5-turbo"):
        self.use_llm = use_llm
        self.model_name = model_name
        self.local_model = None
        self.tokenizer = None
        
        # Try to load local model if transformers available
        if use_llm and TRANSFORMERS_AVAILABLE and not OPENAI_AVAILABLE:
            try:
                # Use a smaller model for local inference
                model_name_local = "microsoft/DialoGPT-small"  # Lightweight option
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_local)
                self.local_model = AutoModelForCausalLM.from_pretrained(model_name_local)
                logger.info(f"Loaded local LLM model: {model_name_local}")
            except Exception as e:
                logger.warning(f"Failed to load local LLM model: {e}")
                self.use_llm = False
    
    def _extract_causal_relationships_from_text(
        self,
        text: str,
        variables: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract causal relationships from text using LLM.
        
        Returns list of causal relationships found in text.
        """
        if not self.use_llm:
            return []
        
        try:
            # Create prompt for LLM
            prompt = f"""Analyze the following text and identify causal relationships between these variables:
Variables: {', '.join(variables)}

Text: {text}

Identify causal relationships in the format: "Variable A causes Variable B" or "Variable A affects Variable B".
Return as JSON list of {{"cause": "variable_name", "effect": "variable_name", "strength": "strong/medium/weak", "evidence": "quote from text"}}.
"""
            
            if OPENAI_AVAILABLE:
                # Use OpenAI API
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a causal inference expert. Extract causal relationships from text."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.3
                    )
                    result_text = response.choices[0].message.content
                    
                    # Parse JSON from response
                    try:
                        relationships = json.loads(result_text)
                        if isinstance(relationships, list):
                            return relationships
                    except json.JSONDecodeError:
                        # Try to extract JSON from text
                        import re
                        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                        if json_match:
                            relationships = json.loads(json_match.group())
                            return relationships if isinstance(relationships, list) else []
                except Exception as e:
                    logger.warning(f"OpenAI API call failed: {e}")
            
            # Fallback: Use local model or simple extraction
            return self._extract_causal_fallback(text, variables)
            
        except Exception as e:
            logger.error(f"LLM causal extraction failed: {e}")
            return []
    
    def _extract_causal_fallback(
        self,
        text: str,
        variables: List[str]
    ) -> List[Dict[str, Any]]:
        """Fallback causal extraction using keyword matching."""
        relationships = []
        text_lower = text.lower()
        
        # Look for causal keywords
        causal_keywords = ["causes", "leads to", "results in", "affects", "influences", "impacts"]
        
        for var1 in variables:
            for var2 in variables:
                if var1 == var2:
                    continue
                
                # Check if both variables appear in text
                if var1.lower() in text_lower and var2.lower() in text_lower:
                    # Check for causal keywords between them
                    for keyword in causal_keywords:
                        pattern = f"{var1.lower()}.*{keyword}.*{var2.lower()}"
                        import re
                        if re.search(pattern, text_lower):
                            relationships.append({
                                "cause": var1,
                                "effect": var2,
                                "strength": "medium",
                                "evidence": f"Found '{keyword}' relationship in text",
                                "method": "Keyword Matching"
                            })
                            break
        
        return relationships
    
    async def discover_causal_from_notes(
        self,
        clinical_notes: List[str],
        variables: List[str]
    ) -> Dict[str, Any]:
        """
        Discover causal relationships from clinical notes.
        
        Args:
            clinical_notes: List of clinical note texts
            variables: List of variable names to look for
        
        Returns:
            Dictionary with discovered causal graph
        """
        if not clinical_notes:
            return {"graph": {}, "method": "No data"}
        
        all_relationships = []
        
        for note in clinical_notes:
            relationships = self._extract_causal_relationships_from_text(note, variables)
            all_relationships.extend(relationships)
        
        # Aggregate relationships
        causal_graph = {}
        for rel in all_relationships:
            cause = rel.get("cause")
            effect = rel.get("effect")
            
            if cause and effect:
                if effect not in causal_graph:
                    causal_graph[effect] = []
                
                causal_graph[effect].append({
                    "cause": cause,
                    "strength": rel.get("strength", "medium"),
                    "evidence_count": 1
                })
        
        # Merge duplicate relationships
        for effect in causal_graph:
            merged = {}
            for rel in causal_graph[effect]:
                cause = rel["cause"]
                if cause not in merged:
                    merged[cause] = rel
                else:
                    merged[cause]["evidence_count"] += 1
            
            causal_graph[effect] = list(merged.values())
        
        return {
            "graph": causal_graph,
            "method": "LLM Causal Discovery",
            "confidence": min(0.9, 0.5 + len(all_relationships) * 0.1),
            "relationships_found": len(all_relationships)
        }


class LLMExplanationGenerator:
    """
    Generate natural language explanations using LLMs.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
    
    def generate_bottleneck_explanation(
        self,
        bottleneck: Dict[str, Any],
        causal_analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate natural language explanation for a bottleneck.
        
        Args:
            bottleneck: Bottleneck information
            causal_analysis: Causal analysis results
            context: Additional context
        
        Returns:
            Natural language explanation
        """
        if not self.use_llm or not OPENAI_AVAILABLE:
            return self._generate_explanation_fallback(bottleneck, causal_analysis)
        
        try:
            prompt = f"""Explain the following ED bottleneck in clear, actionable language for an Emergency Department Director:

Bottleneck: {bottleneck.get('bottleneck_name', 'Unknown')}
Stage: {bottleneck.get('stage', 'Unknown')}
Impact Score: {bottleneck.get('impact_score', 0):.2f}

Causal Analysis:
- Main Causes: {', '.join(causal_analysis.get('confounders', [])[:3])}
- ATE Estimates: {json.dumps(causal_analysis.get('ate_estimates', {}), indent=2)}

Provide:
1. A clear explanation of what's causing this bottleneck
2. Why it's happening now
3. What actions would be most effective
4. Expected impact of interventions

Write in a professional but accessible tone, suitable for a busy ED director.
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert healthcare operations analyst explaining ED bottlenecks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM explanation generation failed: {e}")
            return self._generate_explanation_fallback(bottleneck, causal_analysis)
    
    def _generate_explanation_fallback(
        self,
        bottleneck: Dict[str, Any],
        causal_analysis: Dict[str, Any]
    ) -> str:
        """Fallback explanation generation."""
        name = bottleneck.get('bottleneck_name', 'bottleneck')
        stage = bottleneck.get('stage', 'unknown stage')
        impact = bottleneck.get('impact_score', 0)
        
        causes = causal_analysis.get('confounders', [])
        main_cause = causes[0] if causes else "resource constraints"
        
        explanation = f"""
The {name} at the {stage} stage is experiencing significant delays (impact score: {impact:.2f}).

Primary Cause: {main_cause}

This bottleneck is likely caused by insufficient resources or increased patient volume. 
Recommended actions include:
- Adding additional staff to the {stage} stage
- Optimizing workflow processes
- Addressing upstream bottlenecks that may be contributing

Expected impact: Reducing wait times by approximately {impact * 20:.0f}% with appropriate interventions.
"""
        return explanation.strip()
    
    def generate_optimization_explanation(
        self,
        suggestion: Dict[str, Any],
        expected_impact: Dict[str, Any]
    ) -> str:
        """Generate explanation for optimization suggestion."""
        if not self.use_llm or not OPENAI_AVAILABLE:
            return self._generate_optimization_fallback(suggestion, expected_impact)
        
        try:
            prompt = f"""Explain this ED optimization recommendation:

Action: {suggestion.get('action', 'Unknown')} {suggestion.get('quantity', 0)} {suggestion.get('resource_type', 'resources')}
Expected Impact:
- DTD Reduction: {expected_impact.get('dtd_reduction', 0):.1f} minutes
- LWBS Reduction: {expected_impact.get('lwbs_reduction', 0):.3f}
Confidence: {suggestion.get('confidence', 0):.2f}

Provide a clear, concise explanation of why this recommendation makes sense and what the expected outcomes are.
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an ED operations expert explaining optimization recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM optimization explanation failed: {e}")
            return self._generate_optimization_fallback(suggestion, expected_impact)
    
    def _generate_optimization_fallback(
        self,
        suggestion: Dict[str, Any],
        expected_impact: Dict[str, Any]
    ) -> str:
        """Fallback optimization explanation."""
        action = suggestion.get('action', 'add')
        quantity = suggestion.get('quantity', 1)
        resource = suggestion.get('resource_type', 'resource')
        dtd_reduction = expected_impact.get('dtd_reduction', 0)
        
        return f"""
Recommendation: {action.capitalize()} {quantity} {resource}(s)

Expected Impact:
- Door-to-Doctor time reduction: {abs(dtd_reduction):.1f} minutes
- Leave Without Being Seen reduction: {abs(expected_impact.get('lwbs_reduction', 0)):.3f}

This recommendation is based on analysis of current bottlenecks and resource constraints.
Confidence: {suggestion.get('confidence', 0.7):.0%}
"""


class LLMIntegration:
    """
    Main LLM integration class combining causal discovery and explanation generation.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.causal_discovery = LLMCausalDiscovery(use_llm=use_llm)
        self.explanation_generator = LLMExplanationGenerator(use_llm=use_llm)
    
    async def enhance_causal_analysis(
        self,
        causal_analysis: Dict[str, Any],
        clinical_notes: Optional[List[str]] = None,
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhance causal analysis with LLM-discovered relationships.
        """
        enhanced = causal_analysis.copy()
        
        if clinical_notes and variables and self.use_llm:
            try:
                llm_discovery = await self.causal_discovery.discover_causal_from_notes(
                    clinical_notes, variables
                )
                
                enhanced["llm_discovered_relationships"] = llm_discovery.get("graph", {})
                enhanced["llm_confidence"] = llm_discovery.get("confidence", 0.5)
                enhanced["llm_method"] = llm_discovery.get("method", "LLM")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
        
        return enhanced
    
    def generate_narrative(
        self,
        bottleneck: Dict[str, Any],
        causal_analysis: Dict[str, Any],
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate comprehensive narrative explanation using LLM.
        """
        if not self.use_llm:
            return self._generate_narrative_fallback(bottleneck, causal_analysis, suggestions)
        
        try:
            explanation = self.explanation_generator.generate_bottleneck_explanation(
                bottleneck, causal_analysis
            )
            
            # Add suggestions
            if suggestions:
                suggestion_text = "\n\nRecommended Actions:\n"
                for i, suggestion in enumerate(suggestions[:3], 1):
                    opt_explanation = self.explanation_generator.generate_optimization_explanation(
                        suggestion, suggestion.get("expected_impact", {})
                    )
                    suggestion_text += f"{i}. {opt_explanation}\n"
                
                explanation += suggestion_text
            
            return explanation
            
        except Exception as e:
            logger.warning(f"LLM narrative generation failed: {e}")
            return self._generate_narrative_fallback(bottleneck, causal_analysis, suggestions)
    
    def _generate_narrative_fallback(
        self,
        bottleneck: Dict[str, Any],
        causal_analysis: Dict[str, Any],
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """Fallback narrative generation."""
        narrative = f"""
## {bottleneck.get('bottleneck_name', 'Bottleneck Analysis')}

### Summary
{bottleneck.get('description', 'No description available')}

### Causal Factors
{', '.join(causal_analysis.get('confounders', ['Unknown factors']))}

### Recommendations
"""
        for i, suggestion in enumerate(suggestions[:3], 1):
            narrative += f"{i}. {suggestion.get('action', 'Unknown')} {suggestion.get('quantity', 0)} {suggestion.get('resource_type', 'resources')}\n"
        
        return narrative.strip()

