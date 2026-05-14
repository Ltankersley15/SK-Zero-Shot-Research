"""
LLM-Enhanced Command Reasoner with Internal Monologue.

This module extends the basic CommandReasoner with LLM capabilities (OpenAI or Ollama) for:
- Deep semantic understanding of commands
- Internal monologue / reasoning traces
- Context-aware interpretation
- Handling ambiguous or complex commands
- LLM-enhanced object scoring
"""

import json
from typing import Dict, Any, List, Optional, Union

from .command_reasoner import CommandReasoner, TargetSpec, score_label_against_target
from .openai_client import OpenAIClient

# Type alias for LLM clients
LLMClient = Union[OpenAIClient, Any]  # Any can be OllamaClient


class LLMCommandReasoner:
    """
    Enhanced command reasoner using LLM for deep semantic understanding.

    This reasoner provides:
    - Internal monologue showing the reasoning process
    - Context-aware command interpretation
    - Handling of relational and affordance-based commands
    - Confidence scoring with explanations
    - LLM-enhanced object scoring
    
    Supports both OpenAI and Ollama clients.
    """

    SYSTEM_PROMPT = """You are an AI reasoning engine for a robotic manipulation system.
The robot has a Franka Emika FR3 arm with a two-finger gripper.

Your task is to analyze natural language commands and provide deep reasoning about:
1. What task the user wants the robot to perform
2. What object(s) are being referenced
3. Any constraints or special requirements
4. Potential ambiguities and how to resolve them

Think step-by-step and provide your reasoning as an "internal monologue" that can help debug the robot's behavior."""

    REASONING_PROMPT = """Analyze this robot command and provide detailed reasoning.

Command: "{command}"

{scene_context}

Provide your analysis in JSON format:
{{
    "internal_monologue": "Step-by-step reasoning about what the user wants. Think aloud about possible interpretations and which is most likely.",
    "task_type": "pick|place|move|inspect|pour|stack|unknown",
    "target_description": "Clear description of target object(s)",
    "target_attributes": {{
        "color": "expected color or null",
        "shape": "expected shape or null",
        "size": "expected size or null"
    }},
    "spatial_constraints": ["list of spatial requirements"],
    "manipulation_constraints": ["list of manipulation requirements"],
    "ambiguities": ["any ambiguities in the command"],
    "confidence": 0.0-1.0,
    "suggested_verification": "How the robot should verify it understood correctly"
}}

Be thorough in your internal monologue. Show your reasoning process."""

    SCORING_PROMPT = """Command: {command}
Candidate object: {object_desc}

Score how well this object matches the command.

Consider:
1. Does the object match the description (color, shape, size)?
2. Is the object in a reachable/valid position?
3. Are there any constraints that would prevent manipulation?
4. How confident are you that this is the intended target?

Provide your reasoning in JSON format:
{{
    "score": 0.0-1.0,
    "reasoning_trace": "Step-by-step reasoning about the match",
    "color_match": true/false/null (null if color not specified),
    "shape_match": true/false/null (null if shape not specified),
    "position_valid": true/false,
    "is_best_candidate": true/false,
    "confidence": 0.0-1.0
}}"""

    def __init__(
        self,
        llm_client: LLMClient,
        fallback_to_rule_based: bool = True,
    ):
        """
        Initialize LLM command reasoner.

        Args:
            llm_client: Configured LLM client (OpenAI or Ollama)
            fallback_to_rule_based: If True, use rule-based parser when LLM fails
        """
        self.client = llm_client
        self.fallback_to_rule_based = fallback_to_rule_based

        # Keep a rule-based reasoner for fallback
        self.rule_based_reasoner = CommandReasoner()
    
    def reason_about_command(
        self,
        command: str,
        scene_context: Optional[Dict[str, Any]] = None,
        include_internal_monologue: bool = True,
    ) -> Dict[str, Any]:
        """
        Use LLM to reason about a command with internal monologue.

        Args:
            command: Natural language command from user.
            scene_context: Optional scene analysis from VLM.
            include_internal_monologue: Whether to include reasoning trace.

        Returns:
            Dictionary with reasoning results including internal monologue.
        """
        # Format scene context if available
        scene_context_str = ""
        if scene_context:
            if isinstance(scene_context, dict):
                scene_context_str = f"Scene context:\n{json.dumps(scene_context, indent=2)}\n"
            else:
                scene_context_str = f"Scene context:\n{str(scene_context)}\n"

        try:
            # Check if client has reason_about_command method (Ollama style)
            if hasattr(self.client, 'reason_about_command'):
                result = self.client.reason_about_command(
                    command=command,
                    scene_context=scene_context if scene_context else None,
                )
            else:
                # OpenAI style - use _chat_with_json_output
                prompt = self.REASONING_PROMPT.format(
                    command=command,
                    scene_context=scene_context_str,
                )
                response_text = self.client._chat_with_json_output(
                    user_message=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                )
                # Parse JSON response
                json_str = response_text
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                result = json.loads(json_str.strip())

            # Ensure internal monologue is included (or removed based on flag)
            if not include_internal_monologue and "internal_monologue" in result:
                del result["internal_monologue"]

            return result

        except (json.JSONDecodeError, IndexError, Exception) as e:
            # Fallback to rule-based reasoning
            if self.fallback_to_rule_based:
                target_spec = self.rule_based_reasoner.parse(command)
                return {
                    "internal_monologue": f"LLM parsing failed: {str(e)}. Using rule-based fallback." if include_internal_monologue else None,
                    "task_type": "pick",  # Default assumption
                    "target_description": target_spec.noun_phrase or command,
                    "target_attributes": {
                        "color": target_spec.target_color,
                        "shape": target_spec.target_shape,
                        "size": None,
                    },
                    "spatial_constraints": [],
                    "manipulation_constraints": [],
                    "ambiguities": [],
                    "confidence": 0.5,
                    "suggested_verification": "Proceed with rule-based interpretation",
                    "fallback_used": True,
                }
            else:
                return {
                    "internal_monologue": f"LLM parsing failed: {str(e)}" if include_internal_monologue else None,
                    "task_type": "unknown",
                    "target_description": command,
                    "target_attributes": {"color": None, "shape": None, "size": None},
                    "spatial_constraints": [],
                    "manipulation_constraints": [],
                    "ambiguities": ["LLM parsing failed"],
                    "confidence": 0.3,
                    "suggested_verification": "Ask user for clarification",
                    "error": str(e),
                }
    
    def score_object_with_reasoning(
        self,
        command: str,
        object_label: str,
        object_attributes: Optional[Dict[str, Any]] = None,
        reasoning_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Score how well an object matches a command with detailed reasoning.

        Args:
            command: User command.
            object_label: Object label (e.g., "red cube").
            object_attributes: Optional object attributes from VLM.
            reasoning_context: Optional context from command reasoning.

        Returns:
            Dictionary with score and detailed reasoning.
        """
        obj_desc = object_label
        if object_attributes:
            attrs = []
            if object_attributes.get("color"):
                attrs.append(f"color: {object_attributes['color']}")
            if object_attributes.get("shape"):
                attrs.append(f"shape: {object_attributes['shape']}")
            if object_attributes.get("position"):
                attrs.append(f"position: {object_attributes['position']}")
            if attrs:
                obj_desc += f" ({', '.join(attrs)})"

        prompt = self.SCORING_PROMPT.format(
            command=command,
            object_desc=obj_desc,
        )
        if reasoning_context:
            prompt += f"\n\nContext: {json.dumps(reasoning_context)}"

        try:
            # Check if client has score_object_with_reasoning method (Ollama style)
            if hasattr(self.client, 'score_object_with_reasoning'):
                result = self.client.score_object_with_reasoning(
                    command=command,
                    object_label=object_label,
                    object_attributes=object_attributes,
                    reasoning_context=reasoning_context,
                )
            else:
                # OpenAI style - use _chat_with_json_output
                response_text = self.client._chat_with_json_output(
                    user_message=prompt,
                    system_prompt="You are an AI assistant scoring object-command matches. Be precise and fair.",
                )
                # Parse JSON response
                json_str = response_text
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                result = json.loads(json_str.strip())

            # Normalize scores
            result["score"] = max(0.0, min(1.0, float(result.get("score", 0.0))))
            result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            result["is_best_candidate"] = result["score"] >= 0.7

            return result

        except (json.JSONDecodeError, IndexError, ValueError, Exception) as e:
            # Fallback to rule-based scoring
            target = TargetSpec(
                raw_command=command,
                normalized_command=command.lower(),
                noun_phrase="",
                target_terms=[object_label],
                target_color=None,
                target_shape=None,
                canonical_terms=[],
                relative_size=None,
                target_class_scope="object",
            )
            semantic_score = score_label_against_target(
                label=object_label,
                target=target,
                min_score=1.0,
            )

            return {
                "score": semantic_score.score / 5.0,  # Normalize to 0-1
                "reasoning_trace": f"LLM scoring failed, using rule-based fallback: {str(e)}",
                "color_match": semantic_score.color_match,
                "shape_match": semantic_score.shape_match,
                "position_valid": True,
                "is_best_candidate": semantic_score.semantic_pass,
                "confidence": 0.5,
                "fallback_used": True,
            }
    
    def explain_decision(
        self,
        command: str,
        selected_object: str,
        alternative_objects: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a natural language explanation for why an object was selected.
        
        Args:
            command: User command.
            selected_object: The object that was chosen.
            alternative_objects: Other candidate objects that were considered.
            
        Returns:
            Natural language explanation.
        """
        alt_str = ""
        if alternative_objects:
            alt_str = f"Alternative objects considered: {', '.join(alternative_objects)}"
        
        prompt = f"""Command: {command}
Selected object: {selected_object}
{alt_str}

Explain in 1-2 sentences why this object was selected. Be clear and concise."""
        
        try:
            response = self.client._chat_with_json_output(
                user_message=prompt,
                system_prompt="You are explaining a robot's decision to a user. Be clear and helpful.",
            )
            return response
        except Exception:
            return f"Selected '{selected_object}' as the target for the command."


def create_llm_reasoner(
    llm_client: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    fallback_to_rule_based: bool = True,
) -> Optional[LLMCommandReasoner]:
    """
    Factory function to create LLM command reasoner.
    
    Args:
        llm_client: Pre-configured LLM client (OpenAI or Ollama). If provided, takes precedence.
        api_key: OpenAI API key (used if llm_client is None)
        model: Model name (used if llm_client is None)
        fallback_to_rule_based: Use rule-based fallback when LLM fails

    Returns:
        LLMCommandReasoner or None if LLM not available and fallback disabled
    """
    # If client is already provided, use it directly
    if llm_client is not None:
        return LLMCommandReasoner(
            llm_client=llm_client,
            fallback_to_rule_based=fallback_to_rule_based,
        )
    
    # Try to create OpenAI client
    from .openai_client import create_openai_client
    client = create_openai_client(api_key=api_key, model=model)
    if client is None:
        if fallback_to_rule_based:
            # Return a reasoner that always uses rule-based fallback
            # We'll create it with None client - it will always fallback
            reasoner = LLMCommandReasoner(
                llm_client=None,  # type: ignore
                fallback_to_rule_based=True,
            )
            return reasoner
        return None

    return LLMCommandReasoner(
        llm_client=client,
        fallback_to_rule_based=fallback_to_rule_based,
    )
