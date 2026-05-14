"""
OpenAI API Client for GPT-4o-mini vision and language capabilities.

This module provides a wrapper around the OpenAI API for:
- Visual scene analysis (VLM capabilities)
- Natural language command understanding
- Object matching and scoring with reasoning
- Failure analysis and recovery planning
"""

import base64
import json
import os
from io import BytesIO
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None


class OpenAIClient:
    """Client for OpenAI API with GPT-4o-mini support."""
    
    DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_MAX_TOKENS = 1024
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use. Defaults to gpt-4o-mini-2024-07-18.
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens in response.
        """
        if openai is None:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY not set in environment."
            )
        
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_tokens = max_tokens
        
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @staticmethod
    def uses_max_completion_tokens(model: Optional[str]) -> bool:
        """Newer GPT-5 chat models require max_completion_tokens instead of max_tokens."""
        normalized = str(model or "").strip().lower()
        return normalized.startswith("gpt-5")

    @classmethod
    def completion_token_kwargs(cls, model: Optional[str], max_tokens: Optional[int]) -> Dict[str, Any]:
        """Return the token-limit field accepted by the selected chat-completions model."""
        if max_tokens is None:
            return {}
        token_limit = int(max_tokens)
        if token_limit <= 0:
            return {}
        key = "max_completion_tokens" if cls.uses_max_completion_tokens(model) else "max_tokens"
        return {key: token_limit}
        
    def _image_to_base64(self, image: Image.Image, image_format: str = "JPEG") -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format=image_format, quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _numpy_to_image(self, image_np: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if image_np.dtype == np.float32 or image_np.dtype == np.float64:
            # Normalize if needed
            if image_np.max() > 1.0:
                image_np = image_np / 255.0
            image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np)
    
    def analyze_image(
        self,
        image: Any,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Analyze an image with a text prompt using GPT-4o-mini vision.
        
        Args:
            image: PIL Image, numpy array, or path to image file.
            prompt: Text prompt/question about the image.
            system_prompt: Optional system prompt for context.
            max_tokens: Override default max tokens.
            
        Returns:
            Text response from the model.
        """
        # Convert image to PIL if needed
        if isinstance(image, np.ndarray):
            image = self._numpy_to_image(image)
        elif isinstance(image, str):
            image = Image.open(image)
        
        base64_image = self._image_to_base64(image)
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",  # Use low detail for faster processing
                    },
                },
            ],
        })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.completion_token_kwargs(self.model, max_tokens or self.max_tokens),
        )
        
        return response.choices[0].message.content
    
    def reason_about_command(
        self,
        command: str,
        scene_context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to reason about a command and extract intent.
        
        Args:
            command: Natural language command from user.
            scene_context: Optional dictionary with scene analysis results.
            system_prompt: Optional system prompt.
            
        Returns:
            Dictionary with:
            - task_type: Type of task (pick, place, move, etc.)
            - target_description: Description of target object(s)
            - constraints: List of constraints or requirements
            - reasoning_trace: Step-by-step reasoning
            - confidence: Confidence score 0.0-1.0
        """
        default_system_prompt = """You are an AI assistant for a robotic manipulation system. 
Your task is to analyze natural language commands and extract the robot's task intent.

Respond in JSON format with these fields:
- task_type: One of [pick, place, move, inspect, pour, stack, unknown]
- target_description: Clear description of target object(s)
- constraints: List of constraints (e.g., "grasp from top", "avoid collisions")
- reasoning_trace: Step-by-step reasoning about what the user wants
- confidence: Confidence score 0.0-1.0

Be precise and concise. If the command is ambiguous, note it in reasoning_trace."""

        user_message = f"Command: {command}\n"
        
        if scene_context:
            user_message += f"\nScene context:\n{json.dumps(scene_context, indent=2)}\n"
        
        user_message += "\nExtract the task intent in JSON format."
        
        response_text = self._chat_with_json_output(
            user_message=user_message,
            system_prompt=system_prompt or default_system_prompt,
        )
        
        # Parse JSON response
        try:
            # Extract JSON from response (may have markdown formatting)
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            result = json.loads(json_str.strip())
            return result
        except (json.JSONDecodeError, IndexError) as e:
            # Fallback: return structured response with raw text
            return {
                "task_type": "unknown",
                "target_description": command,
                "constraints": [],
                "reasoning_trace": f"Failed to parse LLM response: {response_text}",
                "confidence": 0.5,
                "error": str(e),
            }
    
    def score_object_match(
        self,
        command: str,
        object_description: str,
        scene_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Score how well an object matches a command.
        
        Args:
            command: User's natural language command.
            object_description: Description of candidate object.
            scene_context: Optional scene context.
            
        Returns:
            Dictionary with:
            - score: Match score 0.0-1.0
            - reasoning: Explanation of the score
            - is_match: Boolean whether object should be selected
        """
        prompt = f"""Command: {command}
Candidate object: {object_description}

Rate how well this object matches the command on a scale of 0.0 to 1.0.
Consider:
- Does the object match the description (color, shape, size)?
- Is the object in a reachable position?
- Are there any constraints that would prevent manipulation?

Respond in JSON format:
{{
    "score": 0.0-1.0,
    "reasoning": "brief explanation",
    "is_match": true/false (true if score >= 0.7)
}}"""
        
        if scene_context:
            prompt += f"\n\nScene context: {json.dumps(scene_context)}"
        
        response_text = self._chat_with_json_output(prompt)
        
        try:
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            result = json.loads(json_str.strip())
            # Ensure score is in valid range
            result["score"] = max(0.0, min(1.0, float(result.get("score", 0.0))))
            result["is_match"] = result["score"] >= 0.7
            return result
        except (json.JSONDecodeError, IndexError, ValueError) as e:
            return {
                "score": 0.5,
                "reasoning": f"Failed to parse LLM response: {response_text}",
                "is_match": False,
                "error": str(e),
            }
    
    def analyze_failure(
        self,
        command: str,
        attempted_action: str,
        error_description: str,
        scene_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a failed action and suggest recovery.
        
        Args:
            command: Original user command.
            attempted_action: What the robot tried to do.
            error_description: What went wrong.
            scene_context: Current scene state.
            
        Returns:
            Dictionary with:
            - failure_reason: Explanation of why it failed
            - recovery_strategy: Suggested recovery action
            - should_retry: Whether to retry the same action
            - modified_parameters: Any parameters to change
        """
        prompt = f"""Original command: {command}
Attempted action: {attempted_action}
Error: {error_description}

Analyze why this failed and suggest how to recover.

Respond in JSON format:
{{
    "failure_reason": "explanation of what went wrong",
    "recovery_strategy": "what to try next",
    "should_retry": true/false,
    "modified_parameters": {{"param_name": "new_value"}},
    "confidence": 0.0-1.0
}}"""
        
        if scene_context:
            prompt += f"\n\nCurrent scene: {json.dumps(scene_context)}"
        
        response_text = self._chat_with_json_output(prompt)
        
        try:
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            result = json.loads(json_str.strip())
            return result
        except (json.JSONDecodeError, IndexError) as e:
            return {
                "failure_reason": f"Failed to analyze: {response_text}",
                "recovery_strategy": "abort and report error",
                "should_retry": False,
                "modified_parameters": {},
                "confidence": 0.3,
                "error": str(e),
            }
    
    def _chat_with_json_output(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Internal method for chat completion with JSON expectation."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.completion_token_kwargs(self.model, self.max_tokens),
        )
        
        return response.choices[0].message.content
    
    def is_available(self) -> bool:
        """Check if the client is properly configured."""
        return self.api_key is not None and self.client is not None


def create_openai_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[OpenAIClient]:
    """
    Factory function to create OpenAI client.
    
    Returns None if OpenAI is not installed or API key is missing.
    """
    try:
        return OpenAIClient(api_key=api_key, model=model)
    except (ImportError, ValueError) as e:
        print(f"[OpenAI] Client creation failed: {e}")
        return None
