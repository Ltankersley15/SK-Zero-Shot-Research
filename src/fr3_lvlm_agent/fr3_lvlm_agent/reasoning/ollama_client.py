"""
Local LLM Client using Ollama.

This module provides a drop-in replacement for the OpenAI client,
allowing you to use local LLMs (LLaVA, Llama 3, etc.) for testing
without API costs.

Installation:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2
    ollama pull llava  # For vision tasks

Usage:
    # In launch file or parameter file:
    llm_provider:="ollama"
    llm_model:="llama3.2"
    llm_base_url:="http://localhost:11434"
"""

import json
import requests
from typing import Dict, Any, Optional, List
import base64
from io import BytesIO
from PIL import Image


class OllamaClient:
    """Client for Ollama local LLM inference."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "llama3.2", "llava", "mistral")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._check_connection()

    def _check_connection(self):
        """Check if Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"[Ollama] Connected. Available models: {model_names}")
                # Check for model with prefix matching (handles 'llama3.2:latest' vs 'llama3.2')
                model_found = any(self.model in m or m.startswith(self.model.split(':')[0]) for m in model_names)
                if not model_found:
                    print(f"[Ollama] Warning: Model '{self.model}' not found. Install with: ollama pull {self.model}")
            else:
                print(f"[Ollama] Warning: Unexpected response: {response.status_code}")
        except Exception as e:
            print(f"[Ollama] Warning: Cannot connect to server at {self.base_url}: {e}")
            print("[Ollama] Install: curl -fsSL https://ollama.com/install.sh | sh")
            print("[Ollama] Then run: ollama pull llama3.2")

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Response dict with 'content' field
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.7),
                "num_predict": kwargs.get('max_tokens', 1024),
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return {"content": result['message']['content']}
        except Exception as e:
            return {"content": f"Error: {str(e)}"}

    def analyze_image(
        self,
        image: Image.Image,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze an image with VLM.

        Args:
            image: PIL Image object
            prompt: Question/prompt about the image
            **kwargs: Additional parameters

        Returns:
            Response dict with analysis
        """
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.3),
                "num_predict": kwargs.get('max_tokens', 512),
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return {"content": result['response']}
        except Exception as e:
            return {"content": f"Error: {str(e)}"}

    def reason_about_command(
        self,
        command: str,
        scene_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Reason about a manipulation command.

        Args:
            command: Natural language command
            scene_context: Optional scene description
            **kwargs: Additional parameters

        Returns:
            Structured reasoning result
        """
        system_prompt = """You are a robotic manipulation reasoning engine.
Analyze commands and provide structured output in JSON format.

IMPORTANT: Respond with ONLY valid JSON, no additional text. Format:
{
    "internal_monologue": "Your step-by-step reasoning",
    "task_type": "pick|place|move|stack|inspect|unknown",
    "target_description": "Description of target object(s)",
    "target_attributes": {
        "color": "expected color or null",
        "shape": "expected shape or null"
    },
    "spatial_constraints": ["list of spatial requirements"],
    "confidence": 0.0-1.0
}"""

        user_prompt = f"Analyze this robot command: \"{command}\""
        if scene_context:
            user_prompt += f"\n\nScene context: {json.dumps(scene_context)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.chat(messages, **kwargs)

        # Try to parse JSON from response
        content = response['content']
        try:
            # Extract JSON from response (may have markdown formatting or extra text)
            json_content = content
            if '```json' in content:
                json_content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                json_content = content.split('```')[1].split('```')[0].strip()
            else:
                # Try to find JSON object in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]

            result = json.loads(json_content)
            result['raw_response'] = response['content']
            return result
        except Exception as e:
            # Try to extract useful information from non-JSON response
            return {
                'internal_monologue': content[:500] if len(content) > 500 else content,
                'task_type': self._extract_task_type(content),
                'target_description': content,
                'target_attributes': {
                    'color': self._extract_color(content),
                    'shape': self._extract_shape(content),
                },
                'spatial_constraints': [],
                'confidence': 0.5,
                'raw_response': response['content'],
                'parse_error': str(e),
            }
    
    def _extract_task_type(self, text: str) -> str:
        """Extract task type from non-JSON response."""
        text_lower = text.lower()
        if 'pick' in text_lower or 'grasp' in text_lower or 'pickup' in text_lower:
            return 'pick'
        elif 'place' in text_lower or 'put' in text_lower:
            return 'place'
        elif 'move' in text_lower:
            return 'move'
        elif 'stack' in text_lower:
            return 'stack'
        elif 'inspect' in text_lower or 'examine' in text_lower:
            return 'inspect'
        return 'unknown'
    
    def _extract_color(self, text: str) -> Optional[str]:
        """Extract color from text."""
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white']
        text_lower = text.lower()
        for color in colors:
            if color in text_lower:
                return color
        return None
    
    def _extract_shape(self, text: str) -> Optional[str]:
        """Extract shape from text."""
        shapes = ['cube', 'block', 'cylinder', 'tube', 'sphere', 'box']
        text_lower = text.lower()
        for shape in shapes:
            if shape in text_lower:
                return shape
        return None

    def score_object_with_reasoning(
        self,
        command: str,
        object_label: str,
        object_attributes: Optional[Dict[str, Any]] = None,
        reasoning_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Score how well an object matches a command.

        Args:
            command: User command
            object_label: Object label
            object_attributes: Optional object attributes
            reasoning_context: Optional reasoning context
            **kwargs: Additional parameters

        Returns:
            Structured scoring result
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

        system_prompt = """You are an AI assistant scoring object-command matches.
Score how well the candidate object matches the command on a scale of 0.0 to 1.0.

Consider:
1. Does the object match the description (color, shape, size)?
2. Is the object in a reachable/valid position?
3. Are there any constraints that would prevent manipulation?

IMPORTANT: Respond with ONLY valid JSON. Format:
{
    "score": 0.0-1.0,
    "reasoning_trace": "Step-by-step reasoning about the match",
    "color_match": true/false/null,
    "shape_match": true/false/null,
    "position_valid": true/false,
    "is_best_candidate": true/false,
    "confidence": 0.0-1.0
}"""

        user_prompt = f"Command: {command}\\nCandidate object: {obj_desc}"
        if reasoning_context:
            user_prompt += f"\\n\\nContext: {json.dumps(reasoning_context)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.chat(messages, **kwargs)
        
        # Parse JSON
        content = response['content']
        try:
            # Extract JSON from response
            json_content = content
            if '```json' in content:
                json_content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                json_content = content.split('```')[1].split('```')[0].strip()
            else:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_content = content[start_idx:end_idx]

            result = json.loads(json_content)
            return result
        except Exception as e:
            return {
                "score": 0.5,
                "reasoning_trace": f"Failed to parse scoring response: {str(e)}",
                "confidence": 0.0,
                "error": str(e),
                "raw_response": content
            }


def create_ollama_client(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    **kwargs
) -> OllamaClient:
    """Factory function to create Ollama client."""
    return OllamaClient(model=model, base_url=base_url, **kwargs)


class OllamaReasoner:
    """
    Command reasoner wrapper for Ollama client.
    
    DEPRECATED: Use LLMCommandReasoner with OllamaClient directly instead.
    This class is kept for backward compatibility.

    Provides the same interface as LLMCommandReasoner but uses Ollama.
    """

    def __init__(self, ollama_client: OllamaClient, fallback_to_rule_based: bool = True):
        """
        Initialize Ollama reasoner.

        Args:
            ollama_client: Configured Ollama client
            fallback_to_rule_based: Use rule-based parser when Ollama fails
        """
        # Import here to avoid circular dependency
        from .command_reasoner_llm import LLMCommandReasoner
        self.llm_reasoner = LLMCommandReasoner(
            llm_client=ollama_client,
            fallback_to_rule_based=fallback_to_rule_based,
        )
        self.fallback_to_rule_based = fallback_to_rule_based

    def reason_about_command(
        self,
        command: str,
        scene_context: dict = None,
        include_internal_monologue: bool = True,
    ) -> dict:
        """
        Reason about a command using Ollama.

        Args:
            command: Natural language command
            scene_context: Optional scene description
            include_internal_monologue: Include reasoning trace

        Returns:
            Dictionary with reasoning results
        """
        return self.llm_reasoner.reason_about_command(
            command=command,
            scene_context=scene_context,
            include_internal_monologue=include_internal_monologue,
        )

    def score_object_with_reasoning(
        self,
        command: str,
        object_label: str,
        object_attributes: dict = None,
        reasoning_context: dict = None,
    ) -> dict:
        """
        Score an object using Ollama.

        Args:
            command: Natural language command
            object_label: Object label
            object_attributes: Optional object attributes
            reasoning_context: Optional reasoning context

        Returns:
            Dictionary with score and reasoning
        """
        return self.llm_reasoner.score_object_with_reasoning(
            command=command,
            object_label=object_label,
            object_attributes=object_attributes,
            reasoning_context=reasoning_context,
        )
