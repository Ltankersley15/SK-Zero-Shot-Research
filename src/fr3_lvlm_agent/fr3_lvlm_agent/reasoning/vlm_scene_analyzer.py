"""
VLM Scene Analyzer for robotic manipulation.

Uses GPT-4o-mini vision capabilities to analyze the robot's workspace and extract:
- Object list with attributes (color, shape, size, position)
- Spatial relationships between objects
- Affordances (graspable, pourable, etc.)
- Scene summary for task planning
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

from .openai_client import OpenAIClient


@dataclass
class ObjectDescription:
    """Description of a detected object in the scene."""

    object_id: str
    label: str
    color: Optional[str]
    shape: Optional[str]
    size_category: str  # "small", "medium", "large"
    position_description: str  # "left side", "center", "near red block"
    graspable: bool
    confidence: float
    bounding_box_description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpatialRelationship:
    """Spatial relationship between two objects."""

    object_a: str
    object_b: str
    relationship: str  # "left of", "right of", "behind", "in front of", "on top of"
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneAnalysis:
    """Complete analysis of a scene."""

    objects: List[ObjectDescription]
    relationships: List[SpatialRelationship]
    scene_summary: str
    reasoning_trace: str
    image_dimensions: tuple
    analysis_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "scene_summary": self.scene_summary,
            "reasoning_trace": self.reasoning_trace,
            "image_dimensions": self.image_dimensions,
            "analysis_metadata": self.analysis_metadata,
        }


class VLMSceneAnalyzer:
    """
    Analyze robot workspace scenes using GPT-4o-mini vision.
    
    This analyzer provides rich scene understanding beyond simple segmentation:
    - Natural language object descriptions
    - Spatial reasoning
    - Affordance detection
    - Context-aware interpretation
    """
    
    SYSTEM_PROMPT = """You are a vision assistant for a robotic manipulation system.
The robot has a Franka Emika FR3 arm with a two-finger gripper and an eye-in-hand camera.

Your task is to analyze images of the robot's workspace and identify objects that could be manipulated.

For each object, provide:
- A unique ID (e.g., "object_1", "object_2")
- A label (e.g., "red cube", "blue cylinder", "yellow block")
- Color (if visible)
- Shape (cube, cylinder, sphere, irregular, etc.)
- Size category: small (<5cm), medium (5-10cm), large (>10cm)
- Position description relative to the image (left/center/right, top/middle/bottom)
- Whether it appears graspable with a parallel jaw gripper
- Confidence in your detection (0.0-1.0)

Also identify spatial relationships between objects (e.g., "red cube is left of blue cylinder").

Respond in valid JSON format."""

    ANALYSIS_PROMPT = """Analyze this image from a robot's eye-in-hand camera. The robot needs to manipulate objects in the scene.

Provide a complete scene analysis in JSON format with this structure:
{
    "objects": [
        {
            "object_id": "object_1",
            "label": "red cube",
            "color": "red",
            "shape": "cube",
            "size_category": "small",
            "position_description": "center-left of image",
            "graspable": true,
            "confidence": 0.95,
            "bounding_box_description": "visible cube with clear edges"
        }
    ],
    "relationships": [
        {
            "object_a": "object_1",
            "object_b": "object_2",
            "relationship": "left of",
            "confidence": 0.9
        }
    ],
    "scene_summary": "Brief summary of the scene layout",
    "reasoning_trace": "Step-by-step analysis of what you see"
}

Focus on objects that the robot could potentially manipulate. Ignore background elements unless they're relevant to the task."""

    def __init__(self, openai_client: OpenAIClient):
        """
        Initialize VLM scene analyzer.
        
        Args:
            openai_client: Configured OpenAI client instance.
        """
        self.client = openai_client
    
    def analyze_scene(
        self,
        image: Any,
        custom_prompt: Optional[str] = None,
        include_reasoning: bool = True,
    ) -> SceneAnalysis:
        """
        Analyze a scene image.
        
        Args:
            image: PIL Image, numpy array, or image path.
            custom_prompt: Optional override for the analysis prompt.
            include_reasoning: Whether to include reasoning trace.
            
        Returns:
            SceneAnalysis with detected objects and relationships.
        """
        prompt = custom_prompt or self.ANALYSIS_PROMPT
        
        if not include_reasoning:
            prompt = prompt.replace(',\n    "reasoning_trace": "Step-by-step analysis of what you see"', "")
        
        response_text = self.client.analyze_image(
            image=image,
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=2048,
        )
        
        # Parse JSON response
        try:
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError) as e:
            # Fallback: create minimal analysis
            data = {
                "objects": [],
                "relationships": [],
                "scene_summary": f"Failed to parse VLM response: {response_text[:200]}",
                "reasoning_trace": f"Parse error: {str(e)}",
            }
        
        # Convert to structured objects
        objects = []
        for obj_data in data.get("objects", []):
            try:
                obj = ObjectDescription(
                    object_id=obj_data.get("object_id", f"object_{len(objects)}"),
                    label=obj_data.get("label", "unknown object"),
                    color=obj_data.get("color"),
                    shape=obj_data.get("shape"),
                    size_category=obj_data.get("size_category", "medium"),
                    position_description=obj_data.get("position_description", "unknown"),
                    graspable=obj_data.get("graspable", True),
                    confidence=float(obj_data.get("confidence", 0.5)),
                    bounding_box_description=obj_data.get("bounding_box_description"),
                )
                objects.append(obj)
            except (KeyError, ValueError, TypeError):
                # Skip malformed object entries
                continue
        
        relationships = []
        for rel_data in data.get("relationships", []):
            try:
                rel = SpatialRelationship(
                    object_a=rel_data.get("object_a", ""),
                    object_b=rel_data.get("object_b", ""),
                    relationship=rel_data.get("relationship", ""),
                    confidence=float(rel_data.get("confidence", 0.5)),
                )
                relationships.append(rel)
            except (KeyError, ValueError, TypeError):
                continue
        
        # Get image dimensions if possible
        image_dims = (0, 0)
        try:
            if isinstance(image, np.ndarray):
                image_dims = (image.shape[1], image.shape[0])  # width, height
            elif isinstance(image, Image.Image):
                image_dims = image.size  # width, height
        except Exception:
            pass
        
        return SceneAnalysis(
            objects=objects,
            relationships=relationships,
            scene_summary=data.get("scene_summary", "No summary available"),
            reasoning_trace=data.get("reasoning_trace", "") if include_reasoning else "",
            image_dimensions=image_dims,
            analysis_metadata={
                "model": self.client.model,
                "raw_response": response_text[:500] if len(response_text) > 500 else response_text,
            },
        )
    
    def find_target_object(
        self,
        scene_analysis: SceneAnalysis,
        target_description: str,
    ) -> Optional[ObjectDescription]:
        """
        Find the best matching object for a target description.
        
        Args:
            scene_analysis: Previously analyzed scene.
            target_description: Natural language description of target.
            
        Returns:
            Best matching ObjectDescription or None.
        """
        if not scene_analysis.objects:
            return None
        
        # Use LLM to score each object
        best_object = None
        best_score = 0.0
        
        for obj in scene_analysis.objects:
            obj_desc = f"{obj.label} ({obj.color} {obj.shape}, {obj.size_category}, {obj.position_description})"
            
            score_result = self.client.score_object_match(
                command=target_description,
                object_description=obj_desc,
            )
            
            score = score_result.get("score", 0.0)
            if score > best_score:
                best_score = score
                best_object = obj
        
        # Only return if confidence is reasonable
        if best_score >= 0.6:
            return best_object
        
        return None
    
    def describe_scene_for_reasoning(self, scene_analysis: SceneAnalysis) -> str:
        """
        Create a natural language description of the scene for LLM reasoning.
        
        Args:
            scene_analysis: Analyzed scene.
            
        Returns:
            Natural language scene description.
        """
        if not scene_analysis.objects:
            return "No objects detected in the scene."
        
        lines = ["Scene description:"]
        
        for obj in scene_analysis.objects:
            desc = f"- {obj.label}"
            if obj.color:
                desc += f" (color: {obj.color})"
            if obj.shape:
                desc += f" (shape: {obj.shape})"
            desc += f" - located {obj.position_description}"
            if not obj.graspable:
                desc += " (not easily graspable)"
            lines.append(desc)
        
        if scene_analysis.relationships:
            lines.append("\nSpatial relationships:")
            for rel in scene_analysis.relationships:
                lines.append(f"- {rel.object_a} is {rel.relationship} {rel.object_b}")
        
        lines.append(f"\nSummary: {scene_analysis.scene_summary}")
        
        return "\n".join(lines)


def create_scene_analyzer(api_key: Optional[str] = None) -> Optional[VLMSceneAnalyzer]:
    """
    Factory function to create VLM scene analyzer.
    
    Returns None if OpenAI client cannot be created.
    """
    from .openai_client import create_openai_client
    
    client = create_openai_client(api_key=api_key)
    if client is None:
        return None
    
    return VLMSceneAnalyzer(client)
