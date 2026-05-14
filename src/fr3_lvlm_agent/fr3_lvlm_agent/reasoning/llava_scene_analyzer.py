"""
LLaVA Scene Analyzer for robotic manipulation.

Uses LLaVA (Large Language and Vision Assistant) for open-vocabulary scene understanding.
Unlike GPT-4o, LLaVA can run locally and provides similar vision-language capabilities.

This module:
1. Takes RGB images from the robot's camera
2. Uses LLaVA to generate scene descriptions
3. Extracts object locations and attributes
4. Provides structured output for task planning
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

try:
    import torch
    from transformers import LlavaProcessor, LlavaForConditionalGeneration
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    LlavaProcessor = None
    LlavaForConditionalGeneration = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class LLaVAObjectDescription:
    """Object description from LLaVA analysis."""

    label: str
    color: Optional[str] = None
    shape: Optional[str] = None
    position_description: str = ""
    confidence: float = 0.5
    pixel_coordinates: Optional[Tuple[int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLaVASceneAnalysis:
    """Complete scene analysis from LLaVA."""

    objects: List[LLaVAObjectDescription]
    scene_description: str
    reasoning_trace: str
    image_dimensions: Optional[Tuple[int, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "scene_description": self.scene_description,
            "reasoning_trace": self.reasoning_trace,
            "image_dimensions": self.image_dimensions,
        }


class LLaVASceneAnalyzer:
    """
    Analyze robot workspace scenes using LLaVA.
    
    LLaVA provides open-vocabulary scene understanding:
    - Can identify novel objects without training
    - Provides natural language descriptions
    - Can answer questions about the scene
    - Runs locally (no API costs)
    """
    
    DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
    
    # Prompt templates for different analysis tasks
    SCENE_ANALYSIS_PROMPT = """Analyze this image from a robot's camera. The robot needs to manipulate objects on a table.

Identify all visible objects on the table surface. For each object, provide:
1. Object type (e.g., "cube", "cylinder", "block")
2. Color (e.g., "red", "blue", "green")
3. Approximate position (e.g., "left side", "center", "right side")
4. Any notable features

Format your response as a JSON list of objects with fields: label, color, position, description.

Example format:
[
    {{"label": "cube", "color": "red", "position": "left side", "description": "red cube on left"}},
    {{"label": "cylinder", "color": "blue", "position": "right side", "description": "blue cylinder on right"}}
]

Respond ONLY with the JSON list, no other text."""

    OBJECT_LOCALIZATION_PROMPT = """In this image, locate the {target_object}. 

Describe:
1. Where it is in the image (left/center/right, top/middle/bottom)
2. What it's near or next to
3. Its approximate pixel coordinates if possible

Respond in JSON format: {{"position": "...", "near": "...", "coordinates": [x, y]}}"""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        load_on_init: bool = False,
    ):
        """
        Initialize LLaVA scene analyzer.
        
        Args:
            model_name: HuggingFace model name for LLaVA
            device: Device to run model on ("cuda", "cpu", "auto")
            load_on_init: Whether to load model during initialization
        """
        if not LLAVA_AVAILABLE:
            raise ImportError(
                "LLaVA requires transformers and torch. "
                "Install with: pip install transformers torch pillow"
            )
        
        self.model_name = model_name
        self.device = self._select_device(device)
        self.processor = None
        self.model = None
        self._loaded = False
        
        if load_on_init:
            self.load_model()
    
    def _select_device(self, device: str) -> str:
        """Select best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device
    
    def load_model(self) -> bool:
        """Load LLaVA model and processor."""
        if self._loaded:
            return True
        
        try:
            print(f"[LLaVA] Loading model: {self.model_name} on {self.device}")
            
            self.processor = LlavaProcessor.from_pretrained(self.model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            print("[LLaVA] Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"[LLaVA] Failed to load model: {e}")
            return False
    
    def analyze_scene(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
    ) -> Optional[LLaVASceneAnalysis]:
        """
        Analyze a scene image using LLaVA.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            prompt: Custom prompt (uses default scene analysis if None)
        
        Returns:
            LLaVASceneAnalysis with detected objects and descriptions
        """
        if not self._loaded and not self.load_model():
            return None
        
        if not PIL_AVAILABLE:
            print("[LLaVA] PIL not available")
            return None
        
        # Convert numpy to PIL
        if image.ndim == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            print(f"[LLaVA] Invalid image shape: {image.shape}")
            return None
        
        # Use default prompt if none provided
        if prompt is None:
            prompt = self.SCENE_ANALYSIS_PROMPT
        
        # Run inference
        try:
            response = self._run_inference(pil_image, prompt)
            objects = self._parse_objects_from_response(response)
            
            return LLaVASceneAnalysis(
                objects=objects,
                scene_description=response[:500],
                reasoning_trace="",
                image_dimensions=(image.shape[1], image.shape[0]),
            )
            
        except Exception as e:
            print(f"[LLaVA] Analysis failed: {e}")
            return None
    
    def _run_inference(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 500,
    ) -> str:
        """Run LLaVA inference."""
        if not self._loaded:
            return ""
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        input_ids = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        
        inputs = self.processor(
            images=image,
            text=self.processor.tokenizer.decode(input_ids),
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        # Decode response
        response = self.processor.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def _parse_objects_from_response(
        self,
        response: str,
    ) -> List[LLaVAObjectDescription]:
        """Parse LLaVA response into structured objects."""
        objects = []

        # Try to extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            obj = LLaVAObjectDescription(
                                label=item.get("label", "object"),
                                color=item.get("color"),
                                shape=item.get("shape"),
                                position_description=item.get("position", ""),
                                confidence=0.8,
                            )
                            objects.append(obj)
            except json.JSONDecodeError:
                pass

        # Fallback: parse natural language response
        if not objects:
            objects = self._parse_natural_language_response(response)

        return objects

    def _parse_natural_language_response(self, response: str) -> List[LLaVAObjectDescription]:
        """Parse natural language LLaVA response to extract objects."""
        objects = []
        
        # Common color patterns
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'cyan', 'white', 'black']
        
        # Common shape/object patterns
        shapes = ['cube', 'block', 'cylinder', 'sphere', 'box', 'cone']
        
        response_lower = response.lower()
        
        # Extract color-object pairs
        for color in colors:
            if color in response_lower:
                # Find what object follows the color
                for shape in shapes:
                    if shape in response_lower:
                        # Check if they appear close together
                        color_idx = response_lower.find(color)
                        shape_idx = response_lower.find(shape)
                        if abs(color_idx - shape_idx) < 30:  # Within 30 characters
                            obj = LLaVAObjectDescription(
                                label=shape,
                                color=color,
                                shape=shape,
                                position_description="Detected in scene",
                                confidence=0.7,
                            )
                            # Avoid duplicates
                            if not any(o.label == shape and o.color == color for o in objects):
                                objects.append(obj)
        
        # If no structured objects found, create generic ones from the description
        if not objects:
            # Try to extract object mentions
            words = response.split()
            for i, word in enumerate(words):
                if word.lower() in ['object', 'item', 'thing']:
                    color = None
                    if i > 0 and words[i-1].lower() in colors:
                        color = words[i-1].lower()
                    obj = LLaVAObjectDescription(
                        label="object",
                        color=color,
                        position_description=response[:200],
                        confidence=0.5,
                    )
                    objects.append(obj)
                    break
            
            # Last resort: single object with full description
            if not objects:
                objects.append(LLaVAObjectDescription(
                    label="scene_object",
                    position_description=response[:200],
                    confidence=0.5,
                ))
        
        return objects

    def localize_object_3d(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        target_object: str,
        camera_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Localize a specific object in 3D using LLaVA + depth.
        
        Args:
            color_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W) in meters
            target_object: Object description (e.g., "red cube")
            camera_info: Camera intrinsics {fx, fy, cx, cy}
            
        Returns:
            Dict with 3D position and metadata, or None if not found
        """
        # First, use LLaVA to identify the object location in 2D
        pil_image = Image.fromarray(color_image)
        
        prompt = self.OBJECT_LOCALIZATION_PROMPT.format(target_object=target_object)
        response = self._run_inference(pil_image, prompt, max_new_tokens=200)
        
        # Parse response for pixel coordinates
        coords = self._parse_coordinates_from_response(response, color_image.shape)
        
        if coords is None:
            return None
        
        pixel_u, pixel_v = coords
        
        # Get depth at that pixel
        if 0 <= pixel_v < depth_image.shape[0] and 0 <= pixel_u < depth_image.shape[1]:
            depth = depth_image[pixel_v, pixel_u]
        else:
            return None
        
        if depth <= 0 or depth > 10:  # Invalid depth
            return None
        
        # Convert to 3D camera coordinates
        if camera_info:
            fx = camera_info.get('fx', 634.0)
            fy = camera_info.get('fy', 634.0)
            cx = camera_info.get('cx', 640.0)
            cy = camera_info.get('cy', 360.0)
        else:
            # D455 defaults at 1280x720
            fx = fy = 634.0
            cx, cy = 640.0, 360.0
        
        # Pinhole camera model
        X = (pixel_u - cx) * depth / fx
        Y = (pixel_v - cy) * depth / fy
        Z = depth
        
        return {
            'object': target_object,
            'pixel_coords': (pixel_u, pixel_v),
            'camera_frame_position': [X, Y, Z],
            'depth': depth,
            'confidence': 0.7,
            'llava_response': response[:200],
        }

    def _parse_coordinates_from_response(
        self,
        response: str,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int]]:
        """Parse pixel coordinates from LLaVA response."""
        # Try to find coordinates in format (x, y) or [x, y]
        import re
        
        # Pattern for coordinates
        patterns = [
            r'\((\d+),\s*(\d+)\)',
            r'\[(\d+),\s*(\d+)\]',
            r'coordinates?\s*[:=]?\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?',
            r'at\s+\(?\s*(\d+)\s*,\s*(\d+)\s*\)?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                u = int(match.group(1))
                v = int(match.group(2))
                
                # Validate coordinates are within image bounds
                h, w = image_shape[:2]
                if 0 <= u < w and 0 <= v < h:
                    return (u, v)
        
        # Fallback: return image center
        h, w = image_shape[:2]
        return (w // 2, h // 2)
    
    def locate_object(
        self,
        image: np.ndarray,
        target: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Locate a specific object in the scene.
        
        Args:
            image: RGB image
            target: Object description (e.g., "red cube")
        
        Returns:
            Dictionary with position information
        """
        prompt = self.OBJECT_LOCALIZATION_PROMPT.format(target_object=target)
        
        if not self._loaded and not self.load_model():
            return None
        
        if not PIL_AVAILABLE:
            return None
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        try:
            response = self._run_inference(pil_image, prompt)
            
            # Try to parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"description": response}
            
        except Exception as e:
            print(f"[LLaVA] Localization failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if LLaVA is available and loaded."""
        return LLAVA_AVAILABLE and self._loaded


def create_llava_analyzer(
    model_name: str = LLaVASceneAnalyzer.DEFAULT_MODEL,
    device: str = "auto",
) -> Optional[LLaVASceneAnalyzer]:
    """
    Create LLaVA scene analyzer with error handling.
    
    Returns None if LLaVA is not available.
    """
    if not LLAVA_AVAILABLE:
        print("[LLaVA] Not available - install transformers and torch")
        return None
    
    try:
        return LLaVASceneAnalyzer(model_name=model_name, device=device)
    except Exception as e:
        print(f"[LLaVA] Failed to create analyzer: {e}")
        return None
