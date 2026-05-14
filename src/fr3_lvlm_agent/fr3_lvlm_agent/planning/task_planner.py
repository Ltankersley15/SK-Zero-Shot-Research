"""
LLM-Driven Task Planner for FR3 Robot.

This module provides high-level task planning capabilities using LLM reasoning.
It decomposes complex commands (e.g., "stack the blue cube on the red cube") into
executable sub-tasks and handles replanning when actions fail.

Features:
- Multi-step task decomposition
- Adaptive grasp strategy selection
- Failure recovery with alternative approaches
- Non-predefined pose reasoning
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """Supported task types."""

    PICK = "pick"
    PLACE = "place"
    STACK = "stack"
    MOVE = "move"
    REGRASP = "regrasp"
    INSPECT = "inspect"
    REARRANGE = "rearrange"
    UNKNOWN = "unknown"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


@dataclass
class GraspStrategy:
    """Grasp approach strategy."""

    approach_vector: str  # "top", "side_front", "side_left", "side_right"
    approach_distance: float  # meters above grasp point
    gripper_orientation: str  # "parallel", "perpendicular", "aligned_with_bbox"
    gripper_width: float  # meters (0.00-0.08)
    descend_speed: str  # "slow", "normal", "fast"
    reason: str = ""


@dataclass
class SubTask:
    """A single sub-task in the plan."""

    task_type: TaskType
    description: str
    target_object: Optional[str] = None
    target_location: Optional[Dict[str, float]] = None  # {x, y, z}
    action: Optional[str] = None
    action_params: Dict[str, Any] = field(default_factory=dict)
    grasp_strategies: Optional[List[GraspStrategy]] = None
    preconditions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    result: Optional[str] = None


@dataclass
class TaskPlan:
    """Complete task plan with sub-tasks."""

    original_command: str
    task_type: TaskType
    sub_tasks: List[SubTask] = field(default_factory=list)
    current_step: int = 0
    status: TaskStatus = TaskStatus.PENDING
    llm_reasoning: Optional[str] = None
    recovery_history: List[Dict[str, Any]] = field(default_factory=list)

    def get_current_task(self) -> Optional[SubTask]:
        """Get the current sub-task to execute."""
        if 0 <= self.current_step < len(self.sub_tasks):
            return self.sub_tasks[self.current_step]
        return None

    def get_next_task(self) -> Optional[SubTask]:
        """Advance to next sub-task and return it."""
        self.current_step += 1
        return self.get_current_task()

    def mark_current_complete(self):
        """Mark current sub-task as completed."""
        if self.current_step < len(self.sub_tasks):
            self.sub_tasks[self.current_step].status = TaskStatus.COMPLETED
            self.current_step += 1

    def mark_current_failed(self, reason: str):
        """Mark current sub-task as failed."""
        if self.current_step < len(self.sub_tasks):
            task = self.sub_tasks[self.current_step]
            task.status = TaskStatus.FAILED
            task.attempts += 1
            task.result = reason
            self.recovery_history.append({
                "step": self.current_step,
                "task": task.description,
                "attempt": task.attempts,
                "failure_reason": reason,
            })

    def is_complete(self) -> bool:
        """Check if all sub-tasks are completed."""
        return self.current_step >= len(self.sub_tasks)

    def can_retry_current(self) -> bool:
        """Check if current task can be retried."""
        if self.current_step < len(self.sub_tasks):
            return self.sub_tasks[self.current_step].attempts < self.sub_tasks[self.current_step].max_attempts
        return False


class LLMTaskPlanner:
    """
    LLM-driven task planner for FR3 robot.
    
    Uses LLM to:
    1. Decompose high-level commands into sub-tasks
    2. Select appropriate grasp strategies
    3. Generate recovery plans on failure
    4. Reason about non-predefined poses and approaches
    """

    SYSTEM_PROMPT = """You are a robotic manipulation reasoning engine for a Franka Emika FR3 arm.

Robot capabilities:
- 7-DOF manipulator with parallel-jaw gripper (max opening: 8cm)
- Wrist-mounted Realsense D455 RGB-D camera
- Workspace: x=[0.3-0.7m], y=[-0.4-0.4m], z=[0.0-0.5m] from base frame
- Can detect open-vocabulary tabletop objects and estimate Cartesian positions from RGB-D
- Camera can zoom out and look around to find objects
- Executes only pick, place, move, and inspect tasks plus a small set of safe action primitives

Your role:
1. Analyze manipulation commands through step-by-step reasoning
2. Infer object properties from context (don't assume fixed taxonomy)
3. Determine optimal grasp strategies based on object geometry and scene context
4. Generate executable sub-tasks with clear success criteria
5. Reason about failure modes and recovery strategies

CRITICAL - Search Behavior:
- If target object is NOT detected in initial view, DO NOT attempt to pick
- Instead: Command the robot to zoom out camera and scan the workspace
- Look around systematically: check left, right, center of workspace
- Once object is found, proceed with pick
- If object still not found after search, report failure to user

Think carefully about:
- Object occlusion and visibility - if not visible, search for it first
- Grasp stability vs. task requirements
- Collision avoidance with table and other objects
- When to re-observe vs. proceed with uncertainty
- Real embodiment limits: no force control, no tactile servoing, no fluid dynamics model

Respond with structured JSON that can be directly executed."""

    _SPIN_MARKERS = ("spin", "circle", "rotate wrist", "rotate in place", "turn around")
    _WAVE_MARKERS = ("wave", "hello", "greet")
    _READY_MARKERS = ("ready", "home", "rest pose")
    _PURE_MOVE_MARKERS = ("move", "go", "shift", "slide", "step")
    _LOCATION_MARKERS = (
        "where is",
        "where's",
        "coordinates",
        "coordinate",
        "location",
        "position",
    )
    _SURVEY_MARKERS = (
        "survey",
        "scan",
        "look around",
        "look around the area",
        "what do you see",
        "describe the scene",
        "inspect the area",
        "inspect the scene",
    )
    _MANIPULATION_MARKERS = (
        "pick",
        "pickup",
        "grab",
        "grasp",
        "stack",
        "place",
        "put",
        "cube",
        "block",
        "cylinder",
        "object",
        "bottle",
        "can",
        "mug",
    )
    _GENERIC_TARGET_TERMS = {"target_object", "source_object", "object", "item", "thing"}
    _DIRECT_PICK_MARKERS = ("pick", "pick up", "pickup", "grab", "grasp")
    _DIRECT_PICK_EXCLUSION_MARKERS = (
        "place",
        "put",
        "stack",
        "then",
        "after",
        "before",
        " and ",
        "next to",
        "left of",
        "right of",
        "on top of",
        "onto ",
        "move ",
        "survey",
        "inspect",
        "scan",
    )

    def __init__(self, llm_reasoner=None, *, apply_intent_safety: bool = False, enforce_pick_guards: bool = False):
        """
        Initialize task planner.

        Args:
            llm_reasoner: LLM reasoner for planning (OllamaReasoner or LLMCommandReasoner)
        """
        self.llm_reasoner = llm_reasoner
        self._last_plan: Optional[TaskPlan] = None
        self._scene_context: Optional[Dict[str, Any]] = None
        self._logger = None
        self.apply_intent_safety = bool(apply_intent_safety)
        self.enforce_pick_guards = bool(enforce_pick_guards)
    
    def set_logger(self, logger):
        """Set ROS 2 logger for task planner."""
        self._logger = logger
    
    def get_logger(self):
        """Get logger, create fallback if none set."""
        if self._logger is None:
            import logging
            return logging.getLogger("LLMTaskPlanner")
        return self._logger

    def set_scene_context(self, context: Dict[str, Any]):
        """Set current scene context for planning."""
        self._scene_context = context

    def _is_pure_directional_motion_command(self, command: str) -> bool:
        cmd_lower = str(command or "").lower()
        has_left = re.search(r"\bleft\b", cmd_lower) is not None
        has_right = re.search(r"\bright\b", cmd_lower) is not None
        if not (has_left or has_right):
            return False
        if not any(tok in cmd_lower for tok in self._PURE_MOVE_MARKERS):
            return False
        if any(tok in cmd_lower for tok in self._MANIPULATION_MARKERS):
            return False
        return True

    def _action_from_command(self, command: str) -> Optional[tuple[str, str, Dict[str, Any]]]:
        cmd_lower = str(command or "").lower()
        if any(marker in cmd_lower for marker in self._LOCATION_MARKERS):
            return ("survey_scene", "Survey the area and report visible objects", {"description": command})
        if any(marker in cmd_lower for marker in self._SURVEY_MARKERS):
            return ("survey_scene", "Survey the area and report visible objects", {"description": command})
        if any(marker in cmd_lower for marker in self._SPIN_MARKERS):
            return ("spin_in_place", "Spin the arm in place", {"turns": 1})
        if any(marker in cmd_lower for marker in self._WAVE_MARKERS):
            return ("wave", "Wave gesture", {"cycles": 2})
        if any(marker in cmd_lower for marker in self._READY_MARKERS):
            return ("go_ready_pose", "Move to ready pose", {})
        if "open" in cmd_lower and "gripper" in cmd_lower:
            return ("open_gripper", "Open gripper", {})
        if "close" in cmd_lower and "gripper" in cmd_lower:
            return ("close_gripper", "Close gripper", {})
        if self._is_pure_directional_motion_command(command):
            direction = "left" if re.search(r"\bleft\b", cmd_lower) else "right"
            return (
                "shift_lateral",
                f"Shift arm {direction}",
                {"direction": direction, "distance_m": 0.12},
            )
        return None

    def _clean_object_phrase(self, phrase: str) -> str:
        text = str(phrase or "").strip().lower()
        text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^(the|a|an)\s+", "", text)
        if not text:
            return "target object"
        if text in self._GENERIC_TARGET_TERMS:
            return "target object"
        return text

    def _extract_pick_target_from_command(self, command: str) -> str:
        cmd = str(command or "").strip().lower()
        patterns = [
            r"(?:pick\s*up|pickup|pick|grab|grasp)\s+(?:the|a|an)?\s*(.+)$",
            r"(?:lift|take)\s+(?:the|a|an)?\s*(.+)$",
        ]
        for pat in patterns:
            m = re.search(pat, cmd)
            if m:
                phrase = self._clean_object_phrase(m.group(1))
                return phrase if phrase != "target object" else phrase
        return "target object"

    def _extract_stack_targets(self, command: str) -> tuple[str, str]:
        cmd = str(command or "").strip().lower()
        m = re.search(r"stack\s+(.+?)\s+on(?:\s+top\s+of)?\s+(.+)$", cmd)
        if m:
            source = self._clean_object_phrase(m.group(1))
            target = self._clean_object_phrase(m.group(2))
            return source, target
        return "source object", "target object"

    def _should_bypass_llm_for_direct_pick(self, command: str) -> bool:
        """Route simple single-object pick commands to deterministic rule-based planning."""
        cmd = f" {str(command or '').strip().lower()} "
        if not cmd.strip():
            return False
        if not any(marker in cmd for marker in self._DIRECT_PICK_MARKERS):
            return False
        if any(marker in cmd for marker in self._DIRECT_PICK_EXCLUSION_MARKERS):
            return False
        return True

    def _infer_sub_task_type(self, sub_task: Dict[str, Any]) -> TaskType:
        action_name = str(sub_task.get("action", "") or "").strip().lower()
        if action_name:
            if action_name == "survey_scene":
                return TaskType.INSPECT
            return TaskType.UNKNOWN

        text = (
            f"{sub_task.get('description', '')} "
            f"{sub_task.get('target_object', '')} "
            f"{sub_task.get('task_type', '')}"
        ).lower()
        if any(tok in text for tok in ("pick", "grab", "grasp", "lift")):
            return TaskType.PICK
        if any(tok in text for tok in ("place", "put", "drop", "set down")):
            return TaskType.PLACE
        if any(tok in text for tok in ("inspect", "survey", "look", "scan")):
            return TaskType.INSPECT
        return TaskType.MOVE

    def _normalize_sub_task_type(self, parsed: Optional[TaskType], sub_task: Dict[str, Any]) -> TaskType:
        if parsed is not None and parsed != TaskType.UNKNOWN:
            return parsed
        return self._infer_sub_task_type(sub_task)

    def _plan_has_grasp_steps(self, plan: TaskPlan) -> bool:
        for st in plan.sub_tasks:
            if st.task_type in {TaskType.PICK, TaskType.PLACE, TaskType.STACK, TaskType.REGRASP, TaskType.REARRANGE}:
                return True
            if st.target_object:
                return True
        return False

    def _sanitize_plan_for_intent(self, command: str, plan: TaskPlan) -> TaskPlan:
        action_intent = self._action_from_command(command)
        if action_intent is None:
            return plan

        action_name, action_description, action_params = action_intent
        if len(plan.sub_tasks) == 0:
            action_plan = self._create_action_plan(
                command=command,
                action=action_name,
                description=action_description,
                action_params=action_params,
            )
            action_plan.llm_reasoning = (
                f"{plan.llm_reasoning or ''} "
                f"Sanitized to action '{action_name}' due to non-grasp intent."
            ).strip()
            return action_plan

        if self._plan_has_grasp_steps(plan):
            action_plan = self._create_action_plan(
                command=command,
                action=action_name,
                description=action_description,
                action_params=action_params,
            )
            action_plan.llm_reasoning = (
                f"{plan.llm_reasoning or ''} "
                f"Overrode LLM grasping sub-tasks with action '{action_name}' for command intent alignment."
            ).strip()
            return action_plan

        first = plan.sub_tasks[0]
        if first.task_type == TaskType.MOVE and not first.action:
            first.action = action_name
            first.action_params = dict(action_params)
            first.description = first.description or action_description
            plan.llm_reasoning = (
                f"{plan.llm_reasoning or ''} "
                f"Attached action primitive '{action_name}' to first move sub-task."
            ).strip()
        return plan

    def create_plan(self, command: str) -> TaskPlan:
        """
        Create a task plan from a natural language command.
        
        Args:
            command: Natural language command (e.g., "stack the blue cube on the red cube")
            
        Returns:
            TaskPlan with decomposed sub-tasks
        """
        command_str = str(command or "").strip()
        if not command_str:
            return self._create_rejected_plan(command_str, "empty_command")

        if self._should_bypass_llm_for_direct_pick(command_str):
            direct_pick = self._create_plan_rule_based(command_str)
            direct_pick = self._ensure_primary_intent_steps(command_str, direct_pick)
            if self.enforce_pick_guards:
                direct_pick = self._ensure_pick_execution_guards(direct_pick)
            if direct_pick.llm_reasoning:
                direct_pick.llm_reasoning = (
                    f"{direct_pick.llm_reasoning} Bypassed LLM planning and planner-level survey insertion for a direct pick command."
                ).strip()
            else:
                direct_pick.llm_reasoning = (
                    "Bypassed LLM planning and planner-level survey insertion for a direct pick command."
                )
            return direct_pick

        # Use LLM if available
        if self.llm_reasoner is not None:
            llm_plan = self._create_plan_with_llm(command_str)
            if self.apply_intent_safety:
                llm_plan = self._sanitize_plan_for_intent(command_str, llm_plan)
            if len(llm_plan.sub_tasks) == 0:
                fallback = self._create_plan_rule_based(command_str)
                fallback = self._ensure_primary_intent_steps(command_str, fallback)
                fallback = self._ensure_search_before_unseen_pick(fallback)
                fallback = self._prune_redundant_pick_surveys(fallback)
                if self.enforce_pick_guards:
                    fallback = self._ensure_pick_execution_guards(fallback)
                return fallback
            llm_plan = self._ensure_primary_intent_steps(command_str, llm_plan)
            llm_plan = self._ensure_search_before_unseen_pick(llm_plan)
            llm_plan = self._prune_redundant_pick_surveys(llm_plan)
            if self.enforce_pick_guards:
                llm_plan = self._ensure_pick_execution_guards(llm_plan)
            return llm_plan

        # Fallback to rule-based planning
        fallback = self._create_plan_rule_based(command_str)
        fallback = self._ensure_primary_intent_steps(command_str, fallback)
        fallback = self._ensure_search_before_unseen_pick(fallback)
        fallback = self._prune_redundant_pick_surveys(fallback)
        if self.enforce_pick_guards:
            fallback = self._ensure_pick_execution_guards(fallback)
        return fallback

    def _ensure_pick_execution_guards(self, plan: TaskPlan) -> TaskPlan:
        """Ensure pick plans explicitly open gripper and request post-lift verification."""
        first_pick_idx: Optional[int] = None
        for idx, st in enumerate(plan.sub_tasks):
            if st.task_type == TaskType.PICK:
                first_pick_idx = idx
                break
        if first_pick_idx is None:
            return plan

        has_open_before_pick = any(
            (st.task_type == TaskType.MOVE and str(st.action or "").strip().lower() == "open_gripper")
            for st in plan.sub_tasks[:first_pick_idx]
        )
        if not has_open_before_pick:
            plan.sub_tasks.insert(
                first_pick_idx,
                SubTask(
                    task_type=TaskType.MOVE,
                    description="Open gripper before grasp",
                    action="open_gripper",
                    action_params={},
                    preconditions=["robot_ready"],
                    success_criteria=["gripper_opened"],
                ),
            )

        for st in plan.sub_tasks:
            if st.task_type != TaskType.PICK:
                continue
            if "object_attached_after_lift" not in st.success_criteria:
                st.success_criteria.append("object_attached_after_lift")
            if "object_absent_from_original_location" not in st.success_criteria:
                st.success_criteria.append("object_absent_from_original_location")
        return plan

    def _ensure_primary_intent_steps(self, command: str, plan: TaskPlan) -> TaskPlan:
        """Repair plans whose top-level intent omits the executable step that fulfills it."""
        if plan.task_type != TaskType.PICK:
            return plan
        if any(st.task_type == TaskType.PICK for st in plan.sub_tasks):
            return plan

        target_obj = self._extract_pick_target_from_command(command)
        plan.sub_tasks.append(
            SubTask(
                task_type=TaskType.PICK,
                description=f"Pick up the {target_obj}",
                target_object=target_obj,
                grasp_strategies=[self._default_grasp_strategy()],
                preconditions=["object_detected", "gripper_open"],
                success_criteria=["object_grasped", "lifted_successfully"],
            )
        )
        if plan.llm_reasoning:
            plan.llm_reasoning = (
                f"{plan.llm_reasoning} Added an executable pick step because the LLM plan omitted one."
            ).strip()
        return plan

    def _ensure_search_before_unseen_pick(self, plan: TaskPlan) -> TaskPlan:
        """Insert a survey step before grasping when the requested target is not in the visible-object context."""
        visible_objects = []
        if self._scene_context:
            visible_objects = list(self._scene_context.get("scene", {}).get("visible_objects", []) or [])
        if not visible_objects:
            return plan

        visible_terms: set[str] = set()
        for obj in visible_objects:
            label = str(obj.get("label", "") or "").strip().lower()
            color = str(obj.get("color", "") or "").strip().lower()
            shape = str(obj.get("shape", "") or "").strip().lower()
            if label:
                visible_terms.add(label)
                visible_terms.update(re.findall(r"[a-z0-9]+", label))
            if color:
                visible_terms.add(color)
            if shape:
                visible_terms.add(shape)

        for idx, st in enumerate(plan.sub_tasks):
            if st.task_type != TaskType.PICK:
                continue
            target = str(st.target_object or "").strip().lower()
            if not target:
                continue
            target_tokens = {tok for tok in re.findall(r"[a-z0-9]+", target) if tok}
            if target in visible_terms or (target_tokens and target_tokens.issubset(visible_terms)):
                continue
            already_searching = any(
                t.task_type == TaskType.INSPECT or str(t.action or "").strip().lower() == "survey_scene"
                for t in plan.sub_tasks[:idx]
            )
            if already_searching:
                continue
            plan.sub_tasks.insert(
                idx,
                SubTask(
                    task_type=TaskType.INSPECT,
                    description=f"Survey the workspace to locate {target}",
                    action="survey_scene",
                    action_params={"description": f"locate {target} before grasping"},
                    success_criteria=["scene_observed"],
                ),
            )
            if plan.llm_reasoning:
                plan.llm_reasoning = (
                    f"{plan.llm_reasoning} Added a survey step because '{target}' was not present in scene.visible_objects."
                ).strip()
            break
        return plan

    def _prune_redundant_pick_surveys(self, plan: TaskPlan) -> TaskPlan:
        """Drop leading survey steps when the pick target is already visible."""
        if plan.task_type != TaskType.PICK or len(plan.sub_tasks) < 2 or not self._scene_context:
            return plan

        visible_objects = list(self._scene_context.get("scene", {}).get("visible_objects", []) or [])
        if not visible_objects:
            return plan

        visible_terms: set[str] = set()
        for obj in visible_objects:
            label = str(obj.get("label", "") or "").strip().lower()
            color = str(obj.get("color", "") or "").strip().lower()
            shape = str(obj.get("shape", "") or "").strip().lower()
            if label:
                visible_terms.add(label)
                visible_terms.update(re.findall(r"[a-z0-9]+", label))
            if color:
                visible_terms.add(color)
            if shape:
                visible_terms.add(shape)

        pruned = False
        while len(plan.sub_tasks) >= 2:
            first = plan.sub_tasks[0]
            second = plan.sub_tasks[1]
            first_is_survey = first.task_type == TaskType.INSPECT or str(first.action or "").strip().lower() == "survey_scene"
            if (not first_is_survey) or second.task_type != TaskType.PICK:
                break
            target = str(second.target_object or "").strip().lower()
            target_tokens = {tok for tok in re.findall(r"[a-z0-9]+", target) if tok}
            if target and (target in visible_terms or (target_tokens and target_tokens.issubset(visible_terms))):
                plan.sub_tasks.pop(0)
                pruned = True
                continue
            break

        if pruned and plan.llm_reasoning:
            plan.llm_reasoning = (
                f"{plan.llm_reasoning} Removed a redundant survey step because the target was already visible."
            ).strip()
        return plan

    def _create_rejected_plan(self, command: str, reason: str) -> TaskPlan:
        guidance = (
            "Rejected command because it is not executable with current robot capabilities."
        )
        return TaskPlan(
            original_command=command,
            task_type=TaskType.INSPECT,
            sub_tasks=[],
            llm_reasoning=f"{reason}: {guidance}",
        )

    def _create_plan_with_llm(self, command: str) -> TaskPlan:
        """Create plan using LLM reasoning with zero-shot prompting."""
        try:
            scene_context_str = ""
            if self._scene_context:
                scene_context_str = f"Scene context: {json.dumps(self._scene_context, indent=2)}"

            prompt_lines = [
                "You are a robotic manipulation assistant. Analyze commands and create executable plans.",
                "",
                "**EXAMPLE 1:**",
                'Command: "stack the blue cube on the red cube"',
                (
                    "Reasoning: \"I need to identify two objects: a blue cube (source) and red cube (target). "
                    "First, I'll pick the blue cube using a top-down grasp since cubes have flat tops. Then "
                    "I'll move to the red cube's location and place the blue cube on top. The placement needs "
                    "to be centered for stability.\""
                ),
                'Plan: {"task_type": "stack", "sub_tasks": [',
                (
                    '  {"task_type": "pick", "description": "Pick up the blue cube", '
                    '"target_object": "blue cube", "grasp_strategy": "top_down", "approach_distance": 0.10},'
                ),
                '  {"task_type": "move", "description": "Move above the red cube", "target_object": "red cube"},',
                '  {"task_type": "place", "description": "Place blue cube on red cube", "target_object": "red cube"}',
                "]}",
                "",
                "**EXAMPLE 2:**",
                'Command: "pick up the tall cylinder"',
                (
                    "Reasoning: \"The object is described as 'tall' and 'cylinder'. Cylinders are best "
                    "grasped from the sides for stability. I'll use a side-pinch grasp at the cylinder's "
                    "mid-height for balance.\""
                ),
                'Plan: {"task_type": "pick", "sub_tasks": [',
                (
                    '  {"task_type": "pick", "description": "Pick up the tall cylinder", '
                    '"target_object": "tall cylinder", "grasp_strategy": "side_pinch", "approach_distance": 0.12}'
                ),
                "]}",
                "",
                "**EXAMPLE 3:**",
                'Command: "move the small block to the left side"',
                (
                    "Reasoning: \"I need to find a small block and relocate it to the left workspace area "
                    "(positive y in this setup convention). Small objects may need careful grasping. I'll pick "
                    "it first, then move to approximately y=+0.3.\""
                ),
                'Plan: {"task_type": "move", "sub_tasks": [',
                (
                    '  {"task_type": "pick", "description": "Pick up the small block", '
                    '"target_object": "small block"},'
                ),
                (
                    '  {"task_type": "move", "description": "Move to left side", '
                    '"target_location": {"x": 0.45, "y": 0.30, "z": 0.15}}'
                ),
                "]}",
                "",
                "**EXAMPLE 4:**",
                'Command: "place the green object next to the blue one"',
                (
                    "Reasoning: \"I need to find both a green object and blue object. The green one will be "
                    "picked, then placed adjacent to (not on) the blue one. I'll estimate a position about 10cm "
                    "to the side.\""
                ),
                'Plan: {"task_type": "place", "sub_tasks": [',
                (
                    '  {"task_type": "pick", "description": "Pick up the green object", '
                    '"target_object": "green object"},'
                ),
                (
                    '  {"task_type": "place", "description": "Place next to blue object", '
                    '"target_object": "blue object", "relative_position": "adjacent"}'
                ),
                "]}",
                "",
                "**EXAMPLE 5:**",
                'Command: "spin in a circle"',
                (
                    "Reasoning: \"This is a non-grasping motion intent. The FR3 can execute joint-space "
                    "gestures, so I should run a safe spin primitive.\""
                ),
                'Plan: {"task_type": "move", "sub_tasks": [',
                (
                    '  {"task_type": "move", "description": "Spin the arm in place", '
                    '"action": "spin_in_place", "action_params": {"turns": 1}}'
                ),
                "]}",
                "",
                "**EXAMPLE 6:**",
                'Command: "survey the area and tell me what you see"',
                (
                    "Reasoning: \"This is a scene inspection request. I should scan the workspace and report "
                    "visible objects without attempting any grasp.\""
                ),
                'Plan: {"task_type": "inspect", "sub_tasks": [',
                (
                    '  {"task_type": "inspect", "description": "Survey the area", "action": "survey_scene", '
                    '"action_params": {"description": "survey the area and tell me what you see"}}'
                ),
                "]}",
                "",
                "**YOUR TASK:**",
                f'Command: "{command}"',
            ]
            if scene_context_str:
                prompt_lines.extend([scene_context_str, ""])
            prompt_lines.extend([
                "Planning rules:",
                "- Use the scene.visible_objects list as the best current world model.",
                "- Prefer visible_objects that are workspace_feasible=true and have xyz_base estimates.",
                "- If the command asks for an object not present in scene.visible_objects, inspect/search before any pick.",
                "- Do not invent unsupported task types such as pour, open-drawer, force-control, or fluid actions.",
                "- Use only executable task types: pick, place, move, inspect.",
                "- Use action primitives only when needed, and only if they exist in capabilities.actions.",
                "- If the command is outside current capabilities, return an empty sub_tasks list and explain why in reasoning.",
                "",
                "Analyze this command step-by-step, then provide your plan as JSON with:",
                "- task_type: one of pick|place|move|inspect",
                (
                    "- sub_tasks: array of {task_type, description, target_object (optional), "
                    "target_location (optional), action (optional), action_params (optional dict)}"
                ),
                "- reasoning: your step-by-step analysis",
                "",
                "When target_location is available from scene.visible_objects xyz_base, prefer that over guessed coordinates.",
                "Respond with ONLY valid JSON (no markdown):",
            ])
            prompt = "\n".join(prompt_lines)

            # Call LLM directly
            try:
                from ..reasoning.ollama_client import OllamaClient
                model_name = getattr(self.llm_reasoner, 'model', 'llama3.1:8b')
                base_url = getattr(self.llm_reasoner, 'base_url', 'http://localhost:11434')
                
                client = OllamaClient(model=model_name, base_url=base_url)
                messages = [{"role": "user", "content": prompt}]
                response = client.chat(messages, temperature=0.1)
                content = response['content']
                
                import json as json_lib
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
                
                result = json_lib.loads(json_content)
                result['raw_response'] = content
                self.get_logger().info("[TASK_PLANNER] LLM response parsed successfully")
            except Exception as e:
                self.get_logger().warn(f"[TASK_PLANNER] LLM call failed: {e}, using fallback")
                return self._create_plan_rule_based(command)

            plan = self._parse_llm_plan(command, result)
            
            if len(plan.sub_tasks) == 0:
                return self._create_plan_rule_based(command)
            
            self._last_plan = plan
            return plan

        except Exception as e:
            self.get_logger().error(f"[TASK_PLANNER] Planning failed: {e}")
            return self._create_plan_rule_based(command)

    def _safe_task_type(self, raw_value: Any) -> Optional[TaskType]:
        try:
            return TaskType(str(raw_value).strip().lower())
        except Exception:
            return TaskType.UNKNOWN

    def _parse_llm_plan(self, command: str, llm_result: Dict[str, Any]) -> TaskPlan:
        """Parse LLM result into TaskPlan."""
        # Determine task type
        task_type = self._safe_task_type(llm_result.get("task_type"))
        if task_type is None:
            task_type = TaskType.UNKNOWN

        # Parse sub-tasks
        sub_tasks = []
        for st in llm_result.get("sub_tasks", []):
            sub_task_type = self._normalize_sub_task_type(self._safe_task_type(st.get("task_type")), st)
            grasp_strategies = []
            for gs in st.get("grasp_strategies", []):
                grasp_strategies.append(GraspStrategy(
                    approach_vector=gs.get("approach_vector", "top"),
                    approach_distance=gs.get("approach_distance", 0.10),
                    gripper_orientation=gs.get("gripper_orientation", "parallel"),
                    gripper_width=gs.get("gripper_width", 0.04),
                    descend_speed=gs.get("descend_speed", "normal"),
                    reason=gs.get("reason", ""),
                ))

            sub_tasks.append(SubTask(
                task_type=sub_task_type,
                description=st.get("description", ""),
                target_object=st.get("target_object"),
                target_location=st.get("target_location"),
                action=st.get("action"),
                action_params=st.get("action_params", {}) or {},
                grasp_strategies=grasp_strategies if grasp_strategies else None,
                preconditions=st.get("preconditions", []),
                success_criteria=st.get("success_criteria", []),
            ))

        # Create plan
        plan = TaskPlan(
            original_command=command,
            task_type=task_type,
            sub_tasks=sub_tasks,
            llm_reasoning=llm_result.get("reasoning", llm_result.get("internal_monologue", "")),
        )

        return plan

    def _create_plan_rule_based(self, command: str) -> TaskPlan:
        """Create plan using rule-based decomposition."""
        cmd_lower = command.lower()

        action_intent = self._action_from_command(command)
        if action_intent is not None:
            action_name, action_description, action_params = action_intent
            return self._create_action_plan(
                command=command,
                action=action_name,
                description=action_description,
                action_params=action_params,
            )

        if any(marker in cmd_lower for marker in self._SPIN_MARKERS):
            return self._create_action_plan(
                command=command,
                action="spin_in_place",
                description="Spin the arm in place",
                action_params={"turns": 1},
            )

        if any(marker in cmd_lower for marker in self._WAVE_MARKERS):
            return self._create_action_plan(
                command=command,
                action="wave",
                description="Wave gesture",
                action_params={"cycles": 2},
            )

        if any(marker in cmd_lower for marker in self._READY_MARKERS):
            return self._create_action_plan(
                command=command,
                action="go_ready_pose",
                description="Move to ready pose",
            )

        if "open" in cmd_lower and "gripper" in cmd_lower:
            return self._create_action_plan(
                command=command,
                action="open_gripper",
                description="Open gripper",
            )

        if "close" in cmd_lower and "gripper" in cmd_lower:
            return self._create_action_plan(
                command=command,
                action="close_gripper",
                description="Close gripper",
            )

        # Detect task type from keywords
        if "stack" in cmd_lower:
            task_type = TaskType.STACK
        elif "place" in cmd_lower or "put" in cmd_lower:
            task_type = TaskType.PLACE
        elif "pick" in cmd_lower or "pickup" in cmd_lower or "pick up" in cmd_lower or "grab" in cmd_lower or "grasp" in cmd_lower:
            task_type = TaskType.PICK
        elif "move" in cmd_lower:
            task_type = TaskType.MOVE
        elif "inspect" in cmd_lower or "look at" in cmd_lower:
            task_type = TaskType.INSPECT
        else:
            return self._create_rejected_plan(command, "unsupported_or_ambiguous_intent")

        # Create sub-tasks based on task type
        sub_tasks = []

        if task_type == TaskType.STACK:
            source_obj, target_obj = self._extract_stack_targets(command)
            # Stack = pick source + place on target
            sub_tasks = [
                SubTask(
                    task_type=TaskType.PICK,
                    description=f"Pick up the {source_obj}",
                    target_object=source_obj,
                    grasp_strategies=[self._default_grasp_strategy()],
                    preconditions=["object_detected", "gripper_open"],
                    success_criteria=["object_grasped", "lifted_successfully"],
                ),
                SubTask(
                    task_type=TaskType.MOVE,
                    description=f"Move above the {target_obj}",
                    target_location={"x": 0.0, "y": 0.0, "z": 0.15},
                    preconditions=["object_held", "path_clear"],
                    success_criteria=["positioned_above_target"],
                ),
                SubTask(
                    task_type=TaskType.PLACE,
                    description=f"Place object on the {target_obj}",
                    target_object=target_obj,
                    preconditions=["positioned_above_target", "object_held"],
                    success_criteria=["object_released", "object_stable"],
                ),
            ]

        elif task_type == TaskType.PLACE:
            sub_tasks = [
                SubTask(
                    task_type=TaskType.MOVE,
                    description="Move to place location",
                    preconditions=["object_held"],
                    success_criteria=["positioned_at_place_location"],
                ),
                SubTask(
                    task_type=TaskType.PLACE,
                    description="Release object at location",
                    preconditions=["positioned_at_place_location"],
                    success_criteria=["object_released"],
                ),
            ]

        elif task_type == TaskType.PICK:
            target_obj = self._extract_pick_target_from_command(command)
            sub_tasks = [
                SubTask(
                    task_type=TaskType.PICK,
                    description=f"Pick up the {target_obj}",
                    target_object=target_obj,
                    grasp_strategies=[self._default_grasp_strategy()],
                    preconditions=["object_detected", "gripper_open"],
                    success_criteria=["object_grasped", "lifted_successfully"],
                ),
            ]

        elif task_type == TaskType.INSPECT:
            sub_tasks = [
                SubTask(
                    task_type=TaskType.INSPECT,
                    description="Survey the area and report visible objects",
                    action="survey_scene",
                    action_params={"description": command},
                    success_criteria=["scene_observed"],
                ),
            ]

        else:
            return self._create_rejected_plan(command, "unsupported_or_ambiguous_intent")

        plan = TaskPlan(
            original_command=command,
            task_type=task_type,
            sub_tasks=sub_tasks,
        )

        return plan

    def _create_action_plan(
        self,
        *,
        command: str,
        action: str,
        description: str,
        action_params: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        return TaskPlan(
            original_command=command,
            task_type=TaskType.MOVE,
            sub_tasks=[
                SubTask(
                    task_type=TaskType.MOVE,
                    description=description,
                    action=action,
                    action_params=action_params or {},
                    preconditions=["robot_ready"],
                    success_criteria=["action_completed"],
                )
            ],
            llm_reasoning=f"Mapped command to primitive action '{action}'.",
        )

    def _default_grasp_strategy(self) -> GraspStrategy:
        """Return default grasp strategy."""
        return GraspStrategy(
            approach_vector="top",
            approach_distance=0.10,
            gripper_orientation="parallel",
            gripper_width=0.04,
            descend_speed="normal",
            reason="Default top-down approach for cube grasping",
        )

    def create_recovery_plan(self, plan: TaskPlan, failure_reason: str) -> TaskPlan:
        """
        Create a recovery plan when the current plan fails.
        
        Args:
            plan: Current plan that failed
            failure_reason: Description of what went wrong
            
        Returns:
            New TaskPlan with recovery strategy
        """
        if self.llm_reasoner is not None:
            return self._create_recovery_plan_with_llm(plan, failure_reason)
        
        return self._create_recovery_plan_rule_based(plan, failure_reason)

    def _create_recovery_plan_with_llm(self, plan: TaskPlan, failure_reason: str) -> TaskPlan:
        """Use LLM to generate recovery strategy."""
        try:
            # Build context about current state
            current_task = plan.get_current_task()
            context = {
                "original_command": plan.original_command,
                "current_step": plan.current_step,
                "total_steps": len(plan.sub_tasks),
                "current_task": current_task.description if current_task else "unknown",
                "attempts": current_task.attempts if current_task else 0,
                "failure_reason": failure_reason,
                "recovery_history": plan.recovery_history,
            }

            result = self.llm_reasoner.reason_about_command(
                command=f"Recover from failure: {failure_reason}",
                scene_context=context,
            )

            # Parse recovery plan
            recovery_plan = self._parse_recovery_result(plan, result)
            return recovery_plan

        except Exception:
            return self._create_recovery_plan_rule_based(plan, failure_reason)

    def _parse_recovery_result(self, original_plan: TaskPlan, result: Dict[str, Any]) -> TaskPlan:
        """Parse LLM recovery result."""
        # Create modified plan
        modified_tasks = []
        for st in result.get("modified_sub_tasks", []):
            modified_tasks.append(SubTask(
                task_type=TaskType(st.get("task_type", "pick")),
                description=st.get("description", ""),
                target_object=st.get("target_object"),
                grasp_strategies=[
                    GraspStrategy(
                        approach_vector=gs.get("approach_vector", "top"),
                        approach_distance=gs.get("approach_distance", 0.10),
                        gripper_orientation=gs.get("gripper_orientation", "parallel"),
                        gripper_width=gs.get("gripper_width", 0.04),
                        descend_speed=gs.get("descend_speed", "normal"),
                        reason=gs.get("reason", ""),
                    )
                    for gs in st.get("grasp_strategies", [])
                ],
                preconditions=st.get("preconditions", []),
                success_criteria=st.get("success_criteria", []),
            ))

        # Create new plan with modified tasks
        recovery_plan = TaskPlan(
            original_command=original_plan.original_command,
            task_type=original_plan.task_type,
            sub_tasks=modified_tasks if modified_tasks else original_plan.sub_tasks,
            current_step=result.get("skip_to_step", 0),
            llm_reasoning=result.get("recovery_monologue", ""),
            status=TaskStatus.REPLANNING,
        )

        return recovery_plan

    def _create_recovery_plan_rule_based(self, plan: TaskPlan, failure_reason: str) -> TaskPlan:
        """Rule-based recovery strategies."""
        # Create modified plan with alternative strategies
        current_task = plan.get_current_task()
        if current_task is None:
            return plan

        # Modify grasp strategy based on failure
        new_grasp_strategies = []
        if "grasp_missed" in failure_reason.lower() or "missed" in failure_reason.lower():
            # Try side approach instead of top
            new_grasp_strategies = [
                GraspStrategy(
                    approach_vector="side_front",
                    approach_distance=0.08,
                    gripper_orientation="perpendicular",
                    gripper_width=0.06,
                    descend_speed="slow",
                    reason="Side approach after top-down failure",
                ),
            ]
        elif "dropped" in failure_reason.lower():
            # Try wider gripper
            new_grasp_strategies = [
                GraspStrategy(
                    approach_vector="top",
                    approach_distance=0.10,
                    gripper_orientation="parallel",
                    gripper_width=0.07,
                    descend_speed="slow",
                    reason="Wider grip after drop",
                ),
            ]
        else:
            # Retry with same strategy
            new_grasp_strategies = current_task.grasp_strategies or [self._default_grasp_strategy()]

        # Create recovery plan
        recovery_plan = TaskPlan(
            original_command=plan.original_command,
            task_type=plan.task_type,
            sub_tasks=plan.sub_tasks,
            current_step=plan.current_step,
            status=TaskStatus.REPLANNING,
            llm_reasoning=f"Recovery from: {failure_reason}. Trying alternative grasp strategy.",
        )

        # Update current task with new strategies
        if current_task:
            current_task.grasp_strategies = new_grasp_strategies
            current_task.status = TaskStatus.PENDING  # Reset to pending for retry

        return recovery_plan

    def get_plan_summary(self, plan: TaskPlan) -> str:
        """Get human-readable summary of plan."""
        lines = [
            f"Plan: {plan.original_command}",
            f"Task Type: {plan.task_type.value}",
            f"Status: {plan.status.value}",
            f"Step: {plan.current_step + 1}/{len(plan.sub_tasks)}",
        ]

        if plan.llm_reasoning:
            lines.append(f"Reasoning: {plan.llm_reasoning[:200]}...")

        lines.append("\nSub-tasks:")
        for i, task in enumerate(plan.sub_tasks):
            status_icon = "✓" if task.status == TaskStatus.COMPLETED else "○" if task.status == TaskStatus.PENDING else "✗"
            lines.append(f"  {i+1}. [{status_icon}] {task.description}")
            if task.grasp_strategies:
                gs = task.grasp_strategies[0]
                lines.append(f"      Grasp: {gs.approach_vector} approach, {gs.gripper_orientation}, {gs.descend_speed}")

        return "\n".join(lines)

    def set_scene_context_from_llava(
        self,
        llava_analysis: Dict[str, Any],
    ) -> None:
        """
        Set scene context from LLaVA analysis.
        
        This allows the planner to use LVLM-based scene understanding
        instead of or in addition to YOLO+SAM2 detection.
        
        Args:
            llava_analysis: Dictionary with LLaVA scene analysis results
                - objects: List of object descriptions
                - scene_description: Natural language scene summary
        """
        # Store LLaVA analysis in scene context
        if not hasattr(self, '_scene_context'):
            self._scene_context = {}
        
        self._scene_context['llava_objects'] = llava_analysis.get('objects', [])
        self._scene_context['llava_description'] = llava_analysis.get('scene_description', '')
        
        # Log available objects for debugging
        objects = self._scene_context.get('llava_objects', [])
        if objects:
            object_labels = [obj.get('label', 'unknown') for obj in objects]
            self.get_logger().info(f"[LLaVA] Scene contains: {', '.join(object_labels)}")
