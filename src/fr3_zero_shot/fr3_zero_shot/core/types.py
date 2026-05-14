from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SkillName(str, Enum):
    PICK = "pick"
    PLACE_IN = "place_in"
    PLACE_RELATIVE = "place_relative"
    OBSERVE = "observe"
    STOP = "stop"


class ApproachFamily(str, Enum):
    TOP_DOWN = "top_down"
    TOP_DOWN_YAW_45 = "top_down_yaw_45"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    SIDE_FRONT = "side_front"
    SIDE_BACK = "side_back"
    AUTO = "auto"


class FailureCode(str, Enum):
    NONE = "none"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    SOURCE_NOT_VISIBLE = "source_not_visible"
    TARGET_NOT_VISIBLE = "target_not_visible"
    NO_VALID_GRASP = "no_valid_grasp"
    MOTION_FAILED = "motion_failed"
    CARRY_VERIFY_FAILED = "carry_verify_failed"
    RELEASE_GATE_FAILED = "release_gate_failed"
    PLACE_VERIFY_FAILED = "place_verify_failed"
    UNSUPPORTED_COMMAND = "unsupported_command"


@dataclass(frozen=True)
class ObjectHypothesis:
    object_id: str
    label: str
    color: str | None
    shape: str | None
    xyz: tuple[float, float, float]
    footprint_xy: tuple[float, float]
    height_m: float
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    visible: bool
    pickable: bool
    container_like: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolicPlan:
    command: str
    skill_sequence: list[SkillName]
    source_ref: str | None
    target_ref: str | None
    relation: str | None
    approach_preference: ApproachFamily
    avoid_refs: list[str]
    reason: str


@dataclass(frozen=True)
class GroundedPlan:
    symbolic: SymbolicPlan
    source_id: str | None
    target_id: str | None
    avoid_ids: list[str]


@dataclass(frozen=True)
class SceneRelation:
    object_id: str
    near_ids: tuple[str, ...] = ()
    top_down_clear: bool = True
    side_clear: bool = True


@dataclass(frozen=True)
class SceneSummary:
    objects: tuple[ObjectHypothesis, ...]
    relations: tuple[SceneRelation, ...] = ()


@dataclass(frozen=True)
class GraspCandidate:
    family: ApproachFamily
    pregrasp_xyz: tuple[float, float, float]
    grasp_xyz: tuple[float, float, float]
    retreat_xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    clearance_score: float
    ik_feasible: bool
    collision_free: bool
    keepout_clearance_m: float
    reason: str


@dataclass(frozen=True)
class StrategyOption:
    option_id: str
    description: str
    approach_family: ApproachFamily
    requires_reobserve: bool = False
    reobserve_view: str | None = None
    allow_recovery: bool = False
    avoid_refs: list[str] = field(default_factory=list)
    risk_summary: str = ""


@dataclass(frozen=True)
class StrategyDecision:
    selected_option_id: str
    approach_preference: ApproachFamily
    reobserve_view: str | None
    avoid_refs: list[str]
    allow_recovery: bool
    reason: str
    planner_source: str


@dataclass(frozen=True)
class ReobserveDirective:
    required: bool
    view_goal: str | None
    target_ref: str | None
    reason: str


@dataclass(frozen=True)
class GraspIntent:
    approach_direction: str
    grasp_region: str
    wrist_intent: str
    contact_style: str


@dataclass(frozen=True)
class RetreatIntent:
    primary_direction: str
    secondary_direction: str | None
    lift_after_clearance: bool


@dataclass(frozen=True)
class MotionSketch:
    reobserve: ReobserveDirective
    grasp_intent: GraspIntent
    retreat_intent: RetreatIntent
    constraints: list[str]
    avoid_refs: list[str]
    allow_recovery: bool
    reason: str
    planner_source: str


@dataclass(frozen=True)
class SkillOutcome:
    success: bool
    code: FailureCode
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseGateResult:
    ready: bool
    actual_center_err_m: float
    actual_target_xy_err_m: float
    z_above_limit_m: float
    rim_drop_m: float
    actual_object_center_xy: tuple[float, float]
    target_center_xy: tuple[float, float]
