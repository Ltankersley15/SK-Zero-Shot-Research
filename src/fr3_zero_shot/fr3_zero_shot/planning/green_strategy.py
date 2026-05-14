from __future__ import annotations

from typing import Any

from fr3_zero_shot.core.types import ApproachFamily, ObjectHypothesis, StrategyDecision, StrategyOption
from fr3_zero_shot.live_pick_place_utils import side_family_away_from_keepout


GREEN_TOP_DOWN_RECOVERY_CLEARANCE_M = 0.095


def initial_green_strategy_options(green: ObjectHypothesis, cup: ObjectHypothesis) -> list[StrategyOption]:
    side = green_approach_family(green, cup)
    return [
        StrategyOption(
            option_id='reobserve_green_side',
            description='Move to the named green side reobserve views before choosing the grasp strategy.',
            approach_family=side,
            requires_reobserve=True,
            reobserve_view='green_reobserve',
            allow_recovery=False,
            avoid_refs=['cup'],
            risk_summary='best first step when cup partly hides the green cube',
        ),
        StrategyOption(
            option_id='edge_yawed_pick_away_from_cup',
            description='Use the calibrated yawed top-down edge grasp biased away from the cup.',
            approach_family=ApproachFamily.TOP_DOWN_YAW_45,
            requires_reobserve=False,
            allow_recovery=True,
            avoid_refs=['cup'],
            risk_summary='physically validated, but less informative than reobserving first',
        ),
        StrategyOption(
            option_id='fail_closed',
            description='Stop without moving if the scene is too ambiguous or unsafe.',
            approach_family=ApproachFamily.AUTO,
            avoid_refs=['cup'],
            risk_summary='safe refusal',
        ),
    ]


def execution_green_strategy_options(green: ObjectHypothesis, cup: ObjectHypothesis) -> list[StrategyOption]:
    side = green_approach_family(green, cup)
    side_option = f'{side.value}_pull_away'
    return [
        StrategyOption(
            option_id=side_option,
            description='Approach from the exposed side, enter laterally at grasp height, and pull away from the cup before lifting.',
            approach_family=side,
            requires_reobserve=False,
            allow_recovery=True,
            avoid_refs=['cup'],
            risk_summary='required when vertical descent would contact the cup rim; one side-entry retry is allowed',
        ),
        StrategyOption(
            option_id='edge_yawed_pick_away_from_cup',
            description='Use a yawed wrist at the cube edge, with the target biased away from the cup.',
            approach_family=ApproachFamily.TOP_DOWN_YAW_45,
            requires_reobserve=False,
            allow_recovery=True,
            avoid_refs=['cup'],
            risk_summary='lowest-risk validated physical strategy after reobserve; recovery allowed if cube separates',
        ),
        StrategyOption(
            option_id='recover_yawed_top_down_if_separated',
            description='Authorize one bounded yawed top-down recovery only after the cube is visibly separated from the cup.',
            approach_family=ApproachFamily.TOP_DOWN_YAW_45,
            requires_reobserve=False,
            allow_recovery=True,
            avoid_refs=['cup'],
            risk_summary='not a first-contact strategy; only useful after a failed separating attempt',
        ),
        StrategyOption(
            option_id='fail_closed',
            description='Stop if the cup keepout or source location makes all offered approaches unsafe.',
            approach_family=ApproachFamily.AUTO,
            avoid_refs=['cup'],
            risk_summary='safe refusal',
        ),
    ]


def green_strategy_scene_facts(green: ObjectHypothesis, cup: ObjectHypothesis) -> dict[str, Any]:
    dx = float(green.xyz[0] - cup.xyz[0])
    dy = float(green.xyz[1] - cup.xyz[1])
    return {
        'source': {
            'id': green.object_id,
            'label': green.label,
            'color': green.color,
            'shape': green.shape,
            'visible': green.visible,
        },
        'keepout': {
            'id': cup.object_id,
            'label': cup.label,
            'container_like': cup.container_like,
            'shape': cup.shape,
        },
        'relative_direction_from_cup': {
            'x_sign': 'positive' if dx >= 0.0 else 'negative',
            'y_sign': 'positive' if dy >= 0.0 else 'negative',
            'dominant_axis': 'x' if abs(dx) >= abs(dy) else 'y',
            'suggested_side_family': green_approach_family(green, cup).value,
        },
        'top_down_risk': 'high because the cup is a nearby keepout and partially occludes the source',
        'nearest_keepout_distance_m': xy_distance(green, cup),
        'available_reobserve_views': ['green_reobserve', 'green_wide_reobserve', 'cup_top'],
        'robot_authority_boundary': (
            'LLM chooses strategy only; deterministic code owns exact poses, quaternions, IK, '
            'collision keepouts, gripper commands, and final verification.'
        ),
    }


def green_approach_family(green: ObjectHypothesis, cup: ObjectHypothesis) -> ApproachFamily:
    if float(green.xyz[1] - cup.xyz[1]) > 0.05:
        return ApproachFamily.SIDE_RIGHT
    return side_family_away_from_keepout(green, cup)


def green_recovery_allowed(
    decision: StrategyDecision,
    recovery_green: ObjectHypothesis | None,
    cup: ObjectHypothesis,
) -> bool:
    recovery_clearance = xy_distance(recovery_green, cup) if recovery_green is not None else None
    return bool(
        decision.allow_recovery
        and recovery_clearance is not None
        and recovery_clearance >= GREEN_TOP_DOWN_RECOVERY_CLEARANCE_M
    )


def xy_distance(a: ObjectHypothesis | None, b: ObjectHypothesis | None) -> float | None:
    if a is None or b is None:
        return None
    dx = float(a.xyz[0] - b.xyz[0])
    dy = float(a.xyz[1] - b.xyz[1])
    return (dx * dx + dy * dy) ** 0.5
