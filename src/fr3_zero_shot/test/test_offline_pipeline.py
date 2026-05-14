import json

from fr3_zero_shot.node import ZeroShotOfflinePipeline


def test_offline_pipeline_pick_and_place_in_cup(tmp_path):
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": "obj_blue_cube",
                        "label": "blue cube",
                        "color": "blue",
                        "shape": "cube",
                        "xyz": [0.48, 0.0, 0.02],
                        "footprint_xy": [0.04, 0.04],
                        "height_m": 0.04,
                        "pickable": True,
                    },
                    {
                        "object_id": "obj_cup",
                        "label": "cup",
                        "shape": "cup",
                        "xyz": [0.58, 0.0, 0.04],
                        "footprint_xy": [0.08, 0.08],
                        "height_m": 0.08,
                        "pickable": False,
                        "container_like": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ZeroShotOfflinePipeline().run(
        command="pick up the blue cube and place it in the cup",
        scene_path=str(scene_path),
    )

    assert result["success"]
    assert result["outcomes"][-1].message == "place-in complete"

