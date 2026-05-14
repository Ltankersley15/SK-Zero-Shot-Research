from glob import glob
import os

from setuptools import find_packages, setup


package_name = "fr3_zero_shot"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("fr3_zero_shot/config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="lonnie",
    maintainer_email="lonnie@todo.todo",
    description="Zero-shot FR3 manipulation pipeline with bounded model authority.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "zero_shot = fr3_zero_shot.node:main",
            "zero_shot_live_probe = fr3_zero_shot.live_probe:main",
            "zero_shot_live_pick = fr3_zero_shot.live_pick:main",
            "zero_shot_live_green_occluded_pick = fr3_zero_shot.live_green_occluded_pick:main",
            "zero_shot_live_pick_place = fr3_zero_shot.live_pick_place:main",
            "zero_shot_live_pick_place_sequence = fr3_zero_shot.live_pick_place_sequence:main",
            "zero_shot_live_stack_cubes = fr3_zero_shot.live_stack_cubes:main",
            "zero_shot_command_router = fr3_zero_shot.live_command_router:main",
        ],
    },
)
