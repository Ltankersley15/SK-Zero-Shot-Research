# FR3 LVLM Utilities

This package now serves as the shared perception, reasoning, and UI utility layer for the active FR3 zero-shot agent stack.

Kept here:
- RGB-D geometry helpers
- proposal generation and attribute matching
- command reasoning helpers
- camera info republisher
- Tk command GUI

Retired from this package:
- the old standalone LVLM runtime bringup
- the old integrated `llm_agent` launch path

Use the active stack through:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/lonnie-research/ws_moveit2
colcon build --packages-select fr3_zero_shot_agent fr3_bringup fr3_lvlm_agent --symlink-install
source install/setup.bash
ros2 launch fr3_bringup fr3_isaac_moveit_agent.launch.py
```
