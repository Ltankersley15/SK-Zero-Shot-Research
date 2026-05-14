# SK Zero-Shot Research FR3 Isaac Workspace

This repository is a trimmed ROS 2 Jazzy workspace for the zero-shot FR3 manipulation stack used with Isaac Sim. It contains the custom LVLM/planning code, the FR3 MoveIt/skill-server bridge, the FR3 Isaac scene, and the launch files needed to run live Isaac validation.

## Contents

- `fr3_test.usda`: Isaac Sim scene with the FR3, table objects, cup/container, RGB-D cameras, ROS 2 bridge graph, and test cubes.
- `src/fr3_zero_shot`: zero-shot manipulation pipeline.
- `src/fr3_lvlm_agent`: shared LVLM/OpenAI client and camera-info republisher utilities used by the pipeline.
- `src/fr3_bringup`: launch files for Isaac + MoveIt + FR3 skill server.
- `src/fr3_skill_interfaces`: ROS service definitions used by the skill server and motion adapter.
- `src/fr3_skill_server`: MoveIt-backed pose/joint planning and execution services.
- `src/fr3_moveit_relay`: trajectory/gripper relays used with Isaac controllers.
- `src/fr3_isaac_control`: Isaac controller configuration.
- `src/franka_description` and `src/franka_fr3_moveit_config`: FR3 robot description and MoveIt configuration.

Generated outputs such as `build/`, `install/`, `log/`, Python caches, pytest caches, and local model `.pt` checkpoint files are intentionally excluded.

## Tested Local Versions

The last local validation was performed on:

- Ubuntu 24.04 / ROS 2 Jazzy
- Python 3.12
- Isaac Sim `5.1.0-rc.19`
- MoveIt packages from the ROS 2 Jazzy apt repositories
- OpenAI Python SDK `>=1.0.0`

Isaac Sim is not vendored in this repository. Install it separately from NVIDIA and keep the scene file at `fr3_test.usda`.

## Fresh Machine Setup

Install ROS 2 Jazzy desktop and common build tools:

```bash
sudo apt update
sudo apt install -y software-properties-common curl gnupg lsb-release
sudo add-apt-repository universe
sudo apt install -y ros-jazzy-desktop python3-colcon-common-extensions python3-rosdep python3-pip
sudo rosdep init || true
rosdep update
```

Install MoveIt and runtime dependencies:

```bash
sudo apt install -y \
  ros-jazzy-moveit \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-joint-state-publisher \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-xacro \
  ros-jazzy-tf2-ros \
  ros-jazzy-cv-bridge

python3 -m pip install --user --break-system-packages \
  openai pillow numpy opencv-python pytest
```

Clone and build:

```bash
git clone https://github.com/Ltankersley15/SK-Zero-Shot-Research.git ~/ws_moveit2
cd ~/ws_moveit2
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select \
  fr3_skill_interfaces \
  franka_description \
  franka_fr3_moveit_config \
  fr3_isaac_control \
  fr3_moveit_relay \
  fr3_skill_server \
  fr3_lvlm_agent \
  fr3_bringup \
  fr3_zero_shot
source install/setup.bash
```

Set the OpenAI key for LLM strategy and motion-sketch planning:

```bash
export OPENAI_API_KEY="..."
```

## Open Isaac Sim GUI

Start Isaac Sim from your local Isaac installation, then open the scene:

```bash
/home/$USER/isaac-sim/isaac-sim.sh
```

In the Isaac GUI:

1. Open `~/ws_moveit2/fr3_test.usda`.
2. Press `Stop` once if the simulation is already running.
3. Press `Play`.
4. Confirm ROS 2 topics are publishing:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep -E 'joint_states|d455|zed|clock'
```

## Bring Up MoveIt And The FR3 Skill Server

In a new terminal:

```bash
cd ~/ws_moveit2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch fr3_bringup fr3_isaac_moveit_bringup.launch.py \
  publish_camera_tf:=true \
  enable_camera_info_repub:=true \
  fjt_extra_settle_time:=2.0 \
  fjt_min_traj_duration:=0.8 \
  fjt_goal_tolerance_rad:=0.10
```

Wait for `FR3SkillServer up. Ready.` before running tests.

## Run Tests In Isaac

All live commands should be run from a terminal that has sourced both ROS and the workspace:

```bash
cd ~/ws_moveit2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

Visible cube pickup:

```bash
ros2 run fr3_zero_shot zero_shot_live_pick \
  --execute \
  --color blue \
  --position-tolerance 0.025 \
  --orientation-tolerance 0.45
```

Blue cube into cup:

```bash
ros2 run fr3_zero_shot zero_shot_live_pick_place \
  --execute \
  --position-tolerance 0.025 \
  --orientation-tolerance 0.45
```

Large red, blue, and yellow cubes into the cup:

```bash
ros2 run fr3_zero_shot zero_shot_live_pick_place_sequence \
  --execute \
  --colors red blue yellow \
  --position-tolerance 0.025 \
  --orientation-tolerance 0.45
```

Stack yellow on blue, then small red on yellow:

```bash
ros2 run fr3_zero_shot zero_shot_live_stack_cubes \
  --execute \
  --stack-sources yellow small_red \
  --position-tolerance 0.025 \
  --orientation-tolerance 0.45
```

Partially occluded green cube pickup with LLM strategy/motion-sketch authority:

```bash
ros2 run fr3_zero_shot zero_shot_live_green_occluded_pick \
  --execute \
  --llm-strategy \
  --position-tolerance 0.035 \
  --orientation-tolerance 0.55
```

Each live command emits a JSON result on stdout. Treat terminal success as insufficient by itself: for validation, also inspect the Isaac GUI or saved screenshots and confirm the object state, cup state, and gripper retention.

## Offline Unit Tests

```bash
cd ~/ws_moveit2
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src/fr3_zero_shot pytest -q src/fr3_zero_shot/test
```

## Resetting Isaac Between Runs

Use the Isaac GUI controls in this order:

1. Press square `Stop`.
2. Wait briefly.
3. Press triangle `Play`.
4. Wait for ROS topics and `/joint_states` to resume.

Then rerun the ROS command from a sourced terminal.
