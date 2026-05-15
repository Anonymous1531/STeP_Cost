# STeP-Cost

## Overview

STeP-Cost is a ROS 2 Humble and Navigation2-based framework for adaptive spatiotemporal costmap updates in mobile robot navigation.

The framework detects unexpected obstacles during navigation, classifies obstacle semantics with a vision-language model (VLM), and updates obstacle persistence parameters through an LLM-based TTL adaptation module. The resulting object positions and adaptive costs are projected onto Nav2 costmaps through a custom costmap layer.

This repository has been organized for reproducible experiments with TurtleBot3, Gazebo, ROS 2 Humble, and Ubuntu 22.04.

## 📦 Main Components

This repository contains two project-specific ROS 2 packages and supporting experiment scripts:

- **policy_bridge**: Runtime bridge node for unexpected obstacle detection, VLM-based semantic tagging, speed-aware obstacle classification, mission logging, and LLM-based TTL adaptation.
- **my_costmap_layers**: Nav2 costmap plugin package that applies detected obstacle positions to the global costmap.
- **experiment_suite**: Learning and evaluation scripts for running repeated navigation experiments with dynamic obstacle scenarios.
- **VLM/LLM utilities**: Standalone Gemini-based scripts for visual obstacle tagging and mission-level TTL policy updates.

The repository also includes TurtleBot3 and TurtleBot3 simulation packages for convenience and reproducibility.

## Package 1: policy_bridge

### Overview

`policy_bridge` implements the main runtime node that monitors navigation behavior, detects unexpected obstacles, captures visual evidence, classifies obstacle tags, and publishes obstacle positions for costmap integration.

### Features

- Unexpected obstacle detection from LiDAR, map, robot pose, and global plan data
- Detour-aware triggering based on path-length changes
- RGB image capture for VLM-based semantic classification
- Speed-aware obstacle tagging using online samples and a lightweight GMM classifier
- Mission summary logging for later TTL adaptation
- Optional online VLM-based cost persistence updates
- Optional LLM-based mission-end TTL refinement with lightweight case retrieval
- Automatic global costmap clearing when temporary costs expire

### Main Node

- `policy_bridge`: Runs the unexpected-obstacle detector and adaptive cost publisher.

### Important Published Topics

- `/object_world_positions` (`geometry_msgs/PoseArray`): Active obstacle positions in the map frame
- `/vlm/result` (`std_msgs/String`): VLM classification result for captured obstacle events
- `/llm_decay/result` (`std_msgs/String`): LLM-based TTL adaptation result
- `/cmd_vel` (`geometry_msgs/Twist`): Optional velocity command output used during controlled measurement steps

### Important Subscribed Topics

- `/scan` (`sensor_msgs/LaserScan`): LiDAR scan
- `/map` (`nav_msgs/OccupancyGrid`): Occupancy grid map
- `/plan` (`nav_msgs/Path`): Current global plan
- `/amcl_pose` (`geometry_msgs/PoseWithCovarianceStamped`): Robot pose estimate
- `/camera/image_raw` (`sensor_msgs/Image`): RGB camera stream for VLM evidence capture
- `/camera/depth/image_raw` (`sensor_msgs/Image`): Optional depth stream
- `/odom` (`nav_msgs/Odometry`): Odometry used for speed-aware classification
- `/navigate_to_pose/_action/status`: Nav2 goal status
- `/navigate_through_poses/_action/status`: Nav2 multi-goal status

## Package 2: my_costmap_layers

### Overview

`my_costmap_layers` provides a custom Nav2 costmap plugin that receives obstacle positions from `policy_bridge` and stamps obstacle costs into the global costmap.

### Features

- Implements the `nav2_costmap_2d::Layer` interface
- Subscribes to `/object_world_positions`
- Converts detected object poses into costmap updates
- Applies disc-shaped lethal obstacle costs around active obstacle positions
- Clears stale object marks when the active obstacle set changes
- Compatible with pluginlib and standard Nav2 costmap configuration

### Main Layer

- **ObjectAvoidanceLayer**: Applies temporary obstacle costs around detected object positions.

### Example Nav2 Configuration

Add the custom layer to the `global_costmap` plugin list in your Nav2 parameter file:

```yaml
global_costmap:
  global_costmap:
    ros__parameters:
      plugins: ["static_layer", "obstacle_layer", "object_avoidance_layer", "inflation_layer"]

      object_avoidance_layer:
        plugin: "my_costmap_layers::ObjectAvoidanceLayer"
        enabled: true
        object_positions_topic: "/object_world_positions"
        avoidance_radius: 1.0
        hold_after_clear_s: 0.1
        decay_ttl_s: 0.2
```

## VLM and LLM Modules

### `vlm_gemini_v1.py`

This script classifies captured obstacle-event images into a closed set of navigation tags. It uses the current decay table to restrict allowed tags and returns structured JSON output.

### `llm_decay_gemini_v3.py`

This script updates tag-specific TTL values after a mission. It uses mission summaries, previous TTL entries, and optional lightweight case-based retrieval to suggest updated persistence values.

### API Key Configuration

The Gemini API key can be provided through an environment variable:

```bash
export GEMINI_API_KEY="<your_api_key>"
```

Alternatively, store the key in one of the following local files:

```bash
~/.config/policy_bridge/gemini_api_key.txt
~/.config/gemini_api_key.txt
~/.gemini_api_key
~/STeP_Cost/.secrets/gemini_api_key.txt
```

Do not commit API keys or `.secrets/` directories to the repository.

## 🧪 Experiment Suite

The `experiment_suite` directory contains scripts for repeated learning and evaluation runs.

### Learning Experiments

```bash
python3 experiment_suite/learning/learning_experiment_runner.py \
  --config experiment_suite/learning/learning_experiment_config.yaml
```

Optional arguments:

```bash
--seed 42
--obstacle-type person
--speed-class slow
```

### Evaluation Experiments

```bash
python3 experiment_suite/evaluation/evaluation_runner.py \
  --config experiment_suite/evaluation/evaluation_config.yaml
```

Optional arguments:

```bash
--methods <method_name>
--scenarios <scenario_name>
--freeze-learning
--repeats 10
--seed 42
```

## 🔧 Build Instructions

### 1. Prepare a ROS 2 workspace

```bash
cd ~/STeP_Cost
```

If this repository is placed inside another ROS 2 workspace, make sure the project-specific packages are under the workspace `src/` directory.

### 2. Install ROS dependencies

```bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Install Python dependencies

```bash
pip install google-genai pydantic numpy opencv-python pillow
```

Depending on your environment, additional packages such as `cv_bridge`, `tf_transformations`, and Nav2-related ROS packages may need to be installed through `apt`.

### 4. Build the project packages

```bash
colcon build --packages-select policy_bridge my_costmap_layers
source install/setup.bash
```

To build all included ROS 2 packages:

```bash
colcon build
source install/setup.bash
```

## ▶️ Run Instructions

### 1. Launch the simulation and Nav2 stack

Launch your TurtleBot3/Gazebo environment and Nav2 stack using the included or standard TurtleBot3 launch files.

Example:

```bash
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

In another terminal:

```bash
source ~/STeP_Cost/install/setup.bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True
```

### 2. Run the adaptive cost publisher

```bash
source ~/STeP_Cost/install/setup.bash
ros2 run policy_bridge policy_bridge
```

### 3. Monitor outputs

```bash
ros2 topic echo /object_world_positions
ros2 topic echo /vlm/result
ros2 topic echo /llm_decay/result
```

## Important Runtime Files

By default, runtime logs and learned data are stored under `~/.ros/`:

- `~/.ros/decay_table.json`: Tag-specific TTL table
- `~/.ros/mission_summary.json`: Mission event log
- `~/.ros/mission_summary_out.json`: Processed mission summary
- `~/.ros/llm_decay_rag_archive.json`: Optional case-based retrieval archive
- `~/.ros/gmm_samples.json`: Speed-classifier samples
- `~/.ros/detour_events/`: Captured obstacle-event images

These files are runtime artifacts and should normally not be committed.

## Folder Structure

```text
STeP_Cost/
├── README.md
├── vlm_gemini_v1.py
├── llm_decay_gemini_v3.py
├── obstacle_speed_classifier.py
├── experiment_suite/
│   ├── learning/
│   │   ├── learning_experiment_config.yaml
│   │   ├── learning_experiment_runner.py
│   │   └── learning_obstacle_controller.py
│   ├── evaluation/
│   │   ├── evaluation_config.yaml
│   │   ├── evaluation_runner.py
│   │   └── evaluation_obstacle_controller.py
│   ├── models/
│   ├── exp_person.sdf
│   ├── exp_mecanum.sdf
│   └── exp_cart.sdf
└── src/
    ├── policy_bridge/
    │   ├── policy_bridge/
    │   │   ├── __init__.py
    │   │   └── policybridge.py
    │   ├── setup.py
    │   ├── setup.cfg
    │   └── package.xml
    └── my_costmap_layers/
        ├── src/
        │   └── ObjectAvoidanceLayer.cpp
        ├── include/
        │   └── my_costmap_layers/
        ├── plugins/
        │   └── costmap_plugins.xml
        ├── CMakeLists.txt
        └── package.xml

```

## 🔗 Dependencies

### System

- Ubuntu 22.04
- ROS 2 Humble
- Navigation2
- TurtleBot3 packages
- Gazebo
- `colcon`
- `rosdep`

### ROS Packages

- `rclpy`
- `rclcpp`
- `nav2_costmap_2d`
- `nav2_util`
- `nav2_msgs`
- `geometry_msgs`
- `sensor_msgs`
- `nav_msgs`
- `std_msgs`
- `tf2_ros`
- `tf2_geometry_msgs`
- `pluginlib`

### Python Packages

- `google-genai`
- `pydantic`
- `numpy`
- `opencv-python`
- `pillow`

## 🔧 Troubleshooting

### The custom costmap layer does not appear

- Confirm that `my_costmap_layers` was built successfully.
- Check that `source install/setup.bash` was run in the current terminal.
- Verify that `object_avoidance_layer` is included in the `global_costmap` plugin list.
- Confirm that the plugin name is exactly `my_costmap_layers::ObjectAvoidanceLayer`.

### No obstacle positions are published

- Check that `/scan`, `/map`, `/plan`, and `/amcl_pose` are being published.
- Verify that `policy_bridge` is running.
- Confirm that unexpected obstacle detection is enabled.
- Use `ros2 topic echo /object_world_positions` to inspect outputs.

### VLM classification does not run

- Check that `vlm_enable` is set to `true`.
- Confirm that `/camera/image_raw` is available.
- Verify that `GEMINI_API_KEY` or a local API key file is configured.
- Inspect logs under `~/.ros/detour_events/`.

### LLM TTL updates do not run

- Check that `llm_decay_enable` is set to `true`.
- Verify that mission summary files are being generated under `~/.ros/`.
- Confirm that the Gemini API key is available.

## 📄 License

This code is made available for academic purposes accompanying a manuscript submission.

Unauthorized reproduction, redistribution, or modification outside the review process is not permitted unless explicitly allowed by the authors.

© Anonymous Authors. All rights reserved.

## 📧 Contact

For questions regarding the manuscript or this framework, please refer to the corresponding anonymous submission record.
