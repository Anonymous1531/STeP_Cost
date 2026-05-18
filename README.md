# STeP-Cost

## Overview

STeP-Cost is a ROS 2 Humble and Navigation2-based framework for adaptive costmap TTL policy learning in mobile robot navigation.

The framework identifies detour-inducing obstacle situations during navigation, classifies obstacle semantics with a Vision-Language Model (VLM), and updates compound tag-wise Time-to-Live (TTL) values through an LLM-based post-mission policy update module. The resulting obstacle positions and adaptive residual costs are applied to the Nav2 global costmap through a custom costmap layer.

**STeP-Cost is designed as a plug-in addition to an existing Nav2 stack.** It does not replace the planner or controller — it only extends the global costmap with a learned cost persistence policy. You can integrate `policy_bridge` and `my_costmap_layers` into your own ROS 2 navigation environment.

---

## 📦 Components

- **`policy_bridge`**: Runtime ROS 2 node for detour-gated event detection, VLM-based semantic category estimation, GMM-based motion class estimation, mission summary logging, and LLM-based post-mission TTL policy updates.
- **`my_costmap_layers`**: Nav2 costmap plugin that applies adaptive residual obstacle costs to the global costmap.
- **`vlm_gemini_v1.py`**: Standalone VLM script for visual obstacle category classification.
- **`llm_decay_gemini_v3.py`**: Standalone LLM script for mission-level compound tag-wise TTL policy updates.
- **`obstacle_speed_classifier.py`**: GMM-based speed classifier for category-conditioned motion class estimation.

---

## 🔌 Using STeP-Cost as a Plugin

This section describes how to integrate STeP-Cost into your own ROS 2 navigation environment.

### Step 1: Clone and build

```bash
git clone https://github.com/Anonymous1531/STeP_Cost.git ~/STeP_Cost
cd ~/STeP_Cost
rosdep install --from-paths src --ignore-src -r -y
pip install google-genai pydantic numpy opencv-python pillow scikit-learn
colcon build --packages-select policy_bridge my_costmap_layers
source install/setup.bash
```

### Step 2: Add the costmap layer to your Nav2 config

Add `object_avoidance_layer` to your existing Nav2 parameter file:

```yaml
global_costmap:
  global_costmap:
    ros__parameters:
      plugins: ["static_layer", "obstacle_layer", "object_avoidance_layer", "inflation_layer"]

      object_avoidance_layer:
        plugin: "my_costmap_layers::ObjectAvoidanceLayer"
        enabled: true
        object_positions_topic: "/object_world_positions"
        avoidance_radius: 1.0      # Adjust to your robot footprint (meters)
        hold_after_clear_s: 0.1
        decay_ttl_s: 0.2
```

### Step 3: Configure corridor geometry for your map

`policy_bridge` uses corridor coordinates to compute the depth-ratio TTL correction. Set these to match your map in your launch file or parameter file:

```yaml
policy_bridge:
  ros__parameters:
    # Corridor extent along the x-axis (map frame)
    corridor_start_x: 0.0        # x coordinate of corridor entrance
    corridor_end_x: 32.0         # x coordinate of corridor exit

    # y-center of each corridor lane (one value per corridor)
    corridor_y_centers: [0.0, -4.76, -13.49, -18.17]

    # Half-width of each corridor lane (meters)
    corridor_y_half_width: 2.5

    # Margin at corridor entrance/exit to exclude edge regions
    corridor_x_margin: 1.0
```

> If your environment does not have clearly defined corridor directions (e.g., open areas or intersections), the depth-ratio correction is skipped automatically and the base TTL is applied directly.

### Step 4: Set the detour detection threshold

Adjust these values to match your map resolution and corridor geometry:

```yaml
policy_bridge:
  ros__parameters:
    detour_ratio_threshold: 1.15        # Trigger when path increases by 15%
    detour_min_previous_length_m: 0.50  # Ignore very short reference paths
    detour_hold_s: 8.0                  # Duration to hold a registered detour event (seconds)
    detour_cooldown_s: 2.0              # Minimum interval between consecutive events (seconds)
```

### Step 5: Set your API key and enable VLM/LLM

```bash
export GEMINI_API_KEY="<your_api_key>"
```

Enable VLM and LLM in your parameter file:

```yaml
policy_bridge:
  ros__parameters:
    vlm_enable: true
    vlm_script: "~/STeP_Cost/vlm_gemini_v1.py"
    llm_decay_enable: true
    llm_decay_script: "~/STeP_Cost/llm_decay_gemini_v3.py"
    llm_decay_rag_enable: true
```

Alternatively, store the key in a local file (checked in order):

```
~/.config/policy_bridge/gemini_api_key.txt
~/.config/gemini_api_key.txt
~/.gemini_api_key
~/STeP_Cost/.secrets/gemini_api_key.txt
```

> Do not commit API keys or `.secrets/` directories to the repository.

### Step 6: Run policy_bridge alongside your Nav2 stack

```bash
source ~/STeP_Cost/install/setup.bash
ros2 run policy_bridge policy_bridge
```

Monitor outputs:

```bash
ros2 topic echo /object_world_positions
ros2 topic echo /vlm/result
ros2 topic echo /llm_decay/result
```

---

## Package 1: policy_bridge

### Features

- Detour-gated event detection based on global path length changes
- RGB image capture for VLM-based semantic category classification
- LiDAR-based speed estimation and category-conditioned GMM motion class inference
- Compound tag construction (`category:motion`) per detour event
- Mission summary logging for post-mission TTL adaptation
- LLM-based TTL update proposal generation with confidence-based acceptance policy
- Automatic global costmap clearing when residual costs expire

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/object_world_positions` | `geometry_msgs/PoseArray` | Active obstacle positions in the map frame |
| `/vlm/result` | `std_msgs/String` | VLM classification result |
| `/llm_decay/result` | `std_msgs/String` | LLM-based TTL update result |

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | LiDAR scan |
| `/map` | `nav_msgs/OccupancyGrid` | Occupancy grid map |
| `/plan` | `nav_msgs/Path` | Current global plan |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | Robot pose estimate |
| `/camera/image_raw` | `sensor_msgs/Image` | RGB camera stream for VLM evidence capture |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Optional depth stream |
| `/odom` | `nav_msgs/Odometry` | Odometry for speed estimation |
| `/navigate_to_pose/_action/status` | — | Nav2 goal status |
| `/navigate_through_poses/_action/status` | — | Nav2 multi-goal status |

### Full Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `detour_ratio_threshold` | `1.15` | Path length increase ratio to trigger a detour event |
| `detour_min_previous_length_m` | `0.50` | Minimum reference path length to activate the detour gate |
| `detour_hold_s` | `8.0` | Duration to hold a registered detour event (seconds) |
| `detour_cooldown_s` | `2.0` | Minimum interval between consecutive detour events (seconds) |
| `corridor_start_x` | `0.0` | Corridor entrance x coordinate (map frame) |
| `corridor_end_x` | `32.0` | Corridor exit x coordinate (map frame) |
| `corridor_y_centers` | `[0.0, ...]` | Y centers of corridor lanes (map frame) |
| `corridor_y_half_width` | `2.5` | Half-width of each corridor lane (meters) |
| `corridor_x_margin` | `1.0` | Margin at corridor entrance/exit (meters) |
| `default_cost_ttl_s` | `6.0` | Fallback TTL before a compound tag is confirmed |
| `max_range_m` | `6.0` | Maximum LiDAR range for obstacle detection |
| `min_range_m` | `0.10` | Minimum LiDAR range for obstacle detection |
| `cluster_dist_thresh` | `0.20` | Distance threshold for LiDAR point clustering |
| `cluster_min_points` | `5` | Minimum points to form a valid cluster |
| `vlm_enable` | `false` | Enable VLM-based semantic classification |
| `vlm_script` | `~/STeP_Cost/vlm_gemini_v1.py` | Path to the VLM script |
| `vlm_model` | `gemini-2.5-flash` | Gemini model for VLM |
| `vlm_timeout_sec` | `180.0` | VLM call timeout (seconds) |
| `speed_classifier_enable` | `true` | Enable GMM-based motion class estimation |
| `gmm_min_samples` | `10` | Minimum samples before GMM classification is active |
| `llm_decay_enable` | `false` | Enable LLM-based post-mission TTL updates |
| `llm_decay_script` | `~/STeP_Cost/llm_decay_gemini_v3.py` | Path to the LLM script |
| `llm_decay_model` | `gemini-2.5-flash` | Gemini model for LLM |
| `llm_decay_rag_enable` | `false` | Enable retrieval-augmented memory for LLM |
| `llm_decay_confidence_threshold` | `0.8` | Confidence threshold for auto-acceptance |
| `llm_decay_approval_mode` | `auto` | Approval mode: `auto`, `human`, or `human_all` |
| `llm_decay_retrieval_max_repeat1_cases` | `30` | Maximum retrieved past cases per tag |
| `enable_global_clear_on_expire` | `true` | Clear global costmap when TTL expires |

---

## Package 2: my_costmap_layers

### Features

- Implements the `nav2_costmap_2d::Layer` interface
- Applies disc-shaped lethal obstacle costs around active obstacle positions
- Clears stale obstacle marks when the active obstacle set changes
- Compatible with pluginlib and standard Nav2 costmap configuration

### Main Layer

- **`ObjectAvoidanceLayer`**: Applies temporary residual costs around detected obstacle positions.

---

## 📁 Runtime Files

By default, runtime logs and learned data are stored under `~/.ros/`:

| File | Description |
|---|---|
| `~/.ros/decay_table.json` | Compound tag-wise TTL table |
| `~/.ros/mission_summary.json` | Per-mission detour event log |
| `~/.ros/llm_decay_rag_archive.json` | RAG-based past proposal archive |
| `~/.ros/gmm_samples.json` | Speed-classifier samples |
| `~/.ros/detour_events/` | Captured obstacle-event images |

These files are runtime artifacts and should not be committed to the repository.

---

## 📂 Folder Structure

```
STeP_Cost/
├── README.md
├── vlm_gemini_v1.py              # VLM script
├── llm_decay_gemini_v3.py        # LLM script
├── obstacle_speed_classifier.py  # GMM classifier
└── src/
    ├── policy_bridge/
    │   ├── policy_bridge/
    │   │   ├── __init__.py
    │   │   └── policybridge.py
    │   ├── resource/
    │   ├── setup.py
    │   ├── setup.cfg
    │   └── package.xml
    └── my_costmap_layers/
        ├── src/
        │   └── ObjectAvoidanceLayer.cpp
        ├── include/
        │   └── my_costmap_layers/
        │       └── ObjectAvoidanceLayer.hpp
        ├── plugins/
        │   └── costmap_plugins.xml
        ├── CMakeLists.txt
        └── package.xml
```

---

## 🔗 Dependencies

### System
- Ubuntu 22.04, ROS 2 Humble, Navigation2, Gazebo (Classic), `colcon`, `rosdep`

### ROS Packages
- `rclpy`, `rclcpp`
- `nav2_costmap_2d`, `nav2_util`, `nav2_msgs`, `nav2_simple_commander`
- `geometry_msgs`, `sensor_msgs`, `nav_msgs`, `std_msgs`
- `tf2_ros`, `tf2_geometry_msgs`, `pluginlib`

### Python Packages
- `google-genai`, `pydantic`, `numpy`, `opencv-python`, `pillow`, `scikit-learn`

---

## 🔧 Troubleshooting

### The custom costmap layer does not appear
- Confirm `my_costmap_layers` was built and `source install/setup.bash` was run.
- Verify `object_avoidance_layer` is in the `global_costmap` plugin list.
- Confirm the plugin name is exactly `my_costmap_layers::ObjectAvoidanceLayer`.

### No obstacle positions are published
- Check that `/scan`, `/map`, `/plan`, and `/amcl_pose` are active.
- Verify `policy_bridge` is running.
- Use `ros2 topic echo /object_world_positions` to inspect outputs.

### Detour events are not triggered
- Lower `detour_ratio_threshold` if your corridors are short.
- Confirm corridor coordinates (`corridor_y_centers`, `corridor_start_x`, `corridor_end_x`) match your map frame.
- Verify `detour_min_previous_length_m` is not larger than your actual path lengths.

### VLM classification does not run
- Check `vlm_enable: true` and confirm `/camera/image_raw` is published.
- Verify `GEMINI_API_KEY` is set and `vlm_script` path is correct.
- Inspect captured images under `~/.ros/detour_events/`.

### LLM TTL updates do not run
- Check `llm_decay_enable: true`.
- Verify `~/.ros/mission_summary.json` is being generated after each trial.
- Confirm the Gemini API key is available.

### GMM motion class is not estimated
- Check `speed_classifier_enable: true`.
- Verify `gmm_min_samples` samples have been collected — classification is inactive until the minimum is reached.
- Inspect `~/.ros/gmm_samples.json` to confirm samples are accumulating.

---

## 📄 License

This code is made available for academic purposes accompanying a manuscript submission.

Unauthorized reproduction, redistribution, or modification outside the review process is not permitted unless explicitly allowed by the authors.

© Anonymous Authors. All rights reserved.

## 📧 Contact

For questions regarding the manuscript or this framework, please refer to the corresponding anonymous submission record.
