#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
from nav2_msgs.srv import ClearEntireCostmap
from nav2_msgs.msg import BehaviorTreeLog
from sensor_msgs.msg import LaserScan
from tf_transformations import quaternion_from_euler
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState


def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def now_str():
    return time.strftime("%Y%m%d_%H%M%S")


def yaw_to_quat(yaw: float):
    q = quaternion_from_euler(0.0, 0.0, yaw)
    return q


def pose_stamped_xyyaw(x: float, y: float, yaw: float, stamp, frame_id: str = "map") -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = 0.0
    q = yaw_to_quat(yaw)
    msg.pose.orientation.x = q[0]
    msg.pose.orientation.y = q[1]
    msg.pose.orientation.z = q[2]
    msg.pose.orientation.w = q[3]
    return msg

def map_to_world(cfg: Dict, x_map: float, y_map: float, yaw_map: float):
    tf_cfg = cfg.get("frames", {}).get("map_to_world", {})
    enabled = bool(tf_cfg.get("enabled", True))
    if not enabled:
        return float(x_map), float(y_map), float(yaw_map)

    dx = float(tf_cfg.get("dx", 0.0))
    dy = float(tf_cfg.get("dy", 0.0))
    dyaw = float(tf_cfg.get("dyaw", 0.0))

    c = math.cos(dyaw)
    s = math.sin(dyaw)

    x_world = c * x_map - s * y_map + dx
    y_world = s * x_map + c * y_map + dy
    yaw_world = yaw_map + dyaw

    return x_world, y_world, yaw_world

class ExperimentNode(Node):
    def __init__(self, cfg: Dict):
        super().__init__("factory6_experiment_runner")

        self.cfg = cfg
        self.frame_id = "map"

        topics = cfg.get("topics", {})
        self.initialpose_topic = topics.get("initialpose", "/initialpose")
        self.amcl_pose_topic = topics.get("amcl_pose", "/amcl_pose")
        self.scan_topic = topics.get("scan", "/scan")
        self.bt_log_topic = topics.get("behavior_tree_log", "/behavior_tree_log")

        trial_cfg = cfg.get("trial", {})
        self.path_eps = float(trial_cfg.get("path_accum_min_step_m", 0.01))
        self.collision_range_threshold = float(trial_cfg.get("collision_range_threshold_m", 0.08))
        self.use_map_frame_for_path_length = bool(trial_cfg.get("use_map_frame_for_path_length", True))

        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, self.initialpose_topic, 10)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, self.amcl_pose_topic, self._amcl_cb, 20
        )
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, 20)
        self.bt_log_sub = self.create_subscription(BehaviorTreeLog, self.bt_log_topic, self._bt_log_cb, 20)
        robot_cfg = cfg.get("robot", {})
        self.robot_entity_name = str(robot_cfg.get("entity_name", "waffle"))
        self.robot_reset_service = str(robot_cfg.get("reset_service", "/gazebo/set_entity_state"))

        self.set_entity_state_cli = self.create_client(
            SetEntityState, self.robot_reset_service
        )
        while not self.set_entity_state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for {self.robot_reset_service} ...")

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.nav_through_poses_client = ActionClient(self, NavigateThroughPoses, "navigate_through_poses")
        self.clear_global_costmap_cli = self.create_client(
            ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap")
        self.clear_local_costmap_cli = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap")
        self.latest_amcl_pose = None
        self.latest_amcl_yaw = None
        self.detector_node_name = cfg.get("detector", {}).get("node_name", "/unexpected_obstacle_detector")
        
        self.reset_metrics()

    def reset_metrics(self):
        self.path_length_m = 0.0
        self._last_xy = None
        self.started_tracking = False
        self.collision_flag = False
        self.recovery_count = 0
        self._seen_recovery_tokens = set()

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.latest_amcl_pose = (x, y)
        self.latest_amcl_yaw = yaw

        if not self.started_tracking:
            self._last_xy = (x, y)
            self._last_traj_t = None
            return

        if self._last_xy is None:
            self._last_xy = (x, y)
            self._last_traj_t = None
            return

        dx = x - self._last_xy[0]
        dy = y - self._last_xy[1]
        ds = math.sqrt(dx * dx + dy * dy)
        if ds >= self.path_eps:
            self.path_length_m += ds
            self._last_xy = (x, y)

        now = time.time()
        if not hasattr(self, '_last_traj_t') or self._last_traj_t is None:
            self._last_traj_t = now
            self._last_traj_xy = (x, y)
            if not hasattr(self, 'trajectory'):
                self.trajectory = []
            self.trajectory.append({'x': x, 'y': y, 't': now, 'speed': 0.0})
        elif now - self._last_traj_t >= 0.2:
            dt = now - self._last_traj_t
            ddx = x - self._last_traj_xy[0]
            ddy = y - self._last_traj_xy[1]
            speed = math.sqrt(ddx**2 + ddy**2) / dt
            self.trajectory.append({'x': x, 'y': y, 't': now, 'speed': speed})
            self._last_traj_t = now
            self._last_traj_xy = (x, y)

    def _ros2_param_set(self, node_name: str, param_name: str, value):
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        else:
            value_str = str(value)

        cmd = ["ros2", "param", "set", node_name, param_name, value_str]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5.0,
            )
            if result.returncode != 0:
                self.get_logger().warn(
                    f"[PARAM] Failed: {' '.join(cmd)}\n{result.stdout}"
                )
                return False
            self.get_logger().info(f"[PARAM] {' '.join(cmd)}")
            return True
        except Exception as e:
            self.get_logger().warn(f"[PARAM] Exception while setting {param_name}: {e}")
            return False


    def set_detector_method(self, method_name: str, freeze_learning: bool = False, method_cfg: dict = {}):
        enable_detector = method_name not in ("ros2_default",)
        approval_mode = "auto"
        enable_postrun = enable_detector and (not freeze_learning)

        if method_name == "ours_human":
            approval_mode = "human"
        elif method_name == "ours_auto":
            approval_mode = "auto"
        elif method_name in ("transient", "permanent", "fixed"):
            enable_postrun = False

        if "detector" in method_cfg:
            enable_detector = bool(method_cfg["detector"])
        if "llm_decay_enable" in method_cfg:
            enable_postrun = bool(method_cfg["llm_decay_enable"])

        node_name = self.detector_node_name

        self._ros2_param_set(node_name, "enabled", enable_detector)
        self._ros2_param_set(node_name, "llm_decay_enable", enable_postrun)
        self._ros2_param_set(node_name, "llm_decay_approval_mode", approval_mode)

        if "fixed_ttl_s" in method_cfg:
            fixed_ttl = float(method_cfg["fixed_ttl_s"])
            self._ros2_param_set(node_name, "default_cost_ttl_s", fixed_ttl)
            self._ros2_param_set(node_name, "vlm_enable", False)
            self._ros2_param_set(node_name, "speed_classifier_enable", False)
            self.get_logger().info(
                f"[METHOD] fixed_ttl_s={fixed_ttl}, vlm=off, speed_classifier=off"
            )
        else:
            self._ros2_param_set(node_name, "vlm_enable", True)
            self._ros2_param_set(node_name, "speed_classifier_enable", True)

        self.get_logger().info(
            f"[METHOD] detector={enable_detector} postrun={enable_postrun} approval={approval_mode}"
        )

    def _scan_cb(self, msg: LaserScan):
        finite = [r for r in msg.ranges if math.isfinite(r)]
        if not finite:
            return
        if min(finite) < self.collision_range_threshold:
            self.collision_flag = True

    def _bt_log_cb(self, msg: BehaviorTreeLog):
        for ev in msg.event_log:
            name = (ev.node_name or "").lower()
            current = (ev.current_status or "").lower()

            is_recovery_like = any(
                token in name
                for token in [
                    "recovery",
                    "spin",
                    "backup",
                    "back_up",
                    "wait",
                    "clear",
                    "assisted",
                    "roundrobin",
                ]
            )
            if is_recovery_like and current == "running":
                token = f"{ev.timestamp.sec}-{ev.timestamp.nanosec}-{ev.node_name}"
                if token not in self._seen_recovery_tokens:
                    self._seen_recovery_tokens.add(token)
                    self.recovery_count += 1

    def publish_initial_pose(self, x: float, y: float, yaw: float):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.frame_id
        q = yaw_to_quat(yaw)
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.06853891945200942

        self.get_logger().info(
            f"[INITPOSE] publish map=({x:.2f}, {y:.2f}, {yaw:.2f}) topic={self.initialpose_topic}"
        )

        for _ in range(5):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.initialpose_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

    def reset_robot_in_gazebo(self, x: float, y: float, yaw: float, z: float = 0.0) -> bool:
        req = SetEntityState.Request()
        state = EntityState()
        state.name = self.robot_entity_name
        state.reference_frame = "world"

        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)

        q = yaw_to_quat(yaw)
        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]

        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        req.state = state

        future = self.set_entity_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done() or future.result() is None:
            self.get_logger().error(
                f"[RESET] No response from {self.robot_reset_service} for entity={self.robot_entity_name}"
            )
            return False

        resp = future.result()
        if not resp.success:
            self.get_logger().error(
                f"[RESET] Failed to reset Gazebo robot pose for entity={self.robot_entity_name}"
            )
            return False

        self.get_logger().info(
            f"[RESET] Gazebo robot reset: entity={self.robot_entity_name} pose=({x:.2f}, {y:.2f}, {yaw:.2f})"
        )
        return True

    def clear_costmaps(self, timeout_sec: float = 3.0):
        for name, cli in [
            ("global", self.clear_global_costmap_cli),
            ("local",  self.clear_local_costmap_cli),
        ]:
            if not cli.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().warn(f"[CLEAR] {name} costmap service not available")
                continue
            future = cli.call_async(ClearEntireCostmap.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            if future.done() and future.result() is not None:
                self.get_logger().info(f"[CLEAR] {name} costmap cleared")
            else:
                self.get_logger().warn(f"[CLEAR] {name} costmap clear failed")

    def wait_for_nav_servers(self):
        self.get_logger().info("Waiting for navigate_to_pose server...")
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info("Waiting for navigate_through_poses server...")
        self.nav_through_poses_client.wait_for_server()

    def run_navigate_to_pose(self, goal_xyyaw: List[float], timeout_s: float) -> Tuple[bool, float, str]:
        goal_msg = NavigateToPose.Goal()
        stamp = self.get_clock().now().to_msg()
        goal_msg.pose = pose_stamped_xyyaw(
            goal_xyyaw[0], goal_xyyaw[1], goal_xyyaw[2], stamp, self.frame_id
        )

        self.get_logger().info(
            f"[GOAL_DBG] frame_id={goal_msg.pose.header.frame_id} "
            f"goal=({goal_msg.pose.pose.position.x:.2f}, "
            f"{goal_msg.pose.pose.position.y:.2f}) "
            f"quat=({goal_msg.pose.pose.orientation.x:.3f}, "
            f"{goal_msg.pose.pose.orientation.y:.3f}, "
            f"{goal_msg.pose.pose.orientation.z:.3f}, "
            f"{goal_msg.pose.pose.orientation.w:.3f})"
        )
        
        t0 = time.time()
        send_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                f"[GOAL] navigate_to_pose rejected: goal=({goal_xyyaw[0]:.2f}, {goal_xyyaw[1]:.2f}, {goal_xyyaw[2]:.2f})"
            )
            return False, 0.0, "goal_rejected"

        result_future = goal_handle.get_result_async()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            dt = time.time() - t0
            if result_future.done():
                res = result_future.result()
                status = res.status
                if status == GoalStatus.STATUS_SUCCEEDED:
                    return True, dt, "succeeded"
                return False, dt, f"status_{status}"

            if dt > timeout_s:
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                return False, dt, "timeout"

        return False, time.time() - t0, "interrupted"

    def run_navigate_through_poses(
        self,
        poses: List[List[float]],
        timeout_s: float,
        waypoint_reached_cb=None,
        waypoint_tol_m: float = 2.0,
    ) -> Tuple[bool, float, str]:
        goal_msg = NavigateThroughPoses.Goal()
        stamp = self.get_clock().now().to_msg()
        goal_msg.poses = [pose_stamped_xyyaw(p[0], p[1], p[2], stamp, self.frame_id) for p in poses]
        t0 = time.time()
        send_future = self.nav_through_poses_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(
                f"[GOAL] navigate_through_poses rejected: poses={poses}"
            )
            return False, 0.0, "goal_rejected"

        result_future = goal_handle.get_result_async()
        triggered_waypoints = set()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            dt = time.time() - t0
            if result_future.done():
                res = result_future.result()
                status = res.status
                if status == GoalStatus.STATUS_SUCCEEDED:
                    return True, dt, "succeeded"
                return False, dt, f"status_{status}"

            if waypoint_reached_cb is not None:
                if self.latest_amcl_pose is None:
                    rclpy.spin_once(self, timeout_sec=0.05)
                    continue
                rx, ry = self.latest_amcl_pose
                for i, p in enumerate(poses):
                    if i in triggered_waypoints:
                        continue
                    dist = math.hypot(rx - p[0], ry - p[1])
                    if dist < waypoint_tol_m:
                        triggered_waypoints.add(i)
                        self.get_logger().info(
                            f"[WAYPOINT] reached idx={i} pos=({p[0]:.1f},{p[1]:.1f}) dist={dist:.2f}m"
                        )
                        waypoint_reached_cb(i)

            if dt > timeout_s:
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                return False, dt, "timeout"

        return False, time.time() - t0, "interrupted"

    def map_to_world(self, x_map: float, y_map: float, yaw_map: float):
        tf_cfg = self.cfg.get("frames", {}).get("map_to_world", {})
        enabled = bool(tf_cfg.get("enabled", True))
        if not enabled:
            self.get_logger().info(
                f"[MAP_TO_WORLD] disabled -> passthrough map=({x_map:.2f}, {y_map:.2f}, {yaw_map:.2f})"
            )
            return float(x_map), float(y_map), float(yaw_map)

        dx = float(tf_cfg.get("dx", 0.0))
        dy = float(tf_cfg.get("dy", 0.0))
        dyaw = float(tf_cfg.get("dyaw", 0.0))

        c = math.cos(dyaw)
        s = math.sin(dyaw)

        x_world = c * x_map - s * y_map + dx
        y_world = s * x_map + c * y_map + dy
        yaw_world = yaw_map + dyaw

        self.get_logger().info(
            f"[MAP_TO_WORLD] map=({x_map:.2f}, {y_map:.2f}, {yaw_map:.2f}) "
            f"-> world=({x_world:.2f}, {y_world:.2f}, {yaw_world:.2f}) "
            f"using dx={dx:.2f}, dy={dy:.2f}, dyaw={dyaw:.2f}"
        )
        return x_world, y_world, yaw_world
    
    def wait_for_amcl_near(
        self,
        target_x: float,
        target_y: float,
        target_yaw: float,
        pos_tol: float = 0.30,
        yaw_tol: float = 0.35,
        timeout_s: float = 8.0,
    ) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_amcl_pose is None or self.latest_amcl_yaw is None:
                continue

            ax, ay = self.latest_amcl_pose
            ayaw = self.latest_amcl_yaw

            pos_err = math.hypot(ax - target_x, ay - target_y)
            yaw_err = math.atan2(
                math.sin(ayaw - target_yaw),
                math.cos(ayaw - target_yaw),
            )
            yaw_err = abs(yaw_err)

            self.get_logger().info(
                f"[AMCL_CHECK] target=({target_x:.2f}, {target_y:.2f}, {target_yaw:.2f}) "
                f"actual=({ax:.2f}, {ay:.2f}, {ayaw:.2f}) "
                f"pos_err={pos_err:.3f} yaw_err={yaw_err:.3f}"
            )

            if pos_err <= pos_tol and yaw_err <= yaw_tol:
                self.get_logger().info("[AMCL_CHECK] aligned")
                return True

        self.get_logger().warn("[AMCL_CHECK] timeout")
        return False

def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def write_jsonl(path: Path, row: Dict):
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clear_file_if_exists(path: Path):
    if path.exists():
        path.unlink()


def reset_runtime_files(cfg: Dict):
    paths = cfg["paths"]
    for k in ["mission_summary_path", "mission_summary_out_path"]:
        clear_file_if_exists(expand_path(paths[k]))


def copy_seed_if_present(seed_path: Optional[str], target_path: str):
    if not seed_path:
        return
    src = expand_path(seed_path)
    dst = expand_path(target_path)
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def parse_repeat_count(summary: Optional[Dict]) -> int:
    if not summary:
        return 0
    events = summary.get("events", []) or []
    max_repeat = 0
    for ev in events:
        rc = int(ev.get("tag_repeat_count_in_mission", 0) or 0)
        if rc > max_repeat:
            max_repeat = rc
    return max(0, max_repeat - 1)


def parse_observed_tags(summary: Optional[Dict]) -> List[str]:
    if not summary:
        return []
    tags = []
    for ev in summary.get("events", []) or []:
        tag = ev.get("vlm_tag_key")
        if tag:
            tags.append(tag)
    return sorted(set(tags))


def read_decay_table_ttls(decay_path: Path, tags: List[str]) -> Dict[str, float]:
    out = {}
    d = load_json(decay_path) or {}
    for tag in tags:
        entry = d.get(tag, {})
        if isinstance(entry, dict):
            ttl = entry.get("ttl")
        else:
            ttl = None
        if ttl is not None:
            try:
                out[tag] = float(ttl)
            except Exception:
                pass
    return out


def mean_abs_ttl_error(pred: Dict[str, float], gt: Dict[str, float], observed_tags: List[str]) -> Optional[float]:
    errs = []
    for tag in observed_tags:
        if tag in gt and tag in pred:
            errs.append(abs(float(pred[tag]) - float(gt[tag])))
    if not errs:
        return None
    return sum(errs) / len(errs)


def launch_obstacle_controller(node: ExperimentNode, cfg: Dict, scenario_cfg: Dict) -> subprocess.Popen:
    obs = scenario_cfg["obstacle"]
    controller_script = expand_path(cfg["obstacle"]["controller_script"])
    sdf_path = expand_path(obs.get("sdf_path", cfg["obstacle"]["sdf_path"]))

    obs_frame = str(obs.get("frame", "map")).lower()

    if obs_frame == "world":
        swx, swy = float(obs["standby"][0]), float(obs["standby"][1])
        ewx, ewy = float(obs["enter"][0]), float(obs["enter"][1])
        xwx, xwy = float(obs["exit"][0]), float(obs["exit"][1])
    else:
        swx, swy, _ = node.map_to_world(obs["standby"][0], obs["standby"][1], 0.0)
        ewx, ewy, _ = node.map_to_world(obs["enter"][0], obs["enter"][1], 0.0)
        xwx, xwy, _ = node.map_to_world(obs["exit"][0], obs["exit"][1], 0.0)

    node.get_logger().info(
        f"[OBS] frame={obs_frame} standby=({swx:.2f}, {swy:.2f}) "
        f"enter=({ewx:.2f}, {ewy:.2f}) exit=({xwx:.2f}, {xwy:.2f})"
    )

    cmd = [
        sys.executable,
        str(controller_script),
        "--ros-args",
        "-p", f"entity_name:={obs['entity_name']}",
        "-p", f"sdf_path:={str(sdf_path)}",
        "-p", f"start_delay_s:={float(obs['start_delay_s'])}",
        "-p", f"speed_mps:={float(obs['speed_mps'])}",
        "-p", f"standby_x:={swx}",
        "-p", f"standby_y:={swy}",
        "-p", f"standby_z:={float(obs['standby'][2])}",
        "-p", f"enter_x:={ewx}",
        "-p", f"enter_y:={ewy}",
        "-p", f"enter_z:={float(obs['enter'][2])}",
        "-p", f"exit_x:={xwx}",
        "-p", f"exit_y:={xwy}",
        "-p", f"exit_z:={float(obs['exit'][2])}",
    ]

    pair_mode = bool(obs.get("pair_mode", False))
    if pair_mode:
        pair_sdf_path = expand_path(obs["pair_sdf_path"])
        pair_entity_name = str(obs.get("pair_entity_name", obs["entity_name"] + "_2"))
        pair_offset_m = float(obs.get("pair_offset_m", 1.2))
        cmd += [
            "-p", "pair_mode:=true",
            "-p", f"pair_sdf_path:={str(pair_sdf_path)}",
            "-p", f"pair_entity_name:={pair_entity_name}",
            "-p", f"pair_offset_m:={pair_offset_m}",
        ]
        node.get_logger().info(
            f"[OBS] pair_mode=True entity2={pair_entity_name} "
            f"sdf2={pair_sdf_path} offset={pair_offset_m}m"
        )

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.5)
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        err = proc.stderr.read() if proc.stderr else ""
        node.get_logger().error(
            f"[OBSTACLE] controller exited immediately rc={proc.returncode}"
            f"\nstdout: {out[:300]}\nstderr: {err[:300]}"
        )

    gt_info = {
        "entity_name":  entity_name,
        "corridor_id":  corridor_cfg["id"],
        "gt_tag":       compound_tag,
        "gt_tag_type":  tag_type,
        "gt_speed_cls": speed_cls,
        "gt_speed_mps": speed_mps,
    }
    return proc, entity_name, gt_info


def stop_process_tree(proc: Optional[subprocess.Popen]):
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=3.0)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            proc.kill()


OBSTACLE_TYPES = {
    "person":   "exp_person.sdf",
    "forklift": "exp_mecanum.sdf",
}

_DEFAULT_SPEED_MAP = {
    "person:slow": 0.05, "person:fast": 0.10,
    "forklift:slow": 0.03, "forklift:fast": 0.07,
    "cart:slow": 0.04, "cart:fast": 0.08,
}
_DEFAULT_APPROACH_MAP = {"person": 0.20, "forklift": 0.15, "cart": 0.16}
_DEFAULT_SDF_MAP = {
    "person":   "exp_person.sdf",
    "forklift": "exp_mecanum.sdf",
    "cart":     "exp_cart.sdf",
}


def _build_obstacle_maps(cfg: Dict):
    obs_types = cfg.get("obstacle_types", {})
    sdf_map:     Dict[str, str]   = {}
    speed_map:   Dict[str, float] = {}
    approach_map: Dict[str, float] = {}
    for tag, tc in obs_types.items():
        if "sdf_path" in tc:
            sdf_map[tag] = str(expand_path(tc["sdf_path"]))
        speed_map[f"{tag}:slow"] = float(tc.get("speed_slow_mps", 0.05))
        speed_map[f"{tag}:fast"] = float(tc.get("speed_fast_mps", 0.10))
        approach_map[tag]        = float(tc.get("speed_approach_mps", 0.20))
    for k, v in _DEFAULT_SPEED_MAP.items():
        speed_map.setdefault(k, v)
    for k, v in _DEFAULT_APPROACH_MAP.items():
        approach_map.setdefault(k, v)
    for k, v in _DEFAULT_SDF_MAP.items():
        sdf_map.setdefault(k, str(expand_path(f"~/STeP_Cost/experiment_suite/{v}")))
    return sdf_map, speed_map, approach_map


def launch_corridor_obstacle(
    node: ExperimentNode,
    cfg: Dict,
    corridor_cfg: Dict,
    obstacle_idx: int,
) -> subprocess.Popen:
    import random
    sdf_map, speed_map, approach_map = _build_obstacle_maps(cfg)
    tag_types = list(cfg.get("obstacle_types", {}).keys()) or list(_DEFAULT_SDF_MAP.keys())
    tag_type  = random.choice(tag_types)
    speed_cls = random.choice(["slow", "fast"])
    compound_tag = f"{tag_type}:{speed_cls}"
    speed_mps    = speed_map.get(compound_tag, 0.05)
    approach_mps = approach_map.get(tag_type, 0.20)
    sdf_path     = sdf_map.get(tag_type, str(expand_path("~/STeP_Cost/experiment_suite/exp_person.sdf")))

    script_dir = str(Path(expand_path(cfg["obstacle"]["controller_script"])).parent)
    controller_script = expand_path(cfg["obstacle"]["controller_script"])

    entity_name = f"corridor_{corridor_cfg['id']}_{tag_type}_{obstacle_idx}"

    def m2w(x, y):
        wx, wy, _ = node.map_to_world(x, y, 0.0)
        return wx, wy

    sb_x, sb_y = m2w(corridor_cfg["standby_x"], corridor_cfg["standby_y"])
    en_x, en_y = m2w(corridor_cfg["enter_x"],   corridor_cfg["enter_y"])
    ex_x, ex_y = m2w(corridor_cfg["exit_x"],    corridor_cfg["exit_y"])
    ex2_x, ex2_y = m2w(
        corridor_cfg.get("exit2_x", corridor_cfg["exit_x"]),
        corridor_cfg.get("exit2_y", corridor_cfg["exit_y"])
    )

    rand_ratio = cfg.get("obstacle", {}).get("standby_random_ratio", 0.3)
    if rand_ratio > 0:
        t = random.uniform(0.0, rand_ratio)
        sb_x = sb_x + t * (en_x - sb_x)
        sb_y = sb_y + t * (en_y - sb_y)
        node.get_logger().info(
            f"[CORRIDOR {corridor_cfg['id']}] random standby t={t:.2f} "
            f"→ world=({sb_x:.2f},{sb_y:.2f})"
        )

    node.get_logger().info(
        f"[CORRIDOR {corridor_cfg['id']}] spawn {entity_name} "
        f"tag={compound_tag} speed={speed_mps:.3f}m/s approach={approach_mps:.3f}m/s"
    )

    cmd = [
        sys.executable, str(controller_script),
        "--ros-args",
        "-p", f"entity_name:={entity_name}",
        "-p", f"sdf_path:={sdf_path}",
        "-p", f"start_delay_s:=0.5",
        "-p", f"speed_mps:={speed_mps}",
        "-p", f"speed_approach_mps:={approach_mps}",
        "-p", f"standby_x:={sb_x}",  "-p", f"standby_y:={sb_y}",  "-p", "standby_z:=0.0",
        "-p", f"enter_x:={en_x}",    "-p", f"enter_y:={en_y}",    "-p", "enter_z:=0.0",
        "-p", f"exit_x:={ex_x}",     "-p", f"exit_y:={ex_y}",     "-p", "exit_z:=0.0",
        "-p", f"exit2_x:={ex2_x}",   "-p", f"exit2_y:={ex2_y}",   "-p", "exit2_z:=0.0",
        "-p", f"speed_approach_mps:={approach_mps}",
        "-p", "loop_forever:=false",
        "-p", "despawn_on_exit2:=true",
        "-p", "despawn_on_finish:=false",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    gt_info = {
        "entity_name":  entity_name,
        "corridor_id":  corridor_cfg["id"],
        "gt_tag":       compound_tag,
        "gt_tag_type":  tag_type,
        "gt_speed_cls": speed_cls,
        "gt_speed_mps": speed_mps,
    }
    return proc, entity_name, gt_info


def run_multi_corridor_trial(
    node: ExperimentNode,
    cfg: Dict,
    method_name: str,
    trial_idx: int,
    run_dir: Path,
    freeze_learning: bool = False,
    corridor_procs: Optional[Dict] = None,
) -> tuple:
    scenario_cfg = deepcopy(cfg["scenarios"]["multi_corridor"])
    paths = cfg["paths"]
    corridors = scenario_cfg["corridors"]

    node.set_detector_method(method_name, freeze_learning=freeze_learning,
                             method_cfg=cfg.get("methods", {}).get(method_name, {}))
    reset_runtime_files(cfg)

    if trial_idx == 0:
        maybe_seed_method_state(cfg, method_name)

    timeout_s = float(cfg.get("trial", {}).get("timeout_s", 1200.0))
    settle_s  = float(cfg.get("trial", {}).get("settle_s_after_initialpose", 2.0))

    node.reset_metrics()

    poses = scenario_cfg["poses"]

    if trial_idx == 0 or corridor_procs is None:
        sx, sy, syaw = poses[0]
        wx, wy, wyaw = node.map_to_world(sx, sy, syaw)
        node.reset_robot_in_gazebo(wx, wy, wyaw)
        time.sleep(1.0)
        node.publish_initial_pose(sx, sy, syaw)
        node.wait_for_amcl_near(
            target_x=float(sx), target_y=float(sy), target_yaw=float(syaw),
            pos_tol=0.30, yaw_tol=0.35, timeout_s=8.0,
        )
        time.sleep(settle_s)
        corridor_procs = {}

        obstacle_gt_records: List[Dict] = []
        for c in corridors:
            if c["spawn_trigger_waypoint_idx"] == 0:
                proc, ename, gt_info = launch_corridor_obstacle(
                    node, cfg, c, trial_idx
                )
                corridor_procs[c["id"]] = (proc, ename)
                obstacle_gt_records.append(gt_info)
    else:
        obstacle_gt_records: List[Dict] = []
        sx, sy, syaw = poses[0]
        wx, wy, wyaw = node.map_to_world(sx, sy, syaw)
        node.reset_robot_in_gazebo(wx, wy, wyaw)
        time.sleep(1.0)
        node.publish_initial_pose(sx, sy, syaw)
        node.wait_for_amcl_near(
            target_x=float(sx), target_y=float(sy), target_yaw=float(syaw),
            pos_tol=0.30, yaw_tol=0.35, timeout_s=8.0,
        )
        time.sleep(settle_s)

    node.clear_costmaps()

    def on_waypoint_reached(idx: int):
        for c in corridors:
            if c["spawn_trigger_waypoint_idx"] == idx:
                if idx == 0 and c["id"] in corridor_procs:
                    continue
                proc, ename, gt_info = launch_corridor_obstacle(
                    node, cfg, c, trial_idx
                )
                corridor_procs[c["id"]] = (proc, ename)
                obstacle_gt_records.append(gt_info)

    node.trajectory = []
    node._last_traj_t = None
    node.started_tracking = True
    success = True
    completion_time = 0.0
    status_str = "succeeded"
    t_total = time.time()
    per_segment_timeout = timeout_s / max(1, len(poses) - 1)

    for seg_idx in range(len(poses)):
        on_waypoint_reached(seg_idx)

        if seg_idx == len(poses) - 1:
            break

        next_pose = poses[seg_idx + 1]
        seg_success, seg_time, seg_status = node.run_navigate_to_pose(
            next_pose, per_segment_timeout
        )
        completion_time += seg_time
        node.get_logger().info(
            f"[SEG] {seg_idx}→{seg_idx+1} "
            f"goal=({next_pose[0]:.1f},{next_pose[1]:.1f}) "
            f"success={seg_success} time={seg_time:.1f}s"
        )
        for c in corridors:
            if c["spawn_trigger_waypoint_idx"] == seg_idx:
                cid = c["id"]
                if cid in corridor_procs and corridor_procs[cid][0].poll() is not None:
                    corridor_procs.pop(cid)
                    node.get_logger().info(f"[CORRIDOR {cid}] controller finished after segment {seg_idx}")

        if not seg_success:
            success = False
            status_str = seg_status
            break

    node.started_tracking = False

    postrun_wait(cfg, method_name, freeze_learning)

    mission_summary_path     = expand_path(paths["mission_summary_path"])
    mission_summary_out_path = expand_path(paths["mission_summary_out_path"])
    decay_table_path         = expand_path(paths["decay_table_path"])

    ms  = load_json(mission_summary_path)
    mso = load_json(mission_summary_out_path) or ms

    observed_tags = parse_observed_tags(mso)
    pred_ttls = read_decay_table_ttls(decay_table_path, observed_tags)
    gt_ttls   = {k: float(v) for k, v in scenario_cfg.get("gt_ttl_s_by_tag", {}).items()}
    ttl_err   = mean_abs_ttl_error(pred_ttls, gt_ttls, observed_tags)

    collision_free_success = bool(success and not node.collision_flag)
    repeat_count = parse_repeat_count(mso)

    trial_record = {
        "timestamp": time.time(),
        "method": method_name,
        "scenario": "multi_corridor",
        "trial_idx": int(trial_idx),
        "freeze_learning": bool(freeze_learning),
        "success": bool(success),
        "collision_flag": bool(node.collision_flag),
        "collision_free_success": bool(collision_free_success),
        "status": status_str,
        "completion_time_s": float(completion_time),
        "path_length_m": float(node.path_length_m),
        "recovery_count": int(node.recovery_count),
        "reencounter_count": int(repeat_count),
        "observed_tags": observed_tags,
        "predicted_ttls": pred_ttls,
        "gt_ttls": gt_ttls,
        "ttl_error_mean_abs_s": ttl_err,
    }

    snap_dir = run_dir / "snapshots" / method_name / "multi_corridor" / f"trial_{trial_idx:03d}"
    ensure_dir(snap_dir)
    copy_if_exists(mission_summary_path, snap_dir / "mission_summary.json")
    copy_if_exists(decay_table_path, snap_dir / "decay_table.json")

    if hasattr(node, 'trajectory') and node.trajectory:
        with open(snap_dir / "trajectory.json", "w", encoding="utf-8") as f:
            json.dump({"trajectory": node.trajectory}, f, indent=2)
        node.get_logger().info(
            f"[TRAJ] saved {len(node.trajectory)} points → {snap_dir / 'trajectory.json'}"
        )

    write_jsonl(run_dir / "trial_records.jsonl", trial_record)
    node.get_logger().info(
        f"[TRIAL] method={method_name} trial={trial_idx} "
        f"success={success} time={completion_time:.1f}s "
        f"ttl_err={ttl_err}"
    )
    return trial_record, corridor_procs


def maybe_seed_method_state(cfg: Dict, method_name: str):
    method_cfg = cfg.get("methods", {}).get(method_name, {})
    paths = cfg["paths"]

    copy_seed_if_present(method_cfg.get("decay_table_seed"), paths["decay_table_path"])
    copy_seed_if_present(method_cfg.get("archive_seed"), paths["archive_path"])


def postrun_wait(cfg: Dict, method_name: str, freeze_learning: bool):
    if method_name == "ros2_default":
        return
    if freeze_learning:
        return
    wait_s = float(cfg.get("trial", {}).get("postrun_wait_s", 12.0))
    time.sleep(wait_s)


def run_single_trial(
    node: ExperimentNode,
    cfg: Dict,
    method_name: str,
    scenario_name: str,
    trial_idx: int,
    run_dir: Path,
    freeze_learning: bool = False,
    obstacle_proc=None,
) -> tuple:
    scenario_cfg = deepcopy(cfg["scenarios"][scenario_name])
    paths = cfg["paths"]

    node.set_detector_method(method_name, freeze_learning=freeze_learning,
                             method_cfg=cfg.get("methods", {}).get(method_name, {}))
    reset_runtime_files(cfg)

    if trial_idx == 0:
        maybe_seed_method_state(cfg, method_name)

    timeout_s = float(cfg.get("trial", {}).get("timeout_s", 120.0))
    settle_s = float(cfg.get("trial", {}).get("settle_s_after_initialpose", 2.0))

    node.reset_metrics()

    if scenario_cfg["mode"] == "navigate_to_pose":
        sx, sy, syaw = scenario_cfg["start"]
    else:
        sx, sy, syaw = scenario_cfg["poses"][0]

    if trial_idx == 0:
        wx, wy, wyaw = node.map_to_world(sx, sy, syaw)
        print(f"[DBG_START_MAP] ({sx:.2f}, {sy:.2f}, {syaw:.2f})")
        print(f"[DBG_START_WORLD] ({wx:.2f}, {wy:.2f}, {wyaw:.2f})")
        ok = node.reset_robot_in_gazebo(wx, wy, wyaw)
        time.sleep(1.0)
        node.publish_initial_pose(sx, sy, syaw)
        aligned = node.wait_for_amcl_near(
            target_x=float(sx),
            target_y=float(sy),
            target_yaw=float(syaw),
            pos_tol=0.30,
            yaw_tol=0.35,
            timeout_s=8.0,
        )
        print(f"[DBG_AMCL_ALIGN] {aligned}")
        time.sleep(settle_s)
        if not ok:
            print(f"[WARN] Gazebo reset failed for start pose")
        obstacle_proc = launch_obstacle_controller(node, cfg, scenario_cfg)
        time.sleep(1.0)
        if obstacle_proc.poll() is not None:
            out = obstacle_proc.stdout.read() if obstacle_proc.stdout else ""
            print("[OBSTACLE_CONTROLLER_EXITED_EARLY]")
            print(out)
    else:
        print(f"[CONTINUOUS] trial_idx={trial_idx} skipping reset/spawn, obstacle continues")
    node.started_tracking = True
    t0 = time.time()

    node.clear_costmaps()

    if scenario_cfg["mode"] == "navigate_to_pose":
        success, completion_time, status_str = node.run_navigate_to_pose(scenario_cfg["goal"], timeout_s)
    else:
        success, completion_time, status_str = node.run_navigate_through_poses(scenario_cfg["poses"], timeout_s)

    node.started_tracking = False
    postrun_wait(cfg, method_name, freeze_learning)

    mission_summary_path = expand_path(paths["mission_summary_path"])
    mission_summary_out_path = expand_path(paths["mission_summary_out_path"])
    decay_table_path = expand_path(paths["decay_table_path"])

    ms = load_json(mission_summary_path)
    mso = load_json(mission_summary_out_path) or ms

    observed_tags = parse_observed_tags(mso)
    pred_ttls = read_decay_table_ttls(decay_table_path, observed_tags)
    gt_ttls = {k: float(v) for k, v in scenario_cfg.get("gt_ttl_s_by_tag", {}).items()}
    ttl_err = mean_abs_ttl_error(pred_ttls, gt_ttls, observed_tags)

    expected_shortest_path_m = float(scenario_cfg.get("expected_shortest_path_m", 0.0))
    normalized_path_length = None
    if expected_shortest_path_m > 1e-6:
        normalized_path_length = node.path_length_m / expected_shortest_path_m

    collision_free_success = bool(success and (not node.collision_flag))
    repeat_count = parse_repeat_count(mso)

    trial_record = {
        "timestamp": time.time(),
        "method": method_name,
        "scenario": scenario_name,
        "trial_idx": int(trial_idx),
        "freeze_learning": bool(freeze_learning),
        "success": bool(success),
        "collision_flag": bool(node.collision_flag),
        "collision_free_success": bool(collision_free_success),
        "status": status_str,
        "completion_time_s": float(completion_time),
        "path_length_m": float(node.path_length_m),
        "normalized_path_length": normalized_path_length,
        "recovery_count": int(node.recovery_count),
        "reencounter_count": int(repeat_count),
        "observed_tags": observed_tags,
        "predicted_ttls": pred_ttls,
        "gt_ttls": gt_ttls,
        "ttl_error_mean_abs_s": ttl_err,
    }

    snap_dir = run_dir / "snapshots" / method_name / scenario_name / f"trial_{trial_idx:03d}"
    ensure_dir(snap_dir)
    copy_if_exists(mission_summary_path, snap_dir / "mission_summary.json")
    copy_if_exists(mission_summary_out_path, snap_dir / "mission_summary_out.json")
    copy_if_exists(decay_table_path, snap_dir / "decay_table.json")

    write_jsonl(run_dir / "results.jsonl", trial_record)

    print(
        f"[TRIAL] method={method_name} scenario={scenario_name} idx={trial_idx} "
        f"success={success} collision_free={collision_free_success} "
        f"path={node.path_length_m:.2f}m time={completion_time:.2f}s "
        f"recovery={node.recovery_count} reencounter={repeat_count} ttl_err={ttl_err}"
    )

    return trial_record, obstacle_proc


def resolve_methods_and_scenarios(cfg: Dict, args):
    if args.methods:
        methods = args.methods
    else:
        methods = cfg["experiments"]["learning"]["methods"] + cfg["experiments"]["evaluation"]["methods"]

    if args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = list(cfg["scenarios"].keys())

    return methods, scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to evaluation_config.yaml")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--freeze-learning", action="store_true")
    parser.add_argument("--repeats", type=int, default=None, help="Override repeats per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"[SEED] random seed={args.seed}")

    cfg = yaml.safe_load(expand_path(args.config).read_text(encoding="utf-8"))
    output_root = expand_path(cfg.get("output_root", "~/factory6_experiment_results"))
    run_dir = output_root / f"run_{now_str()}"
    ensure_dir(run_dir)

    cfg["_seed"] = args.seed  
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rclpy.init()
    node = ExperimentNode(cfg)
    node.wait_for_nav_servers()

    methods, scenarios = resolve_methods_and_scenarios(cfg, args)

    if args.repeats is not None:
        repeats = int(args.repeats)
    else:
        if args.freeze_learning:
            repeats = int(cfg["experiments"]["evaluation"]["repeats_per_scenario"])
        else:
            repeats = int(cfg["experiments"]["learning"]["repeats_per_scenario"])

    try:
        for method_name in methods:
            for scenario_name in scenarios:
                if scenario_name == "multi_corridor":
                    corridor_procs = None
                    for trial_idx in range(repeats):
                        _, corridor_procs = run_multi_corridor_trial(
                            node=node,
                            cfg=cfg,
                            method_name=method_name,
                            trial_idx=trial_idx,
                            run_dir=run_dir,
                            freeze_learning=args.freeze_learning,
                            corridor_procs=corridor_procs,
                        )
                        if corridor_procs:
                            node.get_logger().info(
                                f"[TRIAL {trial_idx}] cleaning up {len(corridor_procs)} corridor procs"
                            )
                            for cid, (proc, ename) in list(corridor_procs.items()):
                                stop_process_tree(proc)
                                try:
                                    node.get_logger().info(f"[CLEANUP] deleting {ename} from Gazebo")
                                    subprocess.run(
                                        ["ros2", "service", "call", "/delete_entity",
                                         "gazebo_msgs/srv/DeleteEntity",
                                         f"{{name: '{ename}'}}"],
                                        timeout=3.0, capture_output=True
                                    )
                                except Exception:
                                    pass
                            corridor_procs = None
                    if corridor_procs:
                        for proc, ename in corridor_procs.values():
                            stop_process_tree(proc)
                    corridor_procs = None
                else:
                    obstacle_proc = None
                    for trial_idx in range(repeats):
                        _, obstacle_proc = run_single_trial(
                            node=node,
                            cfg=cfg,
                            method_name=method_name,
                            scenario_name=scenario_name,
                            trial_idx=trial_idx,
                            run_dir=run_dir,
                            freeze_learning=args.freeze_learning,
                            obstacle_proc=obstacle_proc,
                        )
                    stop_process_tree(obstacle_proc)
                    obstacle_proc = None
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f"[DONE] results saved to: {run_dir}")


if __name__ == "__main__":
    main()
