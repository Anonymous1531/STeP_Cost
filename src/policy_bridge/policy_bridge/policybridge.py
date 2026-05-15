from __future__ import annotations

import math
import os
import json
import time
import threading
import traceback
import subprocess
import re
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from uuid import uuid4
from typing import Any, Deque, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters

from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.srv import ClearEntireCostmap
from std_srvs.srv import Trigger
from std_msgs.msg import String
from action_msgs.msg import GoalStatusArray, GoalStatus

import tf2_ros
import tf_transformations

try:
    from cv_bridge import CvBridge
    import cv2
    _CAPTURE_OK = True
except Exception:
    CvBridge = None
    cv2 = None
    _CAPTURE_OK = False

try:
    import numpy as np
    from nav_msgs.msg import Odometry
    _SPEED_OK = True
except Exception:
    np = None
    _SPEED_OK = False



class _KalmanTrack:
    _id_counter = 0
    def __init__(self, x, y, dt=0.2):
        _KalmanTrack._id_counter += 1
        self.track_id = _KalmanTrack._id_counter
        self.miss_count = 0
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 1.0
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 0.5
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.miss_count = 0
    @property
    def pos(self): return float(self.x[0]), float(self.x[1])
    @property
    def vel(self): return float(self.x[2]), float(self.x[3])
    @property
    def speed(self): return math.hypot(self.x[2], self.x[3])


class _SingleGMM:
    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
        self._samples: List[float] = []
        self._mu    = [0.0, 0.0]
        self._sigma = [1.0, 1.0]
        self._pi    = [0.5, 0.5]
        self._fitted = False

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def is_ready(self) -> bool:
        return self._fitted

    def add_sample(self, speed_mps: float):
        self._samples.append(speed_mps)
        if len(self._samples) >= self.min_samples:
            self._fit()

    def _fit(self):
        import math as _math
        X = self._samples
        n = len(X)
        if n < 2:
            return
        sorted_X = sorted(X)
        mid = sorted_X[n // 2]
        slow_vals = [x for x in X if x <= mid]
        fast_vals = [x for x in X if x > mid]
        mu0 = sum(slow_vals) / max(1, len(slow_vals))
        mu1 = sum(fast_vals) / max(1, len(fast_vals))
        if mu0 >= mu1:
            mu0, mu1 = sorted_X[n//4], sorted_X[3*n//4]
        sigma0 = sigma1 = max(0.01, (mu1 - mu0) / 4)
        pi0 = pi1 = 0.5

        def gauss(x, mu, sigma):
            return _math.exp(-0.5*((x-mu)/sigma)**2) / (sigma*_math.sqrt(2*_math.pi)+1e-12)

        for _ in range(50):
            r0, r1 = [], []
            for x in X:
                p0 = pi0 * gauss(x, mu0, sigma0)
                p1 = pi1 * gauss(x, mu1, sigma1)
                tot = p0 + p1 + 1e-12
                r0.append(p0/tot); r1.append(p1/tot)
            n0 = sum(r0)+1e-12; n1 = sum(r1)+1e-12
            mu0n = sum(r*x for r,x in zip(r0,X))/n0
            mu1n = sum(r*x for r,x in zip(r1,X))/n1
            s0n = _math.sqrt(sum(r*(x-mu0n)**2 for r,x in zip(r0,X))/n0)+1e-4
            s1n = _math.sqrt(sum(r*(x-mu1n)**2 for r,x in zip(r1,X))/n1)+1e-4
            if abs(mu0n-mu0)<1e-5 and abs(mu1n-mu1)<1e-5:
                mu0,mu1,sigma0,sigma1,pi0,pi1 = mu0n,mu1n,s0n,s1n,n0/n,n1/n
                break
            mu0,mu1,sigma0,sigma1,pi0,pi1 = mu0n,mu1n,s0n,s1n,n0/n,n1/n
        if mu0 > mu1:
            mu0,mu1 = mu1,mu0
            sigma0,sigma1 = sigma1,sigma0
            pi0,pi1 = pi1,pi0
        self._mu=[mu0,mu1]; self._sigma=[sigma0,sigma1]; self._pi=[pi0,pi1]
        self._fitted = True

    def predict(self, speed_mps: float) -> str:
        if not self._fitted:
            return "fast"
        import math as _math
        def gauss(x,mu,sigma):
            return _math.exp(-0.5*((x-mu)/sigma)**2)/(sigma*_math.sqrt(2*_math.pi)+1e-12)
        p_slow = self._pi[0]*gauss(speed_mps, self._mu[0], self._sigma[0])
        p_fast = self._pi[1]*gauss(speed_mps, self._mu[1], self._sigma[1])
        return "slow" if p_slow > p_fast else "fast"

    def summary(self) -> str:
        if not self._fitted:
            return f"not_fitted(n={self.n_samples})"
        return (f"n={self.n_samples} "
                f"slow_mu={self._mu[0]:.3f}(σ={self._sigma[0]:.3f}) "
                f"fast_mu={self._mu[1]:.3f}(σ={self._sigma[1]:.3f})")

    def to_dict(self) -> dict:
        return {"samples": list(self._samples), "fitted": self._fitted,
                "mu": list(self._mu), "sigma": list(self._sigma), "pi": list(self._pi)}

    @classmethod
    def from_dict(cls, d: dict, min_samples: int = 10) -> "_SingleGMM":
        obj = cls(min_samples)
        obj._samples = list(d.get("samples", []))
        obj._fitted  = bool(d.get("fitted", False))
        obj._mu      = list(d.get("mu",    [0.0, 0.0]))
        obj._sigma   = list(d.get("sigma", [1.0, 1.0]))
        obj._pi      = list(d.get("pi",    [0.5, 0.5]))
        return obj


class _TagGMM:

    def __init__(self, min_samples: int = 10, save_path: str = ""):
        self.min_samples = min_samples
        self.save_path   = save_path
        self._gmms: Dict[str, _SingleGMM] = {}

    def _get_or_create(self, base_tag: str) -> _SingleGMM:
        if base_tag not in self._gmms:
            self._gmms[base_tag] = _SingleGMM(self.min_samples)
        return self._gmms[base_tag]

    def add_sample(self, base_tag: str, speed_mps: float):
        self._get_or_create(base_tag).add_sample(speed_mps)
        self._save()

    def predict(self, base_tag: str, speed_mps: float) -> str:
        return self._get_or_create(base_tag).predict(speed_mps)

    def is_ready(self, base_tag: str) -> bool:
        return self._gmms.get(base_tag, _SingleGMM()).is_ready

    def n_samples(self, base_tag: str) -> int:
        return self._gmms.get(base_tag, _SingleGMM()).n_samples

    def summary(self) -> str:
        if not self._gmms:
            return "  (empty)"
        return "\n".join(f"  {t}: {g.summary()}" for t, g in self._gmms.items())

    def _save(self):
        if not self.save_path:
            return
        try:
            import os as _os
            _os.makedirs(_os.path.dirname(_os.path.expanduser(self.save_path)), exist_ok=True)
            with open(_os.path.expanduser(self.save_path), "w", encoding="utf-8") as f:
                import json as _json
                _json.dump({t: g.to_dict() for t, g in self._gmms.items()},
                           f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass 

    def load(self):
        if not self.save_path:
            return
        try:
            import os as _os, json as _json
            path = _os.path.expanduser(self.save_path)
            if not _os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            for tag, d in data.items():
                self._gmms[tag] = _SingleGMM.from_dict(d, self.min_samples)
        except Exception as e:
            pass


@dataclass
class ActiveCost:
    cost_id: int
    x: float
    y: float
    ttl_s: float
    expires_at: float
    last_update: float
    event_id: str = ""
    tag_key: str = ""
    tag_group_id: str = ""
    prev_x: float = 0.0
    prev_y: float = 0.0
    prev_t: float = 0.0
    speed_samples: object = None


TAG_KEY_TOKEN_RE = re.compile(r"\s+")

def normalize_tag_key(key: Any) -> str:
    if key is None:
        return ""
    k = str(key).strip()
    if not k:
        return ""
    k = TAG_KEY_TOKEN_RE.split(k, 1)[0]
    k = k.strip().strip(",;").lower()
    if "|" in k:
        if ":" in k:
            k = k.split("|", 1)[0]
        else:
            a, b = k.split("|", 1)
            a = a.strip(); b = b.strip()
            k = f"{a}:{b}" if a and b else (a or b)
    if k.endswith(":still"):
        k = k[:-6] + ":static"
    return k

def extract_vlm_tag_key(vlm_field: Any) -> str:
    if vlm_field is None:
        return ""
    if isinstance(vlm_field, dict):
        tk = vlm_field.get("tag_key") or vlm_field.get("tag")
        return normalize_tag_key(tk) if tk else ""
    if isinstance(vlm_field, str):
        s = vlm_field.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                tk = obj.get("tag_key") or obj.get("tag")
                return normalize_tag_key(tk) if tk else ""
        except Exception:
            if ":" in s or "|" in s:
                return normalize_tag_key(s)
    return ""


class UnexpectedObstacleDetector(Node):

    def _current_mission_id(self) -> str:
        with self._mission_lock:
            return str(self._mission.get("mission_id") or "unknown_mission")

    def _make_tag_group_id(self, tag_key: str) -> str:
        tk = normalize_tag_key(tag_key)
        if not tk:
            return ""
        return f"{self._current_mission_id()}::{tk}"

    def __init__(self):
        super().__init__("unexpected_obstacle_detector")

        self.declare_parameter("enabled", True)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("output_topic", "/object_world_positions")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("plan_topic", "/plan")
        self.declare_parameter("pose_topic", "/amcl_pose")

        self.declare_parameter("gate_on_detour_only", True)
        self.declare_parameter("detour_ratio_threshold", 1.15)
        self.declare_parameter("detour_min_previous_length_m", 0.50)
        self.declare_parameter("detour_hold_s", 8.0)
        self.declare_parameter("detour_cooldown_s", 2.0)

        self.declare_parameter("min_range_m", 0.10)
        self.declare_parameter("max_range_m", 6.0)
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("exclude_unknown", True)
        self.declare_parameter("occupied_margin_cells", 10)
        self.declare_parameter("cluster_dist_thresh", 0.20)
        self.declare_parameter("cluster_min_points", 5)
        self.declare_parameter("min_centroid_robot_distance_m", 0.25)
        self.declare_parameter("confirm_with_previous_scan", True)
        self.declare_parameter("confirm_dist_thresh", 0.30)

        self.declare_parameter("default_cost_ttl_s", 6.0)
        self.declare_parameter("track_merge_dist_m", 0.60)
        self.declare_parameter("maintain_rate_hz", 5.0)
        self.declare_parameter("publish_empty_when_no_active", True)

        self.declare_parameter("enable_global_clear_on_expire", True)
        self.declare_parameter("clear_service", "/global_costmap/clear_entirely_global_costmap")
        self.declare_parameter("clear_service_wait_s", 0.2)

        self.declare_parameter("debug_log_interval_s", 2.0)

        self.declare_parameter("enable_capture", True)
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("frames_to_save", 8)
        self.declare_parameter("sample_hz", 6.0)
        self.declare_parameter("save_root", os.path.expanduser("~/.ros/detour_events"))

        self.declare_parameter("vlm_enable", False)
        self.declare_parameter("vlm_python", "python3")
        self.declare_parameter("vlm_script", os.path.expanduser("~/STeP_Cost/vlm_gemini_v1.py"))
        self.declare_parameter("vlm_model", "gemini-2.5-flash")
        self.declare_parameter(
            "vlm_prompt",
            "Classify the main obstacle causing the detour into exactly one allowed tag: person, vehicle, nav_anomaly, or workzone. Tag definitions: person=any human/pedestrian/worker, vehicle=any wheeled machine (truck/forklift/cart/AGV/pallet), nav_anomaly=unclassified obstacle or navigation issue, workzone=marked work area. Never output any other tag. Evidence is required and must be a short structured phrase using only lowercase letters, numbers, and underscores."
        )
        self.declare_parameter("vlm_timeout_sec", 180.0)
        self.declare_parameter("vlm_result_topic", "/vlm/result")
        self.declare_parameter("vlm_singleflight", True)
        self.declare_parameter("vlm_decay_table", os.path.expanduser("~/.ros/decay_table.json"))
        self.declare_parameter("debug_vlm_stdout_chars", 350)
        self.declare_parameter("debug_vlm_stderr_chars", 350)
        self.declare_parameter("llm_debug_prompt_path", "")
        self.declare_parameter("speed_classifier_enable", True)
        self.declare_parameter("gmm_min_samples", 10)
        self.declare_parameter("gmm_samples_path", os.path.expanduser("~/.ros/gmm_samples.json"))
        self.declare_parameter("corridor_start_x", 0.0)
        self.declare_parameter("corridor_end_x",   32.0)
        self.declare_parameter("corridor_y_centers", [0.0, -4.76, -13.49, -18.17])
        self.declare_parameter("corridor_y_half_width", 2.5) 
        self.declare_parameter("corridor_x_margin", 1.0)  
        self.declare_parameter("speed_threshold_mps", 0.3)
        self.declare_parameter("speed_knn_min_samples", 10)
        self.declare_parameter("speed_cluster_dist", 0.3)
        self.declare_parameter("speed_cluster_min_points", 3)
        self.declare_parameter("speed_max_range_m", 6.0)
        self.declare_parameter("speed_query_window_before_s", 5.0)
        self.declare_parameter("speed_query_window_after_s", 1.0)
        self.declare_parameter("speed_measure_stop_s", 5.0)  
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("llm_stdout_log_chars", 2000)
        self.declare_parameter("llm_stderr_log_chars", 2000)  

        self.declare_parameter("mission_summary_path", os.path.expanduser("~/.ros/mission_summary.json"))
        self.declare_parameter("mission_summary_out_path", os.path.expanduser("~/.ros/mission_summary_out.json"))
        self.declare_parameter("llm_decay_enable", False)
        self.declare_parameter("calibration_mode", False)  
        self.declare_parameter("gmm_freeze", False) 
        self.declare_parameter("llm_decay_python", "python3")
        self.declare_parameter("llm_decay_script", os.path.expanduser("~/STeP_Cost/llm_decay_gemini_v3.py"))
        self.declare_parameter("llm_decay_model", "gemini-2.5-flash")
        self.declare_parameter("llm_decay_result_topic", "/llm_decay/result")
        self.declare_parameter("goal_status_topic", "/navigate_to_pose/_action/status")
        self.declare_parameter("through_poses_status_topic", "/navigate_through_poses/_action/status")
        self.declare_parameter("goal_success_cooldown_s", 5.0)
        self.declare_parameter("llm_decay_singleflight", True)
        self.declare_parameter("llm_decay_rag_enable", False)
        self.declare_parameter("llm_decay_retrieval_archive_path", os.path.expanduser("~/.ros/llm_decay_rag_archive.json"))
        self.declare_parameter("llm_decay_retrieval_max_repeat1_cases", 30)
        self.declare_parameter("llm_decay_append_to_archive", True)
        self.declare_parameter("llm_decay_archive_max_cases", 2000)
        self.declare_parameter("llm_decay_approval_mode", "auto")
        self.declare_parameter("llm_decay_human_approval_threshold_pct", 5.0)
        self.declare_parameter("llm_decay_confidence_threshold", 0.8)  
        self.declare_parameter("llm_decay_auto_approve", True)
        self.declare_parameter("llm_decay_human_approval", False)
        self.declare_parameter("default_decay_tag", "nav_anomaly")
        self.declare_parameter("apply_vlm_decay_online", True)
        self.declare_parameter("start_new_mission_after_llm", True)
        self.declare_parameter("min_online_tag_ttl_s", 5.0)

        self._enabled = bool(self.get_parameter("enabled").value)
        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._map_topic = str(self.get_parameter("map_topic").value)
        self._output_topic = str(self.get_parameter("output_topic").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._plan_topic = str(self.get_parameter("plan_topic").value)
        self._pose_topic = str(self.get_parameter("pose_topic").value)

        self._gate_on_detour_only = bool(self.get_parameter("gate_on_detour_only").value)
        self._detour_ratio_threshold = float(self.get_parameter("detour_ratio_threshold").value)
        self._detour_min_previous_length = float(self.get_parameter("detour_min_previous_length_m").value)
        self._detour_hold_s = float(self.get_parameter("detour_hold_s").value)
        self._detour_cooldown_s = float(self.get_parameter("detour_cooldown_s").value)

        self._min_range = float(self.get_parameter("min_range_m").value)
        self._max_range = float(self.get_parameter("max_range_m").value)
        self._occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self._exclude_unknown = bool(self.get_parameter("exclude_unknown").value)
        self._occupied_margin_cells = int(self.get_parameter("occupied_margin_cells").value)
        self._cluster_dist = float(self.get_parameter("cluster_dist_thresh").value)
        self._cluster_min_points = int(self.get_parameter("cluster_min_points").value)
        self._min_centroid_robot_dist = float(self.get_parameter("min_centroid_robot_distance_m").value)
        self._confirm = bool(self.get_parameter("confirm_with_previous_scan").value)
        self._confirm_dist = float(self.get_parameter("confirm_dist_thresh").value)

        self._default_cost_ttl_s = float(self.get_parameter("default_cost_ttl_s").value)
        self._track_merge_dist = float(self.get_parameter("track_merge_dist_m").value)
        self._maintain_rate_hz = max(0.5, float(self.get_parameter("maintain_rate_hz").value))
        self._publish_empty_when_no_active = bool(self.get_parameter("publish_empty_when_no_active").value)

        self._enable_global_clear_on_expire = bool(self.get_parameter("enable_global_clear_on_expire").value)
        self._clear_service_name = str(self.get_parameter("clear_service").value)
        self._clear_service_wait_s = float(self.get_parameter("clear_service_wait_s").value)

        self._debug_log_interval_s = float(self.get_parameter("debug_log_interval_s").value)

        self._enable_capture = bool(self.get_parameter("enable_capture").value)
        self._camera_topic = str(self.get_parameter("camera_topic").value)
        self._frames_to_save = int(self.get_parameter("frames_to_save").value)
        self._sample_hz = float(self.get_parameter("sample_hz").value)
        self._save_root = os.path.expanduser(str(self.get_parameter("save_root").value))

        self._vlm_enable = bool(self.get_parameter("vlm_enable").value)
        self._vlm_python = str(self.get_parameter("vlm_python").value)
        self._vlm_script = os.path.expanduser(str(self.get_parameter("vlm_script").value))
        self._vlm_model = str(self.get_parameter("vlm_model").value)
        self._vlm_prompt = str(self.get_parameter("vlm_prompt").value)
        self._vlm_timeout_sec = float(self.get_parameter("vlm_timeout_sec").value)
        self._vlm_result_topic = str(self.get_parameter("vlm_result_topic").value)
        self._vlm_singleflight = bool(self.get_parameter("vlm_singleflight").value)
        self._vlm_decay_table = os.path.expanduser(str(self.get_parameter("vlm_decay_table").value))
        self._debug_vlm_stdout_chars = int(self.get_parameter("debug_vlm_stdout_chars").value)
        self._debug_vlm_stderr_chars = int(self.get_parameter("debug_vlm_stderr_chars").value)
        self._llm_debug_prompt_path = str(self.get_parameter("llm_debug_prompt_path").value).strip()
        self._llm_stdout_log_chars = max(200, int(self.get_parameter("llm_stdout_log_chars").value))
        self._llm_stderr_log_chars = max(200, int(self.get_parameter("llm_stderr_log_chars").value))


        self._mission_summary_path = os.path.expanduser(str(self.get_parameter("mission_summary_path").value))
        self._mission_summary_out_path = os.path.expanduser(str(self.get_parameter("mission_summary_out_path").value))
        self._llm_decay_enable = bool(self.get_parameter("llm_decay_enable").value)
        self._calibration_mode = bool(self.get_parameter("calibration_mode").value)
        self._gmm_freeze = bool(self.get_parameter("gmm_freeze").value)
        self._llm_decay_python = str(self.get_parameter("llm_decay_python").value)
        self._llm_decay_script = os.path.expanduser(str(self.get_parameter("llm_decay_script").value))
        self._llm_decay_model = str(self.get_parameter("llm_decay_model").value)
        self._llm_decay_result_topic = str(self.get_parameter("llm_decay_result_topic").value)
        self._goal_status_topic = str(self.get_parameter("goal_status_topic").value)
        self._through_poses_status_topic = str(self.get_parameter("through_poses_status_topic").value)
        self._goal_success_cooldown_s = float(self.get_parameter("goal_success_cooldown_s").value)
        self._llm_decay_singleflight = bool(self.get_parameter("llm_decay_singleflight").value)
        self._llm_decay_rag_enable = bool(self.get_parameter("llm_decay_rag_enable").value)
        self._llm_decay_retrieval_archive_path = os.path.expanduser(str(self.get_parameter("llm_decay_retrieval_archive_path").value))
        self._llm_decay_retrieval_max_repeat1_cases = int(
            self.get_parameter("llm_decay_retrieval_max_repeat1_cases").value
        )
        self._llm_decay_append_to_archive = bool(self.get_parameter("llm_decay_append_to_archive").value)
        self._llm_decay_archive_max_cases = int(self.get_parameter("llm_decay_archive_max_cases").value)
        self._llm_decay_approval_mode = str(self.get_parameter("llm_decay_approval_mode").value).strip().lower()
        if self._llm_decay_approval_mode not in ("auto", "human", "human_all"):
            self.get_logger().warn(f"[LLM] invalid llm_decay_approval_mode={self._llm_decay_approval_mode!r}; fallback to 'auto'")
            self._llm_decay_approval_mode = "auto"
        self._llm_decay_human_approval_threshold_pct = float(self.get_parameter("llm_decay_human_approval_threshold_pct").value)
        self._llm_decay_confidence_threshold = float(self.get_parameter("llm_decay_confidence_threshold").value)
        self._llm_decay_auto_approve = bool(self.get_parameter("llm_decay_auto_approve").value)
        self._llm_decay_human_approval = bool(self.get_parameter("llm_decay_human_approval").value)
        self._default_decay_tag = normalize_tag_key(self.get_parameter("default_decay_tag").value)
        self._apply_vlm_decay_online = bool(self.get_parameter("apply_vlm_decay_online").value)
        self._start_new_mission_after_llm = bool(self.get_parameter("start_new_mission_after_llm").value)
        self._min_online_tag_ttl_s = float(self.get_parameter("min_online_tag_ttl_s").value)

        self._map_msg: Optional[OccupancyGrid] = None
        self._prev_centroids: List[Tuple[float, float]] = []
        self._last_debug_log = 0.0
        self._previous_path_length = 0.0
        self._current_pose_xy: Tuple[float, float] = (0.0, 0.0)
        self._current_yaw: float = 0.0

        self._depth_history: List[Tuple[float, float]] = []  
        self._depth_bbox_history: List[Tuple[float, float]] = []  
        self._depth_lock = threading.Lock()
        self._depth_roi_frac = 0.2
        self._centroid_tracks: Dict[int, Dict] = {}  
        self._centroid_track_id = 0
        self._centroid_track_lock = threading.Lock()
        self._centroid_match_dist = 1.5  
        self._latest_unexpected_centroids: List[Tuple[float, float]] = []  
        self._latest_unexpected_lock = threading.Lock()
        self._speed_stop_active: bool = False 
        self._detour_active_until = 0.0
        self._last_detour_trigger = 0.0

        self._active_costs: Dict[int, ActiveCost] = {}
        self._next_cost_id = 1
        self._last_publish_empty = False
        self._last_scan_stamp = None
        self._clear_inflight = False
        self._clear_pending_republish = False

        self._frame_lock = threading.Lock()
        self._frame_buf: Deque[Tuple[float, any]] = deque(maxlen=max(16, self._frames_to_save * 4))
        self._last_store_time = 0.0
        self._vlm_guard = threading.Lock()
        self._vlm_running = False

        self._event_tag_map: Dict[str, str] = {}
        self._current_event_id: str = ""

        self._speed_classifier_enable = bool(self.get_parameter("speed_classifier_enable").value)
        self._corridor_start_x = float(self.get_parameter("corridor_start_x").value)
        self._corridor_end_x   = float(self.get_parameter("corridor_end_x").value)
        self._corridor_y_centers   = list(self.get_parameter("corridor_y_centers").value)
        self._corridor_y_half_width = float(self.get_parameter("corridor_y_half_width").value)
        self._corridor_x_margin     = float(self.get_parameter("corridor_x_margin").value)
        self._gmm_min_samples  = int(self.get_parameter("gmm_min_samples").value)
        self._gmm_samples_path = str(self.get_parameter("gmm_samples_path").value)
        self._speed_query_window_before_s = float(self.get_parameter("speed_query_window_before_s").value)
        self._speed_query_window_after_s = float(self.get_parameter("speed_query_window_after_s").value)
        self._speed_measure_stop_s = float(self.get_parameter("speed_measure_stop_s").value)
        self._current_speed_class: str = ""
        self._pending_speed_for_gmm: float = -1.0
        self._last_confirmed_base_tag: str = ""
        self._current_obstacle_dist_m: float = -1.0

        if self._speed_classifier_enable and _SPEED_OK:
            self._sc_tracks: Dict[int, _KalmanTrack] = {}
            self._sc_track_lock = threading.Lock()
            self._sc_robot_vx: float = 0.0
            self._sc_robot_vy: float = 0.0
            self._sc_odom_lock = threading.Lock()
            self._sc_speed_history: List[Tuple[float, float]] = []
            self._sc_dist_history: List[Tuple[float, float]] = []
            self._sc_history_lock = threading.Lock()
            self._sc_gmm = _TagGMM(
                min_samples=self._gmm_min_samples,
                save_path=self._gmm_samples_path,
            )
            self._sc_gmm.load()
            self.get_logger().info(
                f"[GMM] loaded from {self._gmm_samples_path}\n{self._sc_gmm.summary()}"
            )
            self._sc_cluster_dist = float(self.get_parameter("speed_cluster_dist").value)
            self._sc_cluster_min  = int(self.get_parameter("speed_cluster_min_points").value)
            self._sc_max_range    = float(self.get_parameter("speed_max_range_m").value)
            self._sc_kalman_dt    = 0.2
            self._sc_track_max_miss = 5
            import queue
            self._sc_scan_queue = queue.Queue(maxsize=3)
            self._sc_worker_thread = threading.Thread(target=self._sc_worker_loop, daemon=True)
            self._sc_worker_thread.start()
            self.create_subscription(
                Odometry,
                str(self.get_parameter("odom_topic").value),
                self._sc_odom_cb, 10
            )
            self.create_timer(self._sc_kalman_dt, self._sc_kalman_predict_cb)
            self.get_logger().info("[SPEED] enabled (TagGMM mode)")
        elif self._speed_classifier_enable and not _SPEED_OK:
            self.get_logger().warn("[SPEED] numpy/Odometry unavailable, disabled")
            self._speed_classifier_enable = False
        self._mission_lock = threading.Lock()
        self._mission: Dict[str, Any] = self._new_mission_template()
        self._seen_goal_success_ids = set()
        self._last_goal_success_time = 0.0
        self._llm_decay_guard = threading.Lock()
        self._llm_decay_running = False
        self._decay_table: Dict[str, Dict[str, float]] = {}

        if self._enable_capture or self._vlm_enable:
            os.makedirs(self._save_root, exist_ok=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.object_pub = self.create_publisher(PoseArray, self._output_topic, 10)
        self.vlm_pub = self.create_publisher(String, self._vlm_result_topic, 10)
        self.llm_decay_pub = self.create_publisher(String, self._llm_decay_result_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._vel_smoother_client = self.create_client(
            SetParameters, '/velocity_smoother/set_parameters'
        )
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_sub = self.create_subscription(OccupancyGrid, self._map_topic, self._map_cb, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.plan_sub = self.create_subscription(Path, self._plan_topic, self._plan_cb, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, self._pose_topic, self._pose_cb, 10)
        self.goal_status_sub = self.create_subscription(GoalStatusArray, self._goal_status_topic, self._goal_status_cb, 10)
        self.through_goal_status_sub = self.create_subscription(GoalStatusArray, self._through_poses_status_topic, self._goal_status_cb, 10)
        if self._enable_capture or self._vlm_enable:
            self.image_sub = self.create_subscription(Image, self._camera_topic, self._image_cb, qos_profile_sensor_data)
        if self._speed_classifier_enable:
            self.depth_sub = self.create_subscription(
                Image, '/camera/depth/image_raw', self._depth_cb, qos_profile_sensor_data
            )
        self.maintain_timer = self.create_timer(1.0 / self._maintain_rate_hz, self._maintain_cb)

        self.clear_client = self.create_client(ClearEntireCostmap, self._clear_service_name)
        self._reset_mission_srv = self.create_service(
            Trigger, '/unexpected_obstacle_detector/reset_mission', self._reset_mission_cb
        )

        state = "enabled" if self._enabled else "disabled"
        gate_state = "detour-gated" if self._gate_on_detour_only else "always-on"
        self.get_logger().info(
            f"UnexpectedObstacleDetector {state}/{gate_state}: scan={self._scan_topic}, map={self._map_topic}, plan={self._plan_topic}, out={self._output_topic}, ttl={self._default_cost_ttl_s:.1f}s"
        )
        self.get_logger().info(
            f"capture={self._enable_capture} camera_topic={self._camera_topic} frames_to_save={self._frames_to_save} sample_hz={self._sample_hz:.1f} "
            f"vlm_enable={self._vlm_enable} vlm_script={self._vlm_script}"
        )

        self._load_decay_table()
        self._save_mission()

    def _image_cb(self, msg: Image):
        if not (_CAPTURE_OK and (self._enable_capture or self._vlm_enable)):
            return
        now = time.time()
        if self._sample_hz > 0 and (now - self._last_store_time) < (1.0 / self._sample_hz):
            return
        self._last_store_time = now
        try:
            img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            try:
                img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if img is not None and len(getattr(img, "shape", [])) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                self._throttled_log(f"[IMG] cv_bridge convert failed: {e}")
                return
        with self._frame_lock:
            self._frame_buf.append((now, img))

    def _map_cb(self, msg: OccupancyGrid):
        self._map_msg = msg

    def _set_velocity_smoother_max(self, vx: float, vy: float, vth: float) -> bool:
        try:
            if not self._vel_smoother_client.wait_for_service(timeout_sec=1.0):
                return False
            req = SetParameters.Request()
            pv = ParameterValue()
            pv.type = ParameterType.PARAMETER_DOUBLE_ARRAY
            pv.double_array_value = [vx, vy, vth]
            param = Parameter()
            param.name = "max_velocity"
            param.value = pv
            req.parameters = [param]
            future = self._vel_smoother_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            return True
        except Exception as e:
            self.get_logger().warn(f"[SPEED_STOP] set_max_velocity failed: {e}")
            return False

    def _is_static_obstacle(self, map_x: float, map_y: float, occ_threshold: int = 50) -> bool:
        msg = getattr(self, '_map_msg', None)
        if msg is None:
            return False
        res = msg.info.resolution
        ox  = msg.info.origin.position.x
        oy  = msg.info.origin.position.y
        col = int((map_x - ox) / res)
        row = int((map_y - oy) / res)
        w, h = msg.info.width, msg.info.height
        if not (0 <= col < w and 0 <= row < h):
            return False
        val = msg.data[row * w + col]
        return val >= occ_threshold

    def _normalize_feedback_text(self, text: Optional[str]) -> str:
        s = (text or "").strip().lower()
        if not s:
            return ""
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-z0-9_ ./:\-]", "", s)
        return s[:300]

    def _prompt_optional_feedback(self) -> str:
        try:
            return input("Optional feedback (press Enter to skip): ").strip()
        except EOFError:
            return ""
        except Exception:
            return ""

    def _new_mission_template(self) -> Dict[str, Any]:
        mid = f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return {
            "mission_id": mid,
            "started_at": time.time(),
            "started_at_local": datetime.now().isoformat(),
            "events": [],
            "tag_groups": {},
        }

    def _rebuild_mission_tag_groups(self):
        summary: Dict[str, Any] = {}
        for ev in self._mission.get("events", []) or []:
            gid = ev.get("tag_group_id")
            if not gid:
                continue
            item = summary.setdefault(gid, {
                "tag_key": ev.get("vlm_tag_key"),
                "count": 0,
                "events": [],
            })
            item["count"] += 1
            item["events"].append(ev.get("event"))
        self._mission["tag_groups"] = summary

    def _merge_close_centroids(self, centroids, merge_dist):
        if not centroids:
            return []

        remaining = centroids[:]
        merged = []

        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            changed = True

            while changed:
                changed = False
                gx = sum(p[0] for p in group) / len(group)
                gy = sum(p[1] for p in group) / len(group)

                keep = []
                for p in remaining:
                    if math.hypot(p[0] - gx, p[1] - gy) <= merge_dist:
                        group.append(p)
                        changed = True
                    else:
                        keep.append(p)
                remaining = keep

            mx = sum(p[0] for p in group) / len(group)
            my = sum(p[1] for p in group) / len(group)
            merged.append((mx, my))

        return merged

    def _save_mission(self):
        try:
            os.makedirs(os.path.dirname(self._mission_summary_path), exist_ok=True)
            with open(self._mission_summary_path, "w", encoding="utf-8") as f:
                json.dump(self._mission, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"[MISSION SAVE ERROR] {e}")

    def _start_new_mission(self, save: bool = True):
        with self._mission_lock:
            self._mission = self._new_mission_template()
        if save:
            self._save_mission()

    def _mission_append_event(self, ev: Dict[str, Any]):
        with self._mission_lock:
            self._mission.setdefault("events", []).append(ev)
            self._rebuild_mission_tag_groups()
        self._save_mission()

    def _mission_update_event(self, event_id: str, patch: Dict[str, Any]):
        with self._mission_lock:
            for ev in self._mission.get("events", []):
                if ev.get("event") == event_id:
                    ev.update(patch)
                    break
            self._rebuild_mission_tag_groups()
        self._save_mission()

    def _count_tag_group_in_mission(self, tag_group_id: str) -> int:
        if not tag_group_id:
            return 0
        with self._mission_lock:
            events = self._mission.get("events", []) or []
            return sum(1 for ev in events if ev.get("tag_group_id") == tag_group_id)

    def _load_decay_table(self):
        path = self._vlm_decay_table
        try:
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            table: Dict[str, Dict[str, float]] = {}
            if isinstance(obj, dict):
                for raw_k, raw_v in obj.items():
                    k = normalize_tag_key(raw_k)
                    if not k:
                        continue
                    if isinstance(raw_v, dict):
                        ttl = float(raw_v.get("ttl", self._default_cost_ttl_s))
                    else:
                        ttl = float(raw_v)
                        lam = 1.0
                    table[k] = {"ttl": ttl}
            self._decay_table = table
            self.get_logger().info(f"[DECAY] loaded tags={len(self._decay_table)} from {path}")
        except Exception as e:
            self.get_logger().warn(f"[DECAY] load failed: {e}")
            self._decay_table = {}

    def _lookup_decay_ttl(self, tag_key: str) -> float:
        k = normalize_tag_key(tag_key)
        if k and k in self._decay_table:
            try:
                ttl = max(0.1, float(self._decay_table[k].get("ttl", self._default_cost_ttl_s)))
                self.get_logger().info(f"[DECAY] hit tag={k} ttl={ttl:.2f}s")
                return ttl
            except Exception:
                pass
        if k and ":" in k:
            base_k = k.split(":", 1)[0]
            if base_k in self._decay_table:
                try:
                    ttl = max(0.1, float(self._decay_table[base_k].get("ttl", self._default_cost_ttl_s)))
                    self.get_logger().info(f"[DECAY] hit tag={k} via base={base_k} ttl={ttl:.2f}s")
                    return ttl
                except Exception:
                    pass
        ttl = max(0.1, float(self._default_cost_ttl_s))
        self.get_logger().warn(f"[DECAY] miss tag={k!r} -> fallback ttl={ttl:.2f}s")
        return ttl

    def _reset_mission_cb(self, request, response):
        self._start_new_mission(save=True)
        self.get_logger().info("[MISSION] reset_mission service called → new mission started")
        response.success = True
        response.message = f"New mission started: {self._mission.get('mission_id', '')}"
        return response

    def _goal_status_cb(self, msg: GoalStatusArray):
        now = time.time()
        newly_succeeded = []
        for st in msg.status_list:
            if int(st.status) == int(GoalStatus.STATUS_SUCCEEDED):
                gid = ''.join(f'{b:02x}' for b in st.goal_info.goal_id.uuid)
                if gid not in self._seen_goal_success_ids:
                    self._seen_goal_success_ids.add(gid)
                    newly_succeeded.append(gid)
        if not newly_succeeded:
            return
        if now - self._last_goal_success_time < self._goal_success_cooldown_s:
            return
        self._last_goal_success_time = now
        self.get_logger().info(f"[GOAL] succeeded ids={len(newly_succeeded)} -> postrun LLM trigger")
        self._save_mission()  
        self._run_postrun_llm_async()

    def _depth_cb(self, msg: Image):
        if not self._speed_classifier_enable:
            return
        rx, ry = self._current_pose_xy
        if not self._is_in_corridor(rx, ry):
            return
        try:
            import struct
            h, w = msg.height, msg.width
            frac = self._depth_roi_frac
            r0 = int(h * (0.5 - frac / 2))
            r1 = int(h * (0.5 + frac / 2))
            c0 = int(w * (0.5 - frac / 2))
            c1 = int(w * (0.5 + frac / 2))
            depths = []
            step = msg.step 
            for row in range(r0, r1):
                for col in range(c0, c1):
                    offset = row * step + col * 4
                    val = struct.unpack_from('f', bytes(msg.data[offset:offset+4]))[0]
                    if math.isfinite(val) and 0.1 < val < 10.0:
                        depths.append(val)
            if not depths:
                return
            depths.sort()
            median_depth = depths[len(depths) // 2]
            now = time.time()
            with self._depth_lock:
                self._depth_history.append((now, median_depth))
                cutoff = now - 6.0
                self._depth_history = [(t, d) for t, d in self._depth_history if t >= cutoff]
        except Exception as e:
            self.get_logger().debug(f"[DEPTH_CB] error: {e}")

    def _depth_get_speed_at_detour_bbox(
        self, detour_time: float,
        x1: float, y1: float, x2: float, y2: float
    ) -> float:
        t_start = detour_time - self._speed_query_window_before_s
        t_end   = detour_time + self._speed_query_window_after_s
        with self._depth_lock:
            samples = [(t, d) for t, d in self._depth_bbox_history if t_start <= t <= t_end]
        if len(samples) < 4:
            with self._depth_lock:
                samples = [(t, d) for t, d in self._depth_history if t_start <= t <= t_end]
        if len(samples) < 4:
            return -1.0
        n = len(samples)
        ts = [s[0] - samples[0][0] for s in samples]
        ds = [s[1] for s in samples]
        t_mean = sum(ts) / n
        d_mean = sum(ds) / n
        num = sum((ts[i] - t_mean) * (ds[i] - d_mean) for i in range(n))
        den = sum((ts[i] - t_mean) ** 2 for i in range(n))
        if abs(den) < 1e-6:
            return -1.0
        depth_rate = num / den
        with self._sc_odom_lock:
            rvx, rvy = self._sc_robot_vx, self._sc_robot_vy
        robot_speed = math.hypot(rvx, rvy)
        approach_speed = abs(depth_rate)
        obs_speed = max(0.0, approach_speed - robot_speed)
        self.get_logger().info(
            f"[BBOX_DEPTH] depth_rate={depth_rate:.3f}m/s "
            f"approach={approach_speed:.3f}m/s robot={robot_speed:.3f}m/s "
            f"obs_speed={obs_speed:.3f}m/s samples={n}"
        )
        return obs_speed

    def _depth_get_speed_at_detour(self, detour_time: float) -> float:
        t_start = detour_time - self._speed_query_window_before_s
        t_end   = detour_time + self._speed_query_window_after_s
        with self._depth_lock:
            samples = [(t, d) for t, d in self._depth_history if t_start <= t <= t_end]
        if len(samples) < 4:
            return -1.0
        n = len(samples)
        ts = [s[0] - samples[0][0] for s in samples]
        ds = [s[1] for s in samples]
        t_mean = sum(ts) / n
        d_mean = sum(ds) / n
        num = sum((ts[i] - t_mean) * (ds[i] - d_mean) for i in range(n))
        den = sum((ts[i] - t_mean) ** 2 for i in range(n))
        if abs(den) < 1e-6:
            return -1.0
        depth_rate = num / den
        with self._sc_odom_lock:
            rvx, rvy = self._sc_robot_vx, self._sc_robot_vy
        robot_speed = math.hypot(rvx, rvy)
        approach_speed = abs(depth_rate)
        obs_speed = max(0.0, approach_speed - robot_speed)
        self.get_logger().info(
            f"[DEPTH_SPEED] depth_rate={depth_rate:.3f}m/s "
            f"approach={approach_speed:.3f}m/s robot={robot_speed:.3f}m/s "
            f"obs_speed={obs_speed:.3f}m/s samples={n}"
        )
        return obs_speed

    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        pose = msg.pose.pose.position
        self._current_pose_xy = (float(pose.x), float(pose.y))
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _plan_cb(self, msg: Path):
        current_length = self._compute_path_length(msg)
        old_length = self._previous_path_length
        self._previous_path_length = current_length

        if not self._gate_on_detour_only:
            return
        if old_length < self._detour_min_previous_length:
            return
        if current_length < self._detour_ratio_threshold * old_length:
            return

        now = time.time()
        if now - self._last_detour_trigger < self._detour_cooldown_s:
            return

        self._last_detour_trigger = now
        self._detour_active_until = now + self._detour_hold_s
        x, y = self._current_pose_xy
        event_id = self._make_event_id()
        self._current_event_id = event_id

        with self._latest_unexpected_lock:
            instant_centroids = self._latest_unexpected_centroids[:]
        if instant_centroids:
            rx, ry = self._current_pose_xy
            closest = min(instant_centroids, key=lambda c: math.hypot(c[0]-rx, c[1]-ry))
            if self._is_in_corridor(closest[0], closest[1]):
                nearest_cy = min(self._corridor_y_centers,
                                 key=lambda cy: abs(closest[1] - cy))
                snapped = (closest[0], nearest_cy)
                self._upsert_active_costs([snapped], now)
                self.get_logger().info(
                    f"[INSTANT_COST] detour → cost at ({snapped[0]:.2f},{snapped[1]:.2f}) "
                    f"(snapped from {closest[1]:.2f} to corridor center {nearest_cy:.2f})"
                )
            else:
                self.get_logger().info(
                    f"[INSTANT_COST] centroid=({closest[0]:.2f},{closest[1]:.2f}) not in corridor → skip"
                )
        else:
            self.get_logger().info("[INSTANT_COST] no unexpected centroids available at detour time")

        if self._speed_classifier_enable and self._speed_measure_stop_s > 0:
            def _stop_and_measure():
                try:
                    self.get_logger().info(
                        f"[SPEED_STOP] pausing {self._speed_measure_stop_s:.1f}s for accurate speed measurement"
                    )
                    self._speed_stop_active = True 
                    stop_msg = Twist()
                    hz = 50
                    interval = 1.0 / hz
                    steps = int(self._speed_measure_stop_s * hz)
                    for _ in range(steps):
                        self.cmd_vel_pub.publish(stop_msg)
                        time.sleep(interval)
                    self._speed_stop_active = False
                    self.get_logger().info("[SPEED_STOP] resumed navigation")
                except Exception as e:
                    self._speed_stop_active = False
                    self.get_logger().warn(f"[SPEED_STOP] error: {e}")
            threading.Thread(target=_stop_and_measure, daemon=True).start()

        if self._speed_measure_stop_s > 0:
            speed_class, obs_dist, avg_speed = "", -1.0, -1.0
        else:
            speed_class, obs_dist, avg_speed = self._sc_get_at_detour(now)
        self._current_speed_class = speed_class
        self._current_obstacle_dist_m = obs_dist
        trigger_text = (
            f"detour_detected old={old_length:.2f} new={current_length:.2f} ratio={current_length / max(old_length, 1e-6):.3f} "
            f"pose=({x:.2f},{y:.2f}) event={event_id} active_for={self._detour_hold_s:.1f}s"
            + (f" speed={avg_speed:.3f}m/s" if avg_speed >= 0 else "")
            + (f" speed_class={speed_class}" if speed_class else "")
            + (f" obstacle_dist={obs_dist:.2f}m" if obs_dist >= 0 else "")
        )
        self.get_logger().info(trigger_text)
        self._event_tag_map[event_id] = ""
        self._mission_append_event({
            "event": event_id,
            "timestamp_unix": now,
            "timestamp_local": datetime.now().isoformat(),
            "trigger_text": trigger_text,
            "pose_xy": {"x": float(x), "y": float(y)},
            "plan_prev_len": float(old_length),
            "plan_new_len": float(current_length),
            "ratio": float(current_length / max(old_length, 1e-6)),
            "vlm": None,
            "vlm_tag_key": None,
            "tag_group_id": None,
            "tag_repeat_count_in_mission": 0,
            "speed_class": speed_class,
            "obstacle_speed_mps": avg_speed if avg_speed >= 0 else None,
            "obstacle_dist_m": obs_dist if obs_dist >= 0 else None,
        })
        if self._enable_capture or self._vlm_enable:
            self._capture_detour_event(event_id, trigger_text, old_length, current_length, x, y)

    def _sc_odom_cb(self, msg):
        with self._sc_odom_lock:
            self._sc_robot_vx = msg.twist.twist.linear.x
            self._sc_robot_vy = msg.twist.twist.linear.y

    def _sc_worker_loop(self):
        import queue
        while True:
            try:
                msg = self._sc_scan_queue.get(timeout=1.0)
                self._sc_process_scan(msg)
            except queue.Empty:
                continue
            except Exception:
                pass

    def _sc_process_scan(self, scan_msg):
        try:
            pts = []
            angle = scan_msg.angle_min
            min_range = 0.15
            for r in scan_msg.ranges:
                if math.isfinite(r) and min_range <= r <= self._sc_max_range:
                    pts.append((r * math.cos(angle), r * math.sin(angle), r))
                angle += scan_msg.angle_increment
            if not pts:
                return

            closest_dist = min(p[2] for p in pts)

            clusters, current = [], [pts[0]]
            for p in pts[1:]:
                if math.hypot(p[0]-current[-1][0], p[1]-current[-1][1]) <= self._sc_cluster_dist:
                    current.append(p)
                else:
                    if len(current) >= self._sc_cluster_min:
                        clusters.append(current)
                    current = [p]
            if len(current) >= self._sc_cluster_min:
                clusters.append(current)
            if not clusters:
                return

            centroids = [
                (sum(p[0] for p in c)/len(c), sum(p[1] for p in c)/len(c))
                for c in clusters
            ]

            with self._sc_odom_lock:
                rvx, rvy = self._sc_robot_vx, self._sc_robot_vy
            robot_yaw = self._current_yaw

            with self._sc_track_lock:
                matched = set()
                for cx, cy in centroids:
                    best_tid, best_d = None, float("inf")
                    for tid, tr in self._sc_tracks.items():
                        if tid in matched:
                            continue
                        d = math.hypot(cx - tr.pos[0], cy - tr.pos[1])
                        if d < best_d:
                            best_d, best_tid = d, tid
                    if best_tid is not None and best_d < self._sc_cluster_dist * 3:
                        self._sc_tracks[best_tid].update(np.array([cx, cy]))
                        matched.add(best_tid)
                    else:
                        t = _KalmanTrack(cx, cy, dt=self._sc_kalman_dt)
                        self._sc_tracks[t.track_id] = t
                if not self._sc_tracks:
                    return

                cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)
                rx_map, ry_map = self._current_pose_xy
                _match_threshold = 2.0  

                dynamic_tracks = []
                for tr in self._sc_tracks.values():
                    tr_map_x = rx_map + cos_y * tr.pos[0] - sin_y * tr.pos[1]
                    tr_map_y = ry_map + sin_y * tr.pos[0] + cos_y * tr.pos[1]
                    is_static = self._is_static_obstacle(tr_map_x, tr_map_y)
                    self.get_logger().debug(
                        f"[TRACK] map=({tr_map_x:.2f},{tr_map_y:.2f}) "
                        f"static={is_static} speed={tr.speed:.3f}m/s"
                    )
                    if not is_static:
                        dynamic_tracks.append((tr, tr_map_x, tr_map_y))

                if not dynamic_tracks:
                    self.get_logger().debug(
                        f"[SPEED_SKIP] all {len(self._sc_tracks)} tracks are static"
                    )
                    return  

                best = None
                active = list(self._active_costs.values())
                if active:
                    best_dist = float("inf")
                    for tr, tr_map_x, tr_map_y in dynamic_tracks:
                        for cost in active:
                            d = math.hypot(tr_map_x - cost.x, tr_map_y - cost.y)
                            if d < best_dist:
                                best_dist = d
                                best = tr
                    if best_dist > _match_threshold:
                        return
                else:
                    best = min(dynamic_tracks, key=lambda t: math.hypot(*t[0].pos))[0]

                cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)
                obs_vx_map = cos_y * best.vel[0] - sin_y * best.vel[1]
                obs_vy_map = sin_y * best.vel[0] + cos_y * best.vel[1]
                robot_vx_map = cos_y * rvx - sin_y * rvy
                robot_vy_map = sin_y * rvx + cos_y * rvy
                abs_vx = obs_vx_map + robot_vx_map
                abs_vy = obs_vy_map + robot_vy_map
                speed_mps = math.hypot(abs_vx, abs_vy)
                centroid_dist = math.hypot(*best.pos)

            now = time.time()
            cos_y2, sin_y2 = math.cos(robot_yaw), math.sin(robot_yaw)
            rx_m, ry_m = self._current_pose_xy
            obs_map_x = rx_m + cos_y2 * best.pos[0] - sin_y2 * best.pos[1]
            obs_map_y = ry_m + sin_y2 * best.pos[0] + cos_y2 * best.pos[1]
            if not self._is_in_corridor(obs_map_x, obs_map_y):
                return
            with self._sc_history_lock:
                self._sc_speed_history.append((now, speed_mps))
                self._sc_dist_history.append((now, closest_dist))
                cutoff = now - 6.0
                self._sc_speed_history = [(t,s) for t,s in self._sc_speed_history if t >= cutoff]
                self._sc_dist_history  = [(t,d) for t,d in self._sc_dist_history  if t >= cutoff]
            self.get_logger().debug(
                f"[SPEED_HIST] speed={speed_mps:.3f}m/s n={len(self._sc_speed_history)}"
            )
        except Exception as e:
            self.get_logger().warn(f"[SPEED] scan error: {e}")

    def _sc_kalman_predict_cb(self):
        if not self._speed_classifier_enable or not _SPEED_OK:
            return
        with self._sc_track_lock:
            dead = []
            for tid, tr in self._sc_tracks.items():
                tr.predict()
                tr.miss_count += 1
                if tr.miss_count > self._sc_track_max_miss:
                    dead.append(tid)
            for tid in dead:
                del self._sc_tracks[tid]

    def _sc_get_at_detour_window(self, t_start: float, t_end: float) -> Tuple[str, float, float]:
        if not self._speed_classifier_enable or not _SPEED_OK:
            return "", -1.0, -1.0
        with self._sc_history_lock:
            spd_samples  = [s for t,s in self._sc_speed_history if t_start <= t <= t_end]
            dist_samples = [d for t,d in self._sc_dist_history  if t_start <= t <= t_end]
        avg_speed = -1.0
        avg_dist  = -1.0
        if spd_samples:
            sorted_spd = sorted(spd_samples)
            n = len(sorted_spd)
            avg_speed = sorted_spd[n // 2] if n % 2 == 1 else \
                        (sorted_spd[n//2 - 1] + sorted_spd[n//2]) / 2.0
            self.get_logger().info(
                f"[SPEED_WINDOW] t=[{t_start:.1f},{t_end:.1f}] "
                f"speed={avg_speed:.3f}m/s n={n}"
            )
        if dist_samples:
            avg_dist = sum(dist_samples) / len(dist_samples)
        speed_class = ""
        if avg_speed >= 0.0:
            last_tag = getattr(self, "_last_confirmed_base_tag", "")
            if last_tag and self._sc_gmm.is_ready(last_tag):
                speed_class = self._sc_gmm.predict(last_tag, avg_speed)
            else:
                thr = float(self.get_parameter("speed_threshold_mps").value)
                speed_class = "fast" if avg_speed >= thr else "slow"
            self._pending_speed_for_gmm = avg_speed
        return speed_class, avg_dist, avg_speed

    def _sc_get_at_detour(self, detour_time: float) -> Tuple[str, float]:
        if not self._speed_classifier_enable or not _SPEED_OK:
            return "", -1.0, -1.0

        depth_speed = self._depth_get_speed_at_detour(detour_time)

        t_start = detour_time - self._speed_query_window_before_s
        t_end   = detour_time + self._speed_query_window_after_s
        with self._sc_history_lock:
            spd_samples  = [s for t,s in self._sc_speed_history if t_start <= t <= t_end]
            dist_samples = [d for t,d in self._sc_dist_history  if t_start <= t <= t_end]
        if not spd_samples:
            with self._sc_history_lock:
                spd_samples  = [s for _,s in self._sc_speed_history[-5:]]
                dist_samples = [d for _,d in self._sc_dist_history[-5:]]

        avg_speed = -1.0
        if depth_speed >= 0.0:
            avg_speed = depth_speed
            self.get_logger().info(f"[SPEED_SRC] using depth camera speed={avg_speed:.3f}m/s")
        elif spd_samples:
            sorted_spd = sorted(spd_samples)
            n = len(sorted_spd)
            avg_speed = sorted_spd[n // 2] if n % 2 == 1 else \
                        (sorted_spd[n//2 - 1] + sorted_spd[n//2]) / 2.0
            self.get_logger().info(f"[SPEED_SRC] depth unavailable, using LiDAR speed={avg_speed:.3f}m/s")

        speed_class = ""
        avg_dist = -1.0
        if avg_speed >= 0.0:
            last_tag = getattr(self, "_last_confirmed_base_tag", "")
            if last_tag and self._sc_gmm.is_ready(last_tag):
                speed_class = self._sc_gmm.predict(last_tag, avg_speed)
            else:
                thr = float(self.get_parameter("speed_threshold_mps").value)
                speed_class = "fast" if avg_speed >= thr else "slow"
            self.get_logger().info(
                f"[SPEED] detour speed={avg_speed:.3f}m/s class={speed_class}(prelim)"
            )
            self._pending_speed_for_gmm = avg_speed
        if dist_samples:
            avg_dist = sum(dist_samples) / len(dist_samples)
            self.get_logger().debug(f"[DIST] detour dist={avg_dist:.3f}m n={len(dist_samples)}")
        return speed_class, avg_dist, avg_speed

    def _is_in_corridor(self, x: float, y: float) -> bool:

        x_min = min(self._corridor_start_x, self._corridor_end_x) - self._corridor_x_margin
        x_max = max(self._corridor_start_x, self._corridor_end_x) + self._corridor_x_margin
        if not (x_min <= x <= x_max):
            return False
        for cy in self._corridor_y_centers:
            if abs(y - cy) <= self._corridor_y_half_width:
                return True
        return False

    def _detour_gate_open(self) -> bool:
        if not self._gate_on_detour_only:
            return True
        return time.time() <= self._detour_active_until

    def _compute_path_length(self, path_msg: Path) -> float:
        poses = path_msg.poses
        if len(poses) < 2:
            return 0.0
        total_dist = 0.0
        for i in range(len(poses) - 1):
            p1 = poses[i].pose.position
            p2 = poses[i + 1].pose.position
            total_dist += math.hypot(p2.x - p1.x, p2.y - p1.y)
        return total_dist

    def _publish_pose_array(self, centroids: List[Tuple[float, float]], stamp_msg=None):
        out = PoseArray()
        if stamp_msg is not None:
            out.header.stamp = stamp_msg
        out.header.frame_id = self._map_frame
        for cx, cy in centroids:
            p = Pose()
            p.position.x = float(cx)
            p.position.y = float(cy)
            p.position.z = 0.0
            p.orientation.w = 1.0
            out.poses.append(p)
        self.object_pub.publish(out)
        self._last_publish_empty = (len(centroids) == 0)

    def _active_cost_centroids(self) -> List[Tuple[float, float]]:
        return [(c.x, c.y) for c in self._active_costs.values()]

    def _upsert_active_costs(self, centroids: List[Tuple[float, float]], now: float):
        current_event_id = self._current_event_id if self._detour_gate_open() else ""
        current_tag = self._event_tag_map.get(current_event_id, "") if current_event_id else ""
        current_tag_group_id = self._make_tag_group_id(current_tag) if current_tag else ""
        matched_ids = set()
        for cx, cy in centroids:
            best_id = None
            best_dist = self._track_merge_dist
            for cid, cost in self._active_costs.items():
                if cid in matched_ids:
                    continue
                d = math.hypot(cx - cost.x, cy - cost.y)
                if d <= best_dist:
                    best_dist = d
                    best_id = cid

            ttl_s = self._ttl_for_centroid(cx, cy, current_event_id, current_tag)
            if best_id is None:
                cid = self._next_cost_id
                self._next_cost_id += 1
                self._active_costs[cid] = ActiveCost(
                    cost_id=cid,
                    x=cx,
                    y=cy,
                    ttl_s=ttl_s,
                    expires_at=now + ttl_s,
                    last_update=now,
                    event_id=current_event_id,
                    tag_key=current_tag,
                    tag_group_id=current_tag_group_id,
                    prev_x=cx,
                    prev_y=cy,
                    prev_t=now,
                    speed_samples=[],
                )
                self.get_logger().info(
                    f"[ADD] event={current_event_id} cost_id={cid} ttl={ttl_s:.2f}s expires_in={ttl_s:.2f}s"
                )
                matched_ids.add(cid)
            else:
                cost = self._active_costs[best_id]
                dt = now - cost.prev_t
                if dt > 0.05:  
                    dist = math.hypot(cx - cost.prev_x, cy - cost.prev_y)
                    spd = dist / dt
                    if cost.speed_samples is None:
                        cost.speed_samples = []
                    cost.speed_samples.append(spd)
                    if len(cost.speed_samples) > 20:
                        cost.speed_samples = cost.speed_samples[-20:]
                    cost.prev_x = cx
                    cost.prev_y = cy
                    cost.prev_t = now
                cost.x = cx if not self._speed_stop_active else cost.x
                cost.y = cy if not self._speed_stop_active else cost.y
                cost.ttl_s = ttl_s
                cost.last_update = now
                if current_event_id:
                    cost.event_id = current_event_id
                if current_tag:
                    cost.tag_key = current_tag
                if current_tag_group_id:
                    cost.tag_group_id = current_tag_group_id
                cost.expires_at = now + ttl_s
                self.get_logger().debug(
                    f"[REFRESH] event={cost.event_id} cost_id={best_id} ttl={ttl_s:.2f}s expires_in={ttl_s:.2f}s"
                )
                matched_ids.add(best_id)

    def _ttl_for_centroid(self, cx: float, cy: float, event_id: str = "", tag_key: str = "") -> float:
        tk = normalize_tag_key(tag_key or self._event_tag_map.get(event_id, ""))
        if tk:
            ttl = self._lookup_decay_ttl(tk)
            self.get_logger().info(f"[TTL] tagged event={event_id} tag={tk} ttl={ttl:.2f}s")
            return ttl
        ttl = max(0.1, float(self._default_cost_ttl_s))
        self.get_logger().debug(f"[TTL] pre-VLM default event={event_id} ttl={ttl:.2f}s")
        return ttl

    def _remove_expired_costs(self, now: float) -> int:
        expired_ids = [cid for cid, cost in self._active_costs.items() if cost.expires_at <= now]
        for cid in expired_ids:
            self._active_costs.pop(cid, None)
        return len(expired_ids)

    def _maintain_cb(self):
        if not self._enabled:
            return
        now = time.time()
        expired_count = self._remove_expired_costs(now)
        alive_centroids = self._active_cost_centroids()

        if expired_count > 0:
            self.get_logger().info(
                f"[EXPIRE] expired={expired_count} survivors={len(alive_centroids)} "
                f"clear_on_expire={self._enable_global_clear_on_expire}"
            )
            if len(alive_centroids) == 0:
                self._detour_active_until = 0.0
            if self._enable_global_clear_on_expire:
                self.get_logger().info(
                    f"[CLEAR] trigger service={self._clear_service_name} "
                    f"survivors_after_expire={len(alive_centroids)}"
                )
                self._call_clear_service_and_republish()
            else:
                self.get_logger().info(
                    f"[PUBLISH] survivors-only poses={len(alive_centroids)} topic={self._output_topic}"
                )
                self._publish_pose_array(alive_centroids, self._last_scan_stamp)
            return

        if alive_centroids:
            self._publish_pose_array(alive_centroids, self._last_scan_stamp)
        elif self._publish_empty_when_no_active and not self._last_publish_empty:
            self.get_logger().info(f"[PUBLISH] empty PoseArray -> {self._output_topic}")
            self._publish_pose_array([], self._last_scan_stamp)

    def _call_clear_service_and_republish(self):
        if self._clear_inflight:
            self.get_logger().info("[CLEAR] skipped because clear is already inflight")
            self._clear_pending_republish = True
            return

        alive_centroids = self._active_cost_centroids()
        ready = self.clear_client.wait_for_service(timeout_sec=self._clear_service_wait_s)
        self.get_logger().info(
            f"[CLEAR] request service={self._clear_service_name} ready={ready} survivors={len(alive_centroids)}"
        )
        if not ready:
            self.get_logger().warn(f"[CLEAR] unavailable: {self._clear_service_name}")
            self._publish_pose_array(alive_centroids, self._last_scan_stamp)
            return

        self._clear_inflight = True
        request = ClearEntireCostmap.Request()
        future = self.clear_client.call_async(request)
        future.add_done_callback(self._on_clear_done)

    def _on_clear_done(self, future):
        self._clear_inflight = False
        try:
            _ = future.result()
            self.get_logger().info("[CLEAR] completed successfully")
        except Exception as e:
            self.get_logger().error(f"[CLEAR] failed: {e}")

        alive_after = self._active_cost_centroids()
        if alive_after:
            self.get_logger().info(
                f"[PUBLISH] republish survivors poses={len(alive_after)} topic={self._output_topic}"
            )
            self._publish_pose_array(alive_after, self._last_scan_stamp)
        elif self._publish_empty_when_no_active:
            self.get_logger().info(f"[PUBLISH] empty after clear -> {self._output_topic}")
            self._publish_pose_array([], self._last_scan_stamp)

        if self._clear_pending_republish:
            self._clear_pending_republish = False
            self._call_clear_service_and_republish()

    def _make_event_id(self) -> str:
        return f"event_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _capture_detour_event(self, event_id: str, trigger_text: str, old_length: float, new_length: float, x: float, y: float):
        if not (_CAPTURE_OK and (self._enable_capture or self._vlm_enable)):
            return
        try:
            with self._frame_lock:
                if len(self._frame_buf) < self._frames_to_save:
                    self.get_logger().warn(
                        f"[CAPTURE] Not enough frames in buffer: {len(self._frame_buf)}/{self._frames_to_save} (camera_topic={self._camera_topic})"
                    )
                    return
                frames = list(self._frame_buf)[-self._frames_to_save:]

            event_dir = os.path.join(self._save_root, event_id)
            os.makedirs(event_dir, exist_ok=True)

            frame_paths: List[str] = []
            t0 = time.time()
            for i, (_ts, img) in enumerate(frames):
                out_path = os.path.join(event_dir, f"frame_{i:02d}.jpg")
                ok = cv2.imwrite(out_path, img)
                if ok:
                    frame_paths.append(out_path)

            meta = {
                "event": event_id,
                "timestamp_unix": time.time(),
                "timestamp_local": datetime.now().isoformat(),
                "trigger_text": trigger_text,
                "pose_xy": {"x": float(x), "y": float(y)},
                "plan_prev_len": float(old_length),
                "plan_new_len": float(new_length),
                "ratio": float(new_length / max(old_length, 1e-6)),
                "camera_topic": self._camera_topic,
                "frames_saved": len(frame_paths),
                "sample_hz": self._sample_hz,
            }
            with open(os.path.join(event_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self._mission_update_event(event_id, {"event_dir": event_dir, "frames_saved": len(frame_paths)})

            dt = time.time() - t0
            self.get_logger().info(f"[CAPTURED] frames={len(frame_paths)} save_dt={dt:.2f}s -> {event_dir}")

            if self._vlm_enable and frame_paths:
                self._run_vlm_async(event_dir, frame_paths, trigger_text)
        except Exception as e:
            self.get_logger().error(f"[CAPTURE ERROR] {e}")
            self.get_logger().error(traceback.format_exc())

    def _run_vlm_async(self, event_dir: str, frame_paths: List[str], trigger_text: str):
        if self._vlm_singleflight:
            with self._vlm_guard:
                if self._vlm_running:
                    self.get_logger().warn("[VLM] skip: already running (singleflight)")
                    return
                self._vlm_running = True

        def _runner():
            try:
                self._run_vlm_and_save(event_dir, frame_paths, trigger_text)
            finally:
                if self._vlm_singleflight:
                    with self._vlm_guard:
                        self._vlm_running = False

        th = threading.Thread(target=_runner, daemon=True)
        th.start()

    def _run_vlm_and_save(self, event_dir: str, frame_paths: List[str], trigger_text: str):
        t0 = time.time()
        try:
            sc = getattr(self, "_current_speed_class", "")
            od = getattr(self, "_current_obstacle_dist_m", -1.0)
            context_hint = ""
            if sc in ("fast", "slow"):
                motion = "moving quickly" if sc == "fast" else "moving slowly or stationary"
                context_hint += f"[Obstacle Speed] classified as '{sc}' ({motion}). Append :{sc} to tag_key.\n"
            if od >= 0:
                context_hint += f"[Obstacle Distance] approximately {od:.2f}m from robot at detour time.\n"
            prompt = (
                f"{self._vlm_prompt}\n\n"
                f"[Detour Trigger]\n{trigger_text}\n\n"
                + (f"[Context]\n{context_hint}\n" if context_hint else "")
                + "You are given consecutive robot-camera frames in chronological order.\n"
                "Focus on the main obstacle that most likely caused the detour.\n"
                "Choose exactly one tag_key from the allowed tag set.\n"
                "Evidence is required.\n"
                "Evidence must be a very short structured phrase, not a sentence.\n"
                "Use only lowercase letters, numbers, and underscores in evidence.\n"
                "Do not use spaces, quotation marks, commas, colons, braces, or line breaks in evidence.\n"
            )

            cmd = [
                self._vlm_python, self._vlm_script,
                "--images", *frame_paths,
                "--prompt", prompt,
                "--model", (self._vlm_model[7:] if str(self._vlm_model).startswith('models/') else self._vlm_model),
                "--decay_table_path", self._vlm_decay_table,
            ]
            env = os.environ.copy()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self._vlm_timeout_sec, env=env)
            dt = time.time() - t0
            stdout = (r.stdout or "").strip()
            stderr = (r.stderr or "").strip()

            self.get_logger().info(
                f"[VLM] done event={os.path.basename(event_dir)} "
                f"dt={dt:.2f}s rc={int(r.returncode)} frames={len(frame_paths)}"
            )

            if stdout:
                self.get_logger().info(f"[VLM] stdout(head): {stdout[:self._debug_vlm_stdout_chars]}")
            if stderr:
                self.get_logger().warn(f"[VLM] stderr(head): {stderr[:self._debug_vlm_stderr_chars]}")

            result_json = None
            try:
                result_json = json.loads(stdout) if stdout else None
            except Exception:
                result_json = None

            with open(os.path.join(event_dir, "vlm_result.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "event": os.path.basename(event_dir),
                    "model": self._vlm_model,
                    "prompt": self._vlm_prompt,
                    "trigger_text": trigger_text,
                    "returncode": int(r.returncode),
                    "stdout": stdout,
                    "stderr": stderr,
                    "result_json": result_json,
                    "dt_sec": float(dt),
                }, f, ensure_ascii=False, indent=2)

            event_id = os.path.basename(event_dir)
            base_tag = extract_vlm_tag_key(result_json)

            centroid_speed = -1.0
            if self._speed_classifier_enable:
                event_costs = [c for c in self._active_costs.values() if c.event_id == event_id]
                if event_costs:
                    ex, ey = event_costs[0].x, event_costs[0].y
                    with self._centroid_track_lock:
                        best_tr, best_d = None, float("inf")
                        for tr in self._centroid_tracks.values():
                            d = math.hypot(tr["x"] - ex, tr["y"] - ey)
                            if d < best_d:
                                best_d, best_tr = d, tr
                        if best_tr is not None and best_d < 2.0 and len(best_tr["samples"]) >= 3:
                            samples = sorted(best_tr["samples"])
                            n = len(samples)
                            centroid_speed = samples[n // 2]
                            self.get_logger().info(
                                f"[CENTROID_SPEED] event={event_id} "
                                f"speed={centroid_speed:.3f}m/s "
                                f"samples={n} dist={best_d:.2f}m"
                            )
                if centroid_speed >= 0.0:
                    self._pending_speed_for_gmm = centroid_speed

            pending_speed = getattr(self, "_pending_speed_for_gmm", -1.0)
            if base_tag and pending_speed >= 0 and self._speed_classifier_enable:
                if not self._gmm_freeze:
                    self._sc_gmm.add_sample(base_tag, pending_speed)
                else:
                    self.get_logger().info(f"[GMM] frozen, skip add_sample for {base_tag} speed={pending_speed:.3f}m/s")
                self._last_confirmed_base_tag = base_tag
                self.get_logger().info(
                    f"[GMM] {base_tag} +sample speed={pending_speed:.3f}m/s "
                    f"n={self._sc_gmm.n_samples(base_tag)} "
                    f"ready={self._sc_gmm.is_ready(base_tag)}"
                )
                if self._sc_gmm.is_ready(base_tag):
                    self.get_logger().info(
                        f"[GMM] {base_tag}: {self._sc_gmm._gmms[base_tag].summary()}"
                    )
            self._pending_speed_for_gmm = -1.0

            sc = ""
            if base_tag and pending_speed >= 0 and self._speed_classifier_enable:
                if self._sc_gmm.is_ready(base_tag):
                    sc = self._sc_gmm.predict(base_tag, pending_speed)
                else:
                    thr = float(self.get_parameter("speed_threshold_mps").value)
                    sc = "fast" if pending_speed >= thr else "slow"
            if not sc:
                sc = getattr(self, "_current_speed_class", "")

            if base_tag and sc in ("fast", "slow"):
                tag_key = f"{base_tag}:{sc}"
            else:
                tag_key = base_tag
            tag_group_id = self._make_tag_group_id(tag_key) if tag_key else ""
            if tag_key:
                self._event_tag_map[event_id] = tag_key
            if base_tag and tag_key != base_tag:
                self.get_logger().info(
                    f"[TAG] compound tag: {base_tag} + speed:{sc} -> {tag_key}"
                )
            self._mission_update_event(event_id, {
                "vlm": result_json,
                "vlm_tag_key": tag_key or None,
                "vlm_base_tag": base_tag or None,
                "vlm_speed_class": sc if sc in ("fast", "slow") else None,
                "tag_group_id": tag_group_id or None,
                "vlm_returncode": int(r.returncode),
                "vlm_dt_sec": float(dt),
            })
            repeat_count = self._count_tag_group_in_mission(tag_group_id) if tag_group_id else 0
            if repeat_count > 0:
                self._mission_update_event(event_id, {
                    "tag_repeat_count_in_mission": int(repeat_count),
                })
            if self._apply_vlm_decay_online and tag_key:
                ttl_base = max(self._min_online_tag_ttl_s, self._lookup_decay_ttl(tag_key))
                now2 = time.time()
                changed = 0
                for cost in self._active_costs.values():
                    if cost.event_id == event_id:
                        cx = float(cost.x)
                        cy_obs = float(cost.y)
                        if not self._is_in_corridor(cx, cy_obs):
                            self.get_logger().info(
                                f"[DETOUR_SKIP] obstacle=({cx:.2f},{cy_obs:.2f}) not in corridor → skip costmap"
                            )
                            continue

                        c_start = self._corridor_start_x
                        c_end   = self._corridor_end_x

                        robot_xy = getattr(self, '_current_pose_xy', None) or getattr(self, 'latest_amcl_pose', None)
                        if robot_xy is not None:
                            robot_x = float(robot_xy[0])
                            if robot_x > cx:
                                c_start, c_end = c_end, c_start

                        corridor_len = c_end - c_start
                        if abs(corridor_len) > 1e-3:
                            depth_ratio = max(0.0, min(1.0,
                                (cx - c_start) / corridor_len
                            ))
                        else:
                            depth_ratio = 0.0
                        depth_corrected = ttl_base * (1.0 - depth_ratio)
                        learned_ttl = max(
                            self._min_online_tag_ttl_s,
                            depth_corrected - float(dt)
                        )
                        self.get_logger().info(
                            f"[DEPTH] obstacle_x={cx:.2f} robot_x={robot_x if robot_xy else 'N/A':.2f} "
                            f"corridor=[{c_start:.1f},{c_end:.1f}] depth_ratio={depth_ratio:.3f}"
                        )
                        self.get_logger().info(
                            f"[DEPTH->TTL] tag={tag_key} base={ttl_base:.2f}s "
                            f"x(1-{depth_ratio:.3f})-vlm_dt({float(dt):.1f}s)={learned_ttl:.2f}s"
                        )
                        cost.tag_key = tag_key
                        cost.tag_group_id = tag_group_id
                        cost.ttl_s = learned_ttl
                        cost.expires_at = now2 + learned_ttl
                        changed += 1
                self._mission_update_event(event_id, {
                    "applied_ttl_s":         float(learned_ttl) if changed > 0 else None,
                    "depth_corrected_ttl_s": float(depth_corrected) if changed > 0 else None,
                    "vlm_dt_used_s":         float(dt) if changed > 0 else None,
                    "depth_ratio":           float(depth_ratio) if changed > 0 else None,
                    "ttl_base_s":            float(ttl_base),
                })
                if changed > 0:
                    self.get_logger().info(
                        f"[VLM->TTL] event={event_id} tag={tag_key} group={tag_group_id} "
                        f"repeat={repeat_count} ttl_base={ttl_base:.2f}s "
                        f"depth_ratio={depth_ratio:.3f} applied_ttl={learned_ttl:.2f}s "
                        f"active_costs={changed}"
                    )

            msg = String()
            payload = {
                "event": os.path.basename(event_dir),
                "vlm": result_json,
                "returncode": int(r.returncode),
                "dt_sec": float(dt),
            }
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.vlm_pub.publish(msg)
        except Exception as e:
            dt = time.time() - t0
            self.get_logger().error(f"[VLM ERROR] dt={dt:.1f}s event={os.path.basename(event_dir)} err={e}")
            self.get_logger().error(traceback.format_exc())
            try:
                with open(os.path.join(event_dir, "vlm_result.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "event": os.path.basename(event_dir),
                        "model": self._vlm_model,
                        "prompt": self._vlm_prompt,
                        "trigger_text": trigger_text,
                        "error": str(e),
                        "dt_sec": float(dt),
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def _parse_llm_proposal_updates(self, stdout: str) -> List[Tuple[str, float, float]]:
        updates: List[Tuple[str, float, float]] = []
        if not stdout:
            return updates
        pat = re.compile(
            r"^\s*-\s+([A-Za-z0-9_\-:]+):\s+ttl_s=([0-9]+(?:\.[0-9]+)?)",
            re.MULTILINE
        )
        conf_map: Dict[str, float] = {}
        self._llm_reason_map: Dict[str, str] = {}
        try:
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith("[PROPOSAL_JSON]"):
                    json_str = line[len("[PROPOSAL_JSON]"):].strip()
                    obj = json.loads(json_str)
                    for u in obj.get("updates", []):
                        tk = normalize_tag_key(u.get("tag_key", ""))
                        if tk:
                            raw_conf = u.get("confidence")
                            if raw_conf is not None:
                                conf_map[tk] = float(raw_conf)
                                self.get_logger().info(
                                    f"[LLM_CONF] tag={tk} confidence={float(raw_conf):.2f} (from LLM output)"
                                )
                            else:
                                conf_map[tk] = 0.5
                                self.get_logger().info(
                                    f"[LLM_CONF] tag={tk} confidence=0.5 (default, not in LLM output)"
                                )
                            reason = u.get("reason", "")
                            if reason:
                                self._llm_reason_map[tk] = str(reason)
                elif line.startswith("{") or '"updates"' in line:
                    obj = json.loads(line)
                    for u in obj.get("updates", []):
                        tk = normalize_tag_key(u.get("tag_key", ""))
                        if tk and tk not in conf_map:
                            conf_map[tk] = float(u.get("confidence", 0.5))
        except Exception:
            pass
        for m in pat.finditer(stdout):
            tag = normalize_tag_key(m.group(1))
            try:
                ttl = float(m.group(2))
            except Exception:
                continue
            if tag:
                conf = conf_map.get(tag, 0.5)
                updates.append((tag, ttl, conf))
        return updates

    def _approval_significant_updates(self, updates: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float, float, float]]:
        sig: List[Tuple[str, float, float, float, float]] = []
        for tag, new_ttl, conf in updates:
            old_ttl = max(0.1, float(self._decay_table.get(tag, {}).get("ttl", self._default_cost_ttl_s)))
            pct = abs(new_ttl - old_ttl) / max(abs(old_ttl), 1e-6) * 100.0
            self.get_logger().info(
                f"[LLM] compare tag={tag} old_ttl={old_ttl:.2f}s new_ttl={new_ttl:.2f}s "
                f"change={pct:.2f}% confidence={conf:.2f}"
            )
            sig.append((tag, old_ttl, new_ttl, pct, conf))
        return sig

    def _apply_llm_proposal_updates_local(self, proposal_updates: List[Tuple[str, float]]) -> int:
        path = os.path.expanduser(self._vlm_decay_table)

        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    decay_table = json.load(f)
            else:
                decay_table = {}
        except Exception as e:
            self.get_logger().error(f"[LLM] failed to read decay_table for local apply: {e}")
            return 0

        updated = 0
        updated_tags = []

        for tag, new_ttl in proposal_updates:
            try:
                k = normalize_tag_key(tag)
                ttl = float(new_ttl)
            except Exception:
                continue

            cur = decay_table.get(k)
            if isinstance(cur, dict):
                cur = dict(cur)
                cur["ttl"] = ttl
                decay_table[k] = cur
            else:
                decay_table[k] = {"ttl": ttl}

            updated += 1
            updated_tags.append(k)

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(decay_table, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"[LLM] failed to save decay_table after local apply: {e}")
            return 0

        self.get_logger().info(
            f"[LLM] locally applied proposal updates={updated} path={path} tags={updated_tags}"
        )
        return updated


    def _record_llm_policy_update_from_preview(
        self,
        proposal_updates: List[Tuple[str, float]],
        preview_stdout: str = "",
    ):
        try:
            with self._mission_lock:
                ms = dict(self._mission)

            ms.setdefault("llm_policy_update", {})
            ms["llm_policy_update"] = {
                "model": self._llm_decay_model,
                "updated_tags": [str(tag) for tag, *_ in proposal_updates],
                "notes": "Applied from human-preview proposal without second Gemini call.",
                "timestamp": time.time(),
                "source": "policybridge_local_apply_from_preview",
                "preview_stdout_head": (preview_stdout or "")[:1000],
            }

            out_path = os.path.expanduser(self._mission_summary_out_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ms, f, ensure_ascii=False, indent=2)

            in_place = os.path.expanduser(self._mission_summary_path)
            os.makedirs(os.path.dirname(in_place), exist_ok=True)
            with open(in_place, "w", encoding="utf-8") as f:
                json.dump(ms, f, ensure_ascii=False, indent=2)

            self.get_logger().info(f"[LLM] wrote mission summary from preview apply -> {out_path}")
        except Exception as e:
            self.get_logger().warn(f"[LLM] failed to record preview-applied policy update: {e}")

    def _extract_current_event_cases_for_archive(
        self,
        proposal_updates: Optional[List[Tuple[str, float]]] = None,
        approval_status: Optional[str] = None,
        human_feedback: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            with self._mission_lock:
                ms = json.loads(json.dumps(self._mission))

            mission_id = str(ms.get("mission_id") or "unknown_mission")
            events = ms.get("events", []) or []
            proposed_map = {}
            for tag, ttl, *_ in (proposal_updates or []):
                try:
                    proposed_map[normalize_tag_key(tag)] = float(ttl)
                except Exception:
                    pass

            out: List[Dict[str, Any]] = []

            for ev in events:
                if not isinstance(ev, dict):
                    continue

                tag_key = normalize_tag_key(ev.get("vlm_tag_key"))
                if not tag_key:
                    continue

                vlm = ev.get("vlm")
                confidence = None
                evidence = None
                if isinstance(vlm, dict):
                    try:
                        confidence = float(vlm.get("confidence")) if vlm.get("confidence") is not None else None
                    except Exception:
                        confidence = None
                    evidence = vlm.get("evidence") or vlm.get("reason")
                elif isinstance(vlm, str):
                    try:
                        obj = json.loads(vlm)
                        if isinstance(obj, dict):
                            confidence = float(obj.get("confidence")) if obj.get("confidence") is not None else None
                            evidence = obj.get("evidence") or obj.get("reason")
                        else:
                            evidence = vlm[:300]
                    except Exception:
                        evidence = vlm[:300]

                trigger_text = ev.get("trigger_text") or ev.get("trigger") or ev.get("detour_trigger") or ""
                old_len = new_len = detour_ratio = None
                try:
                    m_old = re.search(r"old\s*=\s*([0-9]+(?:\.[0-9]+)?)", trigger_text, re.IGNORECASE)
                    m_new = re.search(r"new\s*=\s*([0-9]+(?:\.[0-9]+)?)", trigger_text, re.IGNORECASE)
                    m_ratio = re.search(r"ratio\s*=\s*([0-9]+(?:\.[0-9]+)?)", trigger_text, re.IGNORECASE)
                    old_len = float(m_old.group(1)) if m_old else None
                    new_len = float(m_new.group(1)) if m_new else None
                    detour_ratio = float(m_ratio.group(1)) if m_ratio else None
                except Exception:
                    pass

                applied_ttl_s = None
                for key in ("applied_ttl_s", "ttl_s", "ttl"):
                    if key in ev:
                        try:
                            applied_ttl_s = float(ev.get(key)) if ev.get(key) is not None else None
                        except Exception:
                            applied_ttl_s = None
                        break

                repeat_count = ev.get("tag_repeat_count_in_mission", 1)
                try:
                    repeat_count = int(repeat_count)
                except Exception:
                    repeat_count = 1
                if repeat_count < 1:
                    repeat_count = 1

                ts = ev.get("timestamp") or ev.get("ts") or ev.get("time") or ev.get("timestamp_unix")
                try:
                    ts = float(ts) if ts is not None else None
                except Exception:
                    ts = None

                feedback_raw = (human_feedback or "").strip()
                feedback_norm = self._normalize_feedback_text(feedback_raw)

                out.append({
                    "mission_id": mission_id,
                    "event_id": str(ev.get("event") or ev.get("event_id") or ev.get("id") or ""),
                    "tag_key": tag_key,
                    "detour_ratio": detour_ratio,
                    "old_len": old_len,
                    "new_len": new_len,
                    "confidence": confidence,
                    "evidence": evidence,
                    "applied_ttl_s": applied_ttl_s,
                    "timestamp": ts,
                    "tag_group_id": str(ev.get("tag_group_id")) if ev.get("tag_group_id") else None,
                    "repeat_count_in_mission": repeat_count,
                    "same_obstacle_reencountered": bool(repeat_count >= 2),
                    "approval_mode": "human",
                    "approval_status": approval_status,
                    "proposed_ttl_s": proposed_map.get(tag_key),
                    "approval_timestamp": time.time(),
                    "human_feedback": feedback_raw,
                    "human_feedback_norm": feedback_norm,
                    "human_feedback_present": bool(feedback_raw),
                })

            return [c for c in out if c.get("event_id")]
        except Exception as e:
            self.get_logger().warn(f"[LLM] failed to extract archive cases from mission: {e}")
            return []


    def _append_archive_from_preview_local(
        self,
        proposal_updates: Optional[List[Tuple[str, float]]] = None,
        approval_status: Optional[str] = None,
        human_feedback: Optional[str] = None,
    ):
        if not self._llm_decay_rag_enable or not self._llm_decay_append_to_archive:
            return

        path = os.path.expanduser(self._llm_decay_retrieval_archive_path)
        cases = self._extract_current_event_cases_for_archive(
            proposal_updates=proposal_updates,
            approval_status=approval_status,
            human_feedback=human_feedback,
        )
        if not cases:
            self.get_logger().info("[LLM] no cases to append to archive from preview")
            return

        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    archive = json.load(f)
            else:
                archive = {}
        except Exception:
            archive = {}

        if not isinstance(archive, dict):
            archive = {}
        old_rows = archive.get("cases", []) or []
        if not isinstance(old_rows, list):
            old_rows = []

        seen = {
            (
                row.get("mission_id"),
                row.get("event_id"),
                row.get("tag_key"),
                row.get("approval_status"),
                row.get("human_feedback_norm") or "",
            )
            for row in old_rows
            if isinstance(row, dict)
        }
        added = 0
        for c in cases:
            key = (
                c.get("mission_id"),
                c.get("event_id"),
                c.get("tag_key"),
                c.get("approval_status"),
                c.get("human_feedback_norm") or "",
            )
            if key in seen:
                continue
            old_rows.append(c)
            seen.add(key)
            added += 1

        max_cases = max(1, int(self._llm_decay_archive_max_cases))
        if len(old_rows) > max_cases:
            old_rows = old_rows[-max_cases:]

        archive["cases"] = old_rows
        archive["updated_at"] = time.time()

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(archive, f, ensure_ascii=False, indent=2)
            self.get_logger().info(
                f"[LLM] locally appended archive cases={added} "
                f"status={approval_status} path={path} total={len(old_rows)}"
            )
        except Exception as e:
            self.get_logger().error(f"[LLM] failed to append archive locally from preview: {e}")

    def _exec_llm_decay(self, cmd: List[str], mode: str, run_input: Optional[str] = None) -> Tuple[subprocess.CompletedProcess, float]:
        t0 = time.time()
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300.0,
            env=os.environ.copy(),
            input=run_input,
        )
        dt = time.time() - t0
        stdout = (r.stdout or "")
        stderr = (r.stderr or "")
        if stdout:
            filtered = "\n".join(
                line for line in stdout.splitlines()
                if "dry_run" not in line.lower()
                and "NOT modified" not in line
                and not line.strip().startswith("[PROPOSAL_JSON]")
            )
            self.get_logger().info(f"[LLM] stdout(head): {filtered[:self._llm_stdout_log_chars]}")
        if stderr:
            self.get_logger().warn(f"[LLM] stderr(head): {stderr[:self._llm_stderr_log_chars]}")
        payload = {
            "returncode": int(r.returncode),
            "dt_sec": float(dt),
            "mode": mode,
            "stdout": stdout[:self._llm_stdout_log_chars],
            "stderr": stderr[:self._llm_stderr_log_chars],
            "mission_summary": self._mission_summary_path,
            "mission_summary_out": self._mission_summary_out_path,
            "decay_table": self._vlm_decay_table,
            "rag": {
                "enabled": bool(self._llm_decay_rag_enable),
                "archive_path": self._llm_decay_retrieval_archive_path if self._llm_decay_rag_enable else "",
                "max_repeat1_cases": int(self._llm_decay_retrieval_max_repeat1_cases) if self._llm_decay_rag_enable else 0,
                "append": bool(self._llm_decay_append_to_archive) if self._llm_decay_rag_enable else False,
            },
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.llm_decay_pub.publish(msg)
        return r, dt

    def _run_postrun_llm_async(self):
        if self._calibration_mode:
            self.get_logger().info(
                f"[CALIBRATION] LLM skipped. GMM status:\n{self._sc_gmm.summary()}"
            )
            if self._start_new_mission_after_llm:
                self._start_new_mission(save=True)
            return
        if not self._llm_decay_enable:
            self.get_logger().info("[LLM] postrun LLM disabled")
            return
        if self._llm_decay_singleflight:
            with self._llm_decay_guard:
                if self._llm_decay_running:
                    self.get_logger().warn("[LLM] skip postrun: already running")
                    return
                self._llm_decay_running = True

        def _runner():
            try:
                self._run_postrun_llm_once()
            finally:
                if self._llm_decay_singleflight:
                    with self._llm_decay_guard:
                        self._llm_decay_running = False

        threading.Thread(target=_runner, daemon=True).start()

    def _run_postrun_llm_once(self):
        try:
            with self._mission_lock:
                events_n = len(self._mission.get("events", []))
            if events_n == 0:
                self.get_logger().info("[LLM] skip postrun: no mission events")
                return

            script_path = self._llm_decay_script
            if self._llm_decay_rag_enable:
                if script_path.endswith('llm_decay_gemini_v3.py'):
                    rag_candidate = script_path[:-3] + '_rag.py'
                    if os.path.exists(rag_candidate):
                        script_path = rag_candidate

            base_cmd = [
                self._llm_decay_python, script_path,
                "--input", self._mission_summary_path,
                "--output", self._mission_summary_out_path,
                "--decay_table_path", self._vlm_decay_table,
                "--init_table_if_missing",
                "--update_mission_summary", self._mission_summary_path,
                "--model", (self._llm_decay_model[7:] if str(self._llm_decay_model).startswith('models/') else self._llm_decay_model),
            ]
            if self._llm_decay_rag_enable:
                base_cmd += [
                    "--retrieval_archive_path", self._llm_decay_retrieval_archive_path,
                    "--retrieval_max_repeat1_cases", str(max(1, self._llm_decay_retrieval_max_repeat1_cases)),
                    "--archive_max_cases", str(max(1, self._llm_decay_archive_max_cases)),
                ]
                if self._llm_decay_append_to_archive:
                    base_cmd.append("--append_to_archive")
            if self._llm_debug_prompt_path:
                base_cmd += ["--debug_prompt", self._llm_debug_prompt_path]

            mode = self._llm_decay_approval_mode
            self.get_logger().info(
                f"[LLM] postrun start events={events_n} mode={mode} script={script_path}" +
                (f" RAG archive={self._llm_decay_retrieval_archive_path} "
                f"max_repeat1_cases={self._llm_decay_retrieval_max_repeat1_cases} "
                f"append={self._llm_decay_append_to_archive}")
            )

            if mode == "auto":
                cmd = list(base_cmd) + ["--yes"]
                r, dt = self._exec_llm_decay(cmd, mode="auto", run_input=None)
                if r.returncode == 0:
                    self.get_logger().info(f"[LLM] postrun done dt={dt:.1f}s mode=auto; reloading decay table")
                    self._load_decay_table()
                    self._mission["llm_postrun_dt_s"] = float(dt)
                    self._mission["llm_postrun_mode"] = "auto"
                    self._mission.setdefault("llm_policy_update", {})
                    self._mission["llm_policy_update"]["timestamp"] = time.time()
                    self._save_mission()
                    try:
                        out_path = os.path.expanduser(self._mission_summary_out_path)
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(self._mission, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        self.get_logger().warn(f"[LLM] failed to write mission_summary_out: {e}")
                    if self._start_new_mission_after_llm:
                        self._start_new_mission(save=True)
                else:
                    self.get_logger().error(f"[LLM] postrun failed rc={r.returncode} stderr={(r.stderr or '')[:300]}")
                return

            preview_cmd = list(base_cmd) + ["--yes", "--dry_run"]
            r_preview, dt_preview = self._exec_llm_decay(preview_cmd, mode="human-preview", run_input=None)
            if r_preview.returncode != 0:
                self.get_logger().error(f"[LLM] postrun preview failed rc={r_preview.returncode} stderr={(r_preview.stderr or '')[:300]}")
                return

            proposal_updates = self._parse_llm_proposal_updates(r_preview.stdout or "")
            if not proposal_updates:
                self.get_logger().info(f"[LLM] no applicable updates in preview dt={dt_preview:.1f}s; nothing to apply")
                if self._start_new_mission_after_llm:
                    self._start_new_mission(save=True)
                return

            if mode == "human_all":
                auto_updates  = []
                human_updates = proposal_updates
            else:
                _thr = self._llm_decay_confidence_threshold
                auto_updates  = [(tag, ttl, conf) for tag, ttl, conf in proposal_updates if conf >= _thr]
                human_updates = [(tag, ttl, conf) for tag, ttl, conf in proposal_updates if conf < _thr]

            if auto_updates:
                auto_lines = [f"  - {tag}: ttl={ttl:.2f}s conf={conf:.2f}" for tag, ttl, conf in auto_updates]
                self.get_logger().info(f"[LLM] auto-applying {len(auto_updates)} high-confidence updates:\n" + "\n".join(auto_lines))
                self._apply_llm_proposal_updates_local([(tag, ttl) for tag, ttl, _ in auto_updates])
                self._load_decay_table()  
                self._append_archive_from_preview_local(
                    proposal_updates=[(tag, ttl) for tag, ttl, _ in auto_updates],
                    approval_status="auto",
                )

            if mode == "human_all":
                significant = [
                    (tag, max(0.1, float(self._decay_table.get(tag, {}).get("ttl", self._default_cost_ttl_s))),
                     ttl, abs(ttl - max(0.1, float(self._decay_table.get(tag, {}).get("ttl", self._default_cost_ttl_s)))) / max(0.1, float(self._decay_table.get(tag, {}).get("ttl", self._default_cost_ttl_s))) * 100.0, conf)
                    for tag, ttl, conf in (human_updates or [])
                ]
            else:
                significant = self._approval_significant_updates(human_updates) if human_updates else []
            approve = True
            if significant:
                reason_map = getattr(self, '_llm_reason_map', {})
                lines = []
                for tag, old, new, pct, conf in significant:
                    reason = reason_map.get(tag, "")
                    line = f"  - {tag}: old={old:.2f}s -> new={new:.2f}s ({pct:.2f}%) conf={conf:.2f}"
                    if reason:
                        line += f"\n    reason: {reason}"
                    lines.append(line)
                self.get_logger().info(
                    f"[LLM] {len(significant)} update(s) need human approval:\n" + "\n".join(lines)
                )
                try:
                    while True:
                        ans = input("[LLM] Apply these updates? [y/N] ").strip().lower()
                        if ans in ("y", "yes"):
                            approve = True
                            break
                        elif ans in ("n", "no"):
                            approve = False
                            break
                        elif ans == "":
                            continue  
                except EOFError:
                    approve = True
                    self.get_logger().info("[LLM] non-interactive mode: auto-approving")
                self.get_logger().info(f"[LLM] approved={approve}")
            else:
                if human_updates:
                    self.get_logger().info("[LLM] no significant human-approval updates; skipping")
                if auto_updates and not human_updates:
                    self._mission["llm_postrun_dt_s"] = float(dt_preview)
                    self._mission["llm_postrun_mode"] = mode
                    self._mission["llm_postrun_approved"] = True
                    self._record_llm_policy_update_from_preview(
                        [(tag, ttl) for tag, ttl, _ in auto_updates],
                        preview_stdout=(r_preview.stdout or "")
                    )
                    if self._start_new_mission_after_llm:
                        self._start_new_mission(save=True)
                    return
                approve = False 

            if not approve:
                feedback_raw = self._prompt_optional_feedback()
                self._mission["llm_postrun_dt_s"] = float(dt_preview)
                self._mission["llm_postrun_mode"] = mode
                self._mission["llm_postrun_approved"] = False
                self._record_llm_policy_update_from_preview(
                    proposal_updates,
                    preview_stdout=(r_preview.stdout or "")
                )
                self._append_archive_from_preview_local(
                    proposal_updates=proposal_updates,
                    approval_status="rejected",
                    human_feedback=feedback_raw,
                )
                self.get_logger().info("[LLM] postrun rejected by user; archived as rejected; not applied")
                self._save_mission()
                if self._start_new_mission_after_llm:
                    self._start_new_mission(save=True)
                return

            applied_n = self._apply_llm_proposal_updates_local(
                [(tag, ttl) for tag, ttl, *_ in proposal_updates]
            )
            if applied_n > 0:
                feedback_raw = self._prompt_optional_feedback()
                self._mission["llm_postrun_dt_s"] = float(dt_preview)
                self._mission["llm_postrun_mode"] = mode
                self._mission["llm_postrun_approved"] = True
                self._record_llm_policy_update_from_preview(
                    proposal_updates,
                    preview_stdout=(r_preview.stdout or "")
                )
                self._append_archive_from_preview_local(
                    proposal_updates=proposal_updates,
                    approval_status="approved",
                    human_feedback=feedback_raw,
                )
                self.get_logger().info(
                    f"[LLM] postrun done mode=human-preview-local-apply; "
                    f"applied={applied_n}; reloading decay table"
                )
                self._load_decay_table()
                self._save_mission()
                if self._start_new_mission_after_llm:
                    self._start_new_mission(save=True)
            else:
                self.get_logger().error("[LLM] local apply from preview failed; nothing applied")
                
        except Exception as e:
            self.get_logger().error(f"[LLM] postrun exception: {e}")
            self.get_logger().error(traceback.format_exc())

    def _scan_cb(self, msg: LaserScan):
        if not self._enabled or self._map_msg is None:
            return

        self._last_scan_stamp = msg.header.stamp

        if self._speed_classifier_enable and _SPEED_OK:
            try:
                self._sc_scan_queue.put_nowait(msg)
            except Exception:
                pass

        try:
            stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            trans = self.tf_buffer.lookup_transform(self._map_frame, msg.header.frame_id, stamp, timeout=Duration(seconds=0.1))
            base_trans = self.tf_buffer.lookup_transform(self._map_frame, self._base_frame, stamp, timeout=Duration(seconds=0.1))
        except Exception:
            if not self._detour_gate_open():
                self._prev_centroids = []
            return

        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        q = trans.transform.rotation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        cyaw = math.cos(yaw)
        syaw = math.sin(yaw)

        bx = base_trans.transform.translation.x
        by = base_trans.transform.translation.y

        unexpected_points_pre: List[Tuple[float, float]] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if not math.isfinite(r):
                angle += msg.angle_increment
                continue
            if r < self._min_range or r > self._max_range:
                angle += msg.angle_increment
                continue
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)
            wx = tx + cyaw * lx - syaw * ly
            wy = ty + syaw * lx + cyaw * ly
            angle += msg.angle_increment
            if self._is_unexpected_point(wx, wy):
                unexpected_points_pre.append((wx, wy))

        if self._speed_classifier_enable and unexpected_points_pre:
            pre_centroids = self._cluster_centroids(unexpected_points_pre)
            pre_centroids = [c for c in pre_centroids
                             if math.hypot(c[0]-bx, c[1]-by) >= self._min_centroid_robot_dist]
            now_pre = time.time()
            with self._latest_unexpected_lock:
                self._latest_unexpected_centroids = pre_centroids[:]
            with self._centroid_track_lock:
                matched = set()
                for cx, cy in pre_centroids:
                    if not self._is_in_corridor(cx, cy):
                        continue
                    best_id, best_d = None, float("inf")
                    for tid, tr in self._centroid_tracks.items():
                        if tid in matched:
                            continue
                        d = math.hypot(cx - tr["x"], cy - tr["y"])
                        if d < best_d:
                            best_d, best_id = d, tid
                    if best_id is not None and best_d < self._centroid_match_dist:
                        tr = self._centroid_tracks[best_id]
                        dt = now_pre - tr["t"]
                        if dt > 0.1:
                            spd = math.hypot(cx - tr["x"], cy - tr["y"]) / dt
                            tr["samples"].append(spd)
                            if len(tr["samples"]) > 30:
                                tr["samples"] = tr["samples"][-30:]
                        tr["x"], tr["y"], tr["t"] = cx, cy, now_pre
                        matched.add(best_id)
                    else:
                        self._centroid_track_id += 1
                        self._centroid_tracks[self._centroid_track_id] = {
                            "x": cx, "y": cy, "t": now_pre, "samples": []
                        }
                dead = [tid for tid, tr in self._centroid_tracks.items()
                        if now_pre - tr["t"] > 3.0]
                for tid in dead:
                    del self._centroid_tracks[tid]

        if not self._detour_gate_open():
            self._prev_centroids = []
            return

        unexpected_points: List[Tuple[float, float]] = unexpected_points_pre

        centroids = self._cluster_centroids(unexpected_points)
        centroids = [c for c in centroids if math.hypot(c[0] - bx, c[1] - by) >= self._min_centroid_robot_dist]

        if self._confirm:
            centroids = [c for c in centroids if self._matches_previous(c)]

        raw_n = len(centroids)
        centroids = self._merge_close_centroids(centroids, 1.0)

        self.get_logger().debug(
            f"[CENTROID_MERGE] raw={raw_n} merged={len(centroids)} centroids={centroids}"
        )

        self._prev_centroids = centroids[:]

        now = time.time()
        if not self._speed_stop_active:
            if centroids and not self._active_costs:
                rx, ry = self._current_pose_xy
                closest = min(centroids, key=lambda c: math.hypot(c[0]-rx, c[1]-ry))
                self._upsert_active_costs([closest], now)

        self.get_logger().debug(
            f"unexpected points={len(unexpected_points)} detections={len(centroids)} active_costs={len(self._active_costs)} published_to={self._output_topic}"
        )

    def _throttled_log(self, msg: str):
        now = time.time()
        if now - self._last_debug_log >= self._debug_log_interval_s:
            self.get_logger().info(msg)
            self._last_debug_log = now

    def _matches_previous(self, c: Tuple[float, float]) -> bool:
        if not self._prev_centroids:
            return False
        return any(math.hypot(c[0] - px, c[1] - py) <= self._confirm_dist for px, py in self._prev_centroids)

    def _cluster_centroids(self, pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not pts:
            return []

        clusters: List[List[Tuple[float, float]]] = []
        current = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - current[-1][0], p[1] - current[-1][1]) <= self._cluster_dist:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)

        centroids: List[Tuple[float, float]] = []
        for cluster in clusters:
            if len(cluster) < self._cluster_min_points:
                continue
            sx = sum(p[0] for p in cluster)
            sy = sum(p[1] for p in cluster)
            n = float(len(cluster))
            centroids.append((sx / n, sy / n))
        return centroids

    def _is_unexpected_point(self, wx: float, wy: float) -> bool:
        grid = self._map_msg
        if grid is None:
            return False
        m = self._world_to_map(wx, wy)
        if m is None:
            return False
        mx, my = m
        idx = my * grid.info.width + mx
        v = grid.data[idx]

        if v < 0:
            return not self._exclude_unknown
        if v >= self._occupied_threshold:
            return False

        if self._occupied_margin_cells > 0 and self._near_occupied(mx, my, self._occupied_margin_cells):
            return False
        return True

    def _near_occupied(self, mx: int, my: int, radius_cells: int) -> bool:
        grid = self._map_msg
        if grid is None:
            return False
        w = int(grid.info.width)
        h = int(grid.info.height)
        for yy in range(max(0, my - radius_cells), min(h, my + radius_cells + 1)):
            row = yy * w
            for xx in range(max(0, mx - radius_cells), min(w, mx + radius_cells + 1)):
                if grid.data[row + xx] >= self._occupied_threshold:
                    return True
        return False

    def _world_to_map(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        grid = self._map_msg
        if grid is None:
            return None
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        res = grid.info.resolution
        mx = int((wx - origin_x) / res)
        my = int((wy - origin_y) / res)
        if mx < 0 or my < 0 or mx >= int(grid.info.width) or my >= int(grid.info.height):
            return None
        return mx, my


def main():
    rclpy.init()
    detector = UnexpectedObstacleDetector()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(detector)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
