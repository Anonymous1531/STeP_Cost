from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger

import tf2_ros


_HERE = pathlib.Path(__file__).resolve().parent.parent.parent.parent


def _default_script_path(filename: str) -> str:
    """Resolve helper scripts both from a source checkout and a normal user clone."""
    candidates = [
        _HERE / filename,
        pathlib.Path.home() / "STeP_Cost" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])

try:
    from cv_bridge import CvBridge
    import cv2

    _CAPTURE_OK = True
except Exception:
    CvBridge = None
    cv2 = None
    _CAPTURE_OK = False


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

TAG_KEY_TOKEN_RE = re.compile(r"\s+")


def normalize_tag_key(key: Any) -> str:
    """Normalize a semantic or semantic-motion tag without inventing new tags."""
    if key is None:
        return ""

    k = str(key).strip()
    if not k:
        return ""

    k = TAG_KEY_TOKEN_RE.split(k, 1)[0]
    k = k.strip().strip(",;").lower().replace("|", ":")

    aliases = {
        "human": "person",
        "pedestrian": "person",
        "worker": "person",
        "staff": "person",
        "people": "person",
        "fork": "forklift",
        "forklifttruck": "forklift",
        "lifttruck": "forklift",
        "truck": "forklift",
        "cartlike": "cart",
        "trolley": "cart",
        "dolly": "cart",
        "handtruck": "cart",
        "shoppingcart": "cart",
    }

    if ":" in k:
        base, motion = k.split(":", 1)
        base = aliases.get(base, base)
        if motion == "still":
            motion = "static"
        return f"{base}:{motion}"

    return aliases.get(k, k)


def extract_vlm_tag_key(vlm_field: Any) -> str:
    """Extract the base semantic tag from a VLM JSON object/string."""
    if vlm_field is None:
        return ""

    if isinstance(vlm_field, dict):
        raw = vlm_field.get("tag_key") or vlm_field.get("tag")
        key = normalize_tag_key(raw)
    elif isinstance(vlm_field, str):
        raw = vlm_field.strip()
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            key = normalize_tag_key(obj.get("tag_key") or obj.get("tag"))
        else:
            key = normalize_tag_key(raw)
    else:
        return ""

    # VLM is responsible only for the base semantic category.
    if ":" in key:
        key = key.split(":", 1)[0]
    return key


def _yaw_from_quaternion(q: Any) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _median(values: List[float]) -> float:
    if not values:
        return -1.0
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _atomic_json_write(path: str, obj: Any) -> None:
    path = os.path.expanduser(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str, default: Any) -> Any:
    path = os.path.expanduser(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Category-conditioned 1-D two-component GMM
# -----------------------------------------------------------------------------


class _SingleGMM:
    """Two-component 1-D GMM with median-split initialization."""

    def __init__(self, min_samples: int = 10):
        self.min_samples = int(min_samples)
        self._samples: List[float] = []
        self._mu = [0.0, 0.0]
        self._sigma = [1.0, 1.0]
        self._pi = [0.5, 0.5]
        self._fitted = False

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def is_ready(self) -> bool:
        return self._fitted

    def add_sample(self, speed_mps: float) -> None:
        x = float(speed_mps)
        if not math.isfinite(x) or x < 0.0:
            return
        self._samples.append(x)
        if len(self._samples) >= self.min_samples:
            self._fit()

    @staticmethod
    def _gauss(x: float, mu: float, sigma: float) -> float:
        sigma = max(1e-6, float(sigma))
        z = (x - mu) / sigma
        return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))

    def _fit(self) -> None:
        X = list(self._samples)
        n = len(X)
        if n < 2:
            return

        sorted_x = sorted(X)
        mid = sorted_x[n // 2]
        slow_vals = [x for x in X if x <= mid]
        fast_vals = [x for x in X if x > mid]

        mu0 = sum(slow_vals) / max(1, len(slow_vals))
        mu1 = sum(fast_vals) / max(1, len(fast_vals))

        if mu0 >= mu1:
            mu0 = sorted_x[n // 4]
            mu1 = sorted_x[(3 * n) // 4]

        sigma0 = sigma1 = max(0.01, abs(mu1 - mu0) / 4.0)
        pi0 = pi1 = 0.5

        for _ in range(50):
            r0: List[float] = []
            r1: List[float] = []

            for x in X:
                p0 = pi0 * self._gauss(x, mu0, sigma0)
                p1 = pi1 * self._gauss(x, mu1, sigma1)
                total = p0 + p1 + 1e-12
                r0.append(p0 / total)
                r1.append(p1 / total)

            n0 = sum(r0) + 1e-12
            n1 = sum(r1) + 1e-12

            mu0_new = sum(r * x for r, x in zip(r0, X)) / n0
            mu1_new = sum(r * x for r, x in zip(r1, X)) / n1

            sigma0_new = math.sqrt(
                sum(r * (x - mu0_new) ** 2 for r, x in zip(r0, X)) / n0
            ) + 1e-4
            sigma1_new = math.sqrt(
                sum(r * (x - mu1_new) ** 2 for r, x in zip(r1, X)) / n1
            ) + 1e-4

            converged = (
                abs(mu0_new - mu0) < 1e-5
                and abs(mu1_new - mu1) < 1e-5
            )

            mu0, mu1 = mu0_new, mu1_new
            sigma0, sigma1 = sigma0_new, sigma1_new
            pi0, pi1 = n0 / n, n1 / n

            if converged:
                break

        if mu0 > mu1:
            mu0, mu1 = mu1, mu0
            sigma0, sigma1 = sigma1, sigma0
            pi0, pi1 = pi1, pi0

        self._mu = [mu0, mu1]
        self._sigma = [sigma0, sigma1]
        self._pi = [pi0, pi1]
        self._fitted = True

    def predict(self, speed_mps: float) -> str:
        # Direct calls before readiness retain the historical fast fallback.
        if not self._fitted:
            return "fast"

        x = float(speed_mps)
        q_slow = self._pi[0] * self._gauss(x, self._mu[0], self._sigma[0])
        q_fast = self._pi[1] * self._gauss(x, self._mu[1], self._sigma[1])
        return "slow" if q_slow > q_fast else "fast"

    def summary(self) -> str:
        if not self._fitted:
            return f"not_fitted(n={self.n_samples})"
        return (
            f"n={self.n_samples} "
            f"slow_mu={self._mu[0]:.3f}(sigma={self._sigma[0]:.3f}) "
            f"fast_mu={self._mu[1]:.3f}(sigma={self._sigma[1]:.3f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples": list(self._samples),
            "fitted": self._fitted,
            "mu": list(self._mu),
            "sigma": list(self._sigma),
            "pi": list(self._pi),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], min_samples: int) -> "_SingleGMM":
        obj = cls(min_samples=min_samples)
        obj._samples = [float(x) for x in d.get("samples", [])]
        obj._fitted = bool(d.get("fitted", False))
        obj._mu = [float(x) for x in d.get("mu", [0.0, 0.0])]
        obj._sigma = [float(x) for x in d.get("sigma", [1.0, 1.0])]
        obj._pi = [float(x) for x in d.get("pi", [0.5, 0.5])]
        return obj


class _TagGMM:
    """One two-component GMM per base semantic category."""

    def __init__(self, min_samples: int, save_path: str):
        self.min_samples = int(min_samples)
        self.save_path = os.path.expanduser(save_path)
        self._gmms: Dict[str, _SingleGMM] = {}
        self._lock = threading.RLock()

    def _get_or_create(self, base_tag: str) -> _SingleGMM:
        base_tag = normalize_tag_key(base_tag).split(":", 1)[0]
        if base_tag not in self._gmms:
            self._gmms[base_tag] = _SingleGMM(self.min_samples)
        return self._gmms[base_tag]

    def add_sample(self, base_tag: str, speed_mps: float) -> None:
        with self._lock:
            self._get_or_create(base_tag).add_sample(speed_mps)
            self._save_locked()

    def predict(self, base_tag: str, speed_mps: float) -> str:
        with self._lock:
            return self._get_or_create(base_tag).predict(speed_mps)

    def is_ready(self, base_tag: str) -> bool:
        key = normalize_tag_key(base_tag).split(":", 1)[0]
        with self._lock:
            gmm = self._gmms.get(key)
            return bool(gmm and gmm.is_ready)

    def n_samples(self, base_tag: str) -> int:
        key = normalize_tag_key(base_tag).split(":", 1)[0]
        with self._lock:
            gmm = self._gmms.get(key)
            return gmm.n_samples if gmm else 0

    def summary(self) -> str:
        with self._lock:
            if not self._gmms:
                return "  (empty)"
            return "\n".join(
                f"  {tag}: {gmm.summary()}"
                for tag, gmm in sorted(self._gmms.items())
            )

    def _save_locked(self) -> None:
        if not self.save_path:
            return
        try:
            payload = {tag: gmm.to_dict() for tag, gmm in self._gmms.items()}
            _atomic_json_write(self.save_path, payload)
        except Exception:
            pass

    def load(self) -> None:
        if not self.save_path:
            return
        data = _read_json(self.save_path, {})
        if not isinstance(data, dict):
            return
        with self._lock:
            for tag, row in data.items():
                if isinstance(row, dict):
                    self._gmms[normalize_tag_key(tag).split(":", 1)[0]] = _SingleGMM.from_dict(
                        row, self.min_samples
                    )


# -----------------------------------------------------------------------------
# Runtime state
# -----------------------------------------------------------------------------


@dataclass
class ActiveCost:
    cost_id: int
    x: float
    y: float
    ttl_s: float
    expires_at: float
    inserted_at: float
    event_id: str
    tag_key: str = ""
    tag_group_id: str = ""


@dataclass
class TrackState:
    track_id: int
    x: float
    y: float
    t: float
    speed_samples: List[float]


# -----------------------------------------------------------------------------
# Policy bridge
# -----------------------------------------------------------------------------


class UnexpectedObstacleDetector(Node):
    """
    Core STeP-Cost runtime bridge.

    Responsibilities:
      * detect relative global-plan length increases,
      * identify path-relevant unexpected LiDAR obstacle centroids,
      * capture the event-time RGB frame,
      * obtain a base semantic category from the VLM,
      * obtain a category-conditioned slow/fast label from a GMM,
      * apply tag-wise residual-cost TTL with optional depth correction,
      * maintain mission summaries,
      * perform post-mission LLM proposal generation and selective review.

    Nav2 remains responsible for global planning and control. This node only
    publishes active residual obstacle positions to the custom costmap layer.
    """

    def __init__(self):
        super().__init__("unexpected_obstacle_detector")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("enabled", True)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("plan_topic", "/plan")
        self.declare_parameter("pose_topic", "/amcl_pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("output_topic", "/object_world_positions")

        # Detour gate. detour_ratio_threshold is a multiplicative factor:
        # 1.15 <=> a relative plan-length increase of 0.15.
        self.declare_parameter("gate_on_detour_only", True)
        self.declare_parameter("detour_ratio_threshold", 1.15)
        self.declare_parameter("detour_min_previous_length_m", 1.0)
        self.declare_parameter("detour_hold_s", 8.0)
        self.declare_parameter("detour_cooldown_s", 1.0)

        # Unexpected-obstacle extraction from LiDAR + static occupancy map.
        self.declare_parameter("min_range_m", 0.10)
        self.declare_parameter("max_range_m", 6.0)
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("exclude_unknown", True)
        self.declare_parameter("occupied_margin_cells", 10)
        self.declare_parameter("cluster_dist_thresh", 0.20)
        self.declare_parameter("cluster_min_points", 5)
        self.declare_parameter("min_centroid_robot_distance_m", 0.25)
        self.declare_parameter("centroid_match_dist_m", 1.5)
        self.declare_parameter("centroid_track_max_age_s", 3.0)
        self.declare_parameter("speed_sample_min_dt_s", 0.10)
        self.declare_parameter("speed_sample_window", 30)

        # Residual-cost lifetime.
        self.declare_parameter("default_cost_ttl_s", 6.0)
        self.declare_parameter("base_ttl_min_s", 0.1)
        self.declare_parameter("base_ttl_max_s", 600.0)
        self.declare_parameter("applied_ttl_min_s", 0.5)
        self.declare_parameter("maintain_rate_hz", 8.0)
        self.declare_parameter("publish_empty_when_no_active", True)
        self.declare_parameter("enable_global_clear_on_expire", False)
        self.declare_parameter("clear_service", "/global_costmap/clear_entirely_global_costmap")
        self.declare_parameter("clear_service_wait_s", 0.2)

        # Corridor metadata for the x-axis Factory-style corridor depth ratio.
        # Keep map-specific coordinates configurable; do not hard-code a paper map.
        self.declare_parameter("corridor_start_x", 0.0)
        self.declare_parameter("corridor_end_x", 32.0)
        self.declare_parameter("corridor_y_centers", [0.0, -4.76, -13.49, -18.17])
        self.declare_parameter("corridor_y_half_width", 2.5)
        self.declare_parameter("corridor_x_margin", 1.0)
        self.declare_parameter("depth_correction_enable", True)

        # VLM: semantic category only.
        self.declare_parameter("enable_capture", True)
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("sample_hz", 6.0)
        self.declare_parameter("frame_buffer_size", 32)
        self.declare_parameter("save_root", os.path.expanduser("~/.ros/detour_events"))
        self.declare_parameter("vlm_enable", False)
        self.declare_parameter("vlm_python", "python3")
        self.declare_parameter("vlm_script", _default_script_path("vlm_gemini_v1.py"))
        self.declare_parameter("vlm_model", "gemini-2.5-flash")
        self.declare_parameter(
            "vlm_prompt",
            (
                "Identify the main obstacle's semantic object type. "
                "Return only a base semantic category from the allowed tag vocabulary "
                "and concise visual evidence. Do not include a motion class."
            ),
        )
        self.declare_parameter("vlm_timeout_sec", 180.0)
        self.declare_parameter("vlm_result_topic", "/vlm/result")
        self.declare_parameter("vlm_singleflight", True)
        self.declare_parameter("vlm_decay_table", os.path.expanduser("~/.ros/decay_table.json"))
        self.declare_parameter("debug_vlm_stdout_chars", 350)
        self.declare_parameter("debug_vlm_stderr_chars", 350)

        # Category-conditioned motion classification.
        self.declare_parameter("speed_classifier_enable", True)
        self.declare_parameter("gmm_min_samples", 10)
        self.declare_parameter("gmm_samples_path", os.path.expanduser("~/.ros/gmm_samples.json"))
        self.declare_parameter("gmm_freeze", False)
        self.declare_parameter("speed_threshold_mps", 0.3)

        # Mission / post-mission LLM update.
        self.declare_parameter("mission_summary_path", os.path.expanduser("~/.ros/mission_summary.json"))
        self.declare_parameter("mission_summary_out_path", os.path.expanduser("~/.ros/mission_summary_out.json"))
        self.declare_parameter("llm_decay_enable", False)
        self.declare_parameter("llm_decay_python", "python3")
        self.declare_parameter("llm_decay_script", _default_script_path("llm_decay_gemini_v3.py"))
        self.declare_parameter("llm_decay_model", "gemini-2.5-flash")
        self.declare_parameter("llm_decay_result_topic", "/llm_decay/result")
        self.declare_parameter("llm_decay_singleflight", True)
        self.declare_parameter("llm_decay_rag_enable", False)
        self.declare_parameter(
            "llm_decay_retrieval_archive_path",
            os.path.expanduser("~/.ros/llm_decay_rag_archive.json"),
        )
        self.declare_parameter("llm_decay_retrieval_max_repeat1_cases", 30)
        self.declare_parameter("llm_decay_append_to_archive", True)
        self.declare_parameter("llm_decay_archive_max_cases", 2000)
        self.declare_parameter("llm_decay_approval_mode", "ours")
        self.declare_parameter("llm_decay_confidence_threshold", 0.9)
        self.declare_parameter("start_new_mission_after_llm", True)
        self.declare_parameter("postrun_on_goal_success", False)
        self.declare_parameter("goal_status_topic", "/navigate_to_pose/_action/status")
        self.declare_parameter("through_poses_status_topic", "/navigate_through_poses/_action/status")
        self.declare_parameter("goal_success_cooldown_s", 5.0)
        self.declare_parameter("llm_stdout_log_chars", 2000)
        self.declare_parameter("llm_stderr_log_chars", 2000)

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        self._enabled = bool(self.get_parameter("enabled").value)
        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._map_topic = str(self.get_parameter("map_topic").value)
        self._plan_topic = str(self.get_parameter("plan_topic").value)
        self._pose_topic = str(self.get_parameter("pose_topic").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._output_topic = str(self.get_parameter("output_topic").value)

        self._gate_on_detour_only = bool(self.get_parameter("gate_on_detour_only").value)
        self._detour_ratio_threshold = float(self.get_parameter("detour_ratio_threshold").value)
        self._detour_min_previous_length = float(
            self.get_parameter("detour_min_previous_length_m").value
        )
        self._detour_hold_s = float(self.get_parameter("detour_hold_s").value)
        self._detour_cooldown_s = float(self.get_parameter("detour_cooldown_s").value)

        self._min_range = float(self.get_parameter("min_range_m").value)
        self._max_range = float(self.get_parameter("max_range_m").value)
        self._occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self._exclude_unknown = bool(self.get_parameter("exclude_unknown").value)
        self._occupied_margin_cells = int(self.get_parameter("occupied_margin_cells").value)
        self._cluster_dist = float(self.get_parameter("cluster_dist_thresh").value)
        self._cluster_min_points = int(self.get_parameter("cluster_min_points").value)
        self._min_centroid_robot_dist = float(
            self.get_parameter("min_centroid_robot_distance_m").value
        )
        self._centroid_match_dist = float(self.get_parameter("centroid_match_dist_m").value)
        self._centroid_track_max_age_s = float(
            self.get_parameter("centroid_track_max_age_s").value
        )
        self._speed_sample_min_dt_s = float(self.get_parameter("speed_sample_min_dt_s").value)
        self._speed_sample_window = max(1, int(self.get_parameter("speed_sample_window").value))

        self._default_cost_ttl_s = float(self.get_parameter("default_cost_ttl_s").value)
        self._base_ttl_min_s = float(self.get_parameter("base_ttl_min_s").value)
        self._base_ttl_max_s = float(self.get_parameter("base_ttl_max_s").value)
        self._applied_ttl_min_s = float(self.get_parameter("applied_ttl_min_s").value)
        self._maintain_rate_hz = max(0.5, float(self.get_parameter("maintain_rate_hz").value))
        self._publish_empty_when_no_active = bool(
            self.get_parameter("publish_empty_when_no_active").value
        )
        self._enable_global_clear_on_expire = bool(
            self.get_parameter("enable_global_clear_on_expire").value
        )
        self._clear_service_name = str(self.get_parameter("clear_service").value)
        self._clear_service_wait_s = float(self.get_parameter("clear_service_wait_s").value)

        self._corridor_start_x = float(self.get_parameter("corridor_start_x").value)
        self._corridor_end_x = float(self.get_parameter("corridor_end_x").value)
        self._corridor_y_centers = [
            float(v) for v in list(self.get_parameter("corridor_y_centers").value)
        ]
        self._corridor_y_half_width = float(self.get_parameter("corridor_y_half_width").value)
        self._corridor_x_margin = float(self.get_parameter("corridor_x_margin").value)
        self._depth_correction_enable = bool(
            self.get_parameter("depth_correction_enable").value
        )

        self._enable_capture = bool(self.get_parameter("enable_capture").value)
        self._camera_topic = str(self.get_parameter("camera_topic").value)
        self._sample_hz = float(self.get_parameter("sample_hz").value)
        self._frame_buffer_size = max(4, int(self.get_parameter("frame_buffer_size").value))
        self._save_root = os.path.expanduser(str(self.get_parameter("save_root").value))
        self._vlm_enable = bool(self.get_parameter("vlm_enable").value)
        self._vlm_python = str(self.get_parameter("vlm_python").value)
        self._vlm_script = os.path.expanduser(str(self.get_parameter("vlm_script").value))
        self._vlm_model = str(self.get_parameter("vlm_model").value)
        self._vlm_prompt = str(self.get_parameter("vlm_prompt").value)
        self._vlm_timeout_sec = float(self.get_parameter("vlm_timeout_sec").value)
        self._vlm_result_topic = str(self.get_parameter("vlm_result_topic").value)
        self._vlm_singleflight = bool(self.get_parameter("vlm_singleflight").value)
        self._vlm_decay_table = os.path.expanduser(
            str(self.get_parameter("vlm_decay_table").value)
        )
        self._debug_vlm_stdout_chars = int(self.get_parameter("debug_vlm_stdout_chars").value)
        self._debug_vlm_stderr_chars = int(self.get_parameter("debug_vlm_stderr_chars").value)

        self._speed_classifier_enable = bool(
            self.get_parameter("speed_classifier_enable").value
        )
        self._gmm_min_samples = int(self.get_parameter("gmm_min_samples").value)
        self._gmm_samples_path = os.path.expanduser(
            str(self.get_parameter("gmm_samples_path").value)
        )
        self._gmm_freeze = bool(self.get_parameter("gmm_freeze").value)
        self._speed_threshold_mps = float(self.get_parameter("speed_threshold_mps").value)

        self._mission_summary_path = os.path.expanduser(
            str(self.get_parameter("mission_summary_path").value)
        )
        self._mission_summary_out_path = os.path.expanduser(
            str(self.get_parameter("mission_summary_out_path").value)
        )
        self._llm_decay_enable = bool(self.get_parameter("llm_decay_enable").value)
        self._llm_decay_python = str(self.get_parameter("llm_decay_python").value)
        self._llm_decay_script = os.path.expanduser(
            str(self.get_parameter("llm_decay_script").value)
        )
        self._llm_decay_model = str(self.get_parameter("llm_decay_model").value)
        self._llm_decay_result_topic = str(
            self.get_parameter("llm_decay_result_topic").value
        )
        self._llm_decay_singleflight = bool(
            self.get_parameter("llm_decay_singleflight").value
        )
        self._llm_decay_rag_enable = bool(
            self.get_parameter("llm_decay_rag_enable").value
        )
        self._llm_decay_retrieval_archive_path = os.path.expanduser(
            str(self.get_parameter("llm_decay_retrieval_archive_path").value)
        )
        self._llm_decay_retrieval_max_repeat1_cases = int(
            self.get_parameter("llm_decay_retrieval_max_repeat1_cases").value
        )
        self._llm_decay_append_to_archive = bool(
            self.get_parameter("llm_decay_append_to_archive").value
        )
        self._llm_decay_archive_max_cases = int(
            self.get_parameter("llm_decay_archive_max_cases").value
        )
        mode = str(self.get_parameter("llm_decay_approval_mode").value).strip().lower()
        if mode == "human":
            mode = "ours"  # backward-compatible alias
        if mode not in ("auto", "ours", "human_all"):
            self.get_logger().warn(
                f"[LLM] invalid llm_decay_approval_mode={mode!r}; fallback to 'ours'"
            )
            mode = "ours"
        self._llm_decay_approval_mode = mode
        self._llm_decay_confidence_threshold = float(
            self.get_parameter("llm_decay_confidence_threshold").value
        )
        self._start_new_mission_after_llm = bool(
            self.get_parameter("start_new_mission_after_llm").value
        )
        self._postrun_on_goal_success = bool(
            self.get_parameter("postrun_on_goal_success").value
        )
        self._goal_status_topic = str(self.get_parameter("goal_status_topic").value)
        self._through_goal_status_topic = str(
            self.get_parameter("through_poses_status_topic").value
        )
        self._goal_success_cooldown_s = float(
            self.get_parameter("goal_success_cooldown_s").value
        )
        self._llm_stdout_log_chars = max(
            200, int(self.get_parameter("llm_stdout_log_chars").value)
        )
        self._llm_stderr_log_chars = max(
            200, int(self.get_parameter("llm_stderr_log_chars").value)
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._map_msg: Optional[OccupancyGrid] = None
        self._current_pose_xy: Tuple[float, float] = (0.0, 0.0)
        self._previous_path_length = 0.0
        self._last_detour_trigger = 0.0
        self._detour_active_until = 0.0
        self._latest_unexpected_centroids: List[Tuple[float, float]] = []
        self._latest_unexpected_lock = threading.RLock()
        self._last_scan_stamp = None

        self._centroid_tracks: Dict[int, TrackState] = {}
        self._centroid_track_lock = threading.RLock()
        self._centroid_track_id = 0

        self._active_costs: Dict[int, ActiveCost] = {}
        self._active_cost_lock = threading.RLock()
        self._next_cost_id = 1
        self._last_publish_empty = False
        self._clear_inflight = False
        self._clear_pending_republish = False

        self._frame_lock = threading.RLock()
        self._frame_buf: Deque[Tuple[float, Any]] = deque(maxlen=self._frame_buffer_size)
        self._last_store_time = 0.0
        self._cv_bridge = CvBridge() if _CAPTURE_OK else None
        self._vlm_guard = threading.Lock()

        self._event_tag_map: Dict[str, str] = {}
        self._event_speed_mps: Dict[str, float] = {}
        self._event_depth_bounds: Dict[str, Tuple[float, float]] = {}
        self._current_event_id = ""

        self._mission_lock = threading.RLock()
        self._mission: Dict[str, Any] = self._new_mission_template()

        self._seen_goal_success_ids: set[str] = set()
        self._last_goal_success_time = 0.0
        self._llm_decay_guard = threading.Lock()
        self._llm_decay_running = False
        self._llm_reason_map: Dict[str, str] = {}
        self._decay_table: Dict[str, Dict[str, float]] = {}

        self._gmm = _TagGMM(
            min_samples=self._gmm_min_samples,
            save_path=self._gmm_samples_path,
        )
        self._gmm.load()

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        if self._enable_capture or self._vlm_enable:
            os.makedirs(self._save_root, exist_ok=True)
            if not _CAPTURE_OK:
                self.get_logger().warn(
                    "[VLM] cv_bridge/OpenCV unavailable; camera capture disabled"
                )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.object_pub = self.create_publisher(PoseArray, self._output_topic, 10)
        self.vlm_pub = self.create_publisher(String, self._vlm_result_topic, 10)
        self.llm_decay_pub = self.create_publisher(String, self._llm_decay_result_topic, 10)

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, self._map_topic, self._map_cb, map_qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, self._scan_topic, self._scan_cb, qos_profile_sensor_data
        )
        self.plan_sub = self.create_subscription(Path, self._plan_topic, self._plan_cb, 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, self._pose_topic, self._pose_cb, 10
        )

        if self._enable_capture or self._vlm_enable:
            self.image_sub = self.create_subscription(
                Image, self._camera_topic, self._image_cb, qos_profile_sensor_data
            )

        if self._postrun_on_goal_success:
            self.goal_status_sub = self.create_subscription(
                GoalStatusArray, self._goal_status_topic, self._goal_status_cb, 10
            )
            self.through_goal_status_sub = self.create_subscription(
                GoalStatusArray,
                self._through_goal_status_topic,
                self._goal_status_cb,
                10,
            )

        self.maintain_timer = self.create_timer(
            1.0 / self._maintain_rate_hz, self._maintain_cb
        )

        self.clear_client = self.create_client(
            ClearEntireCostmap, self._clear_service_name
        )

        self._reset_mission_srv = self.create_service(
            Trigger,
            "/unexpected_obstacle_detector/reset_mission",
            self._reset_mission_cb,
        )
        self._run_postrun_srv = self.create_service(
            Trigger,
            "/unexpected_obstacle_detector/run_postrun_llm",
            self._run_postrun_llm_cb,
        )

        self._load_decay_table()
        self._save_mission()

        self.get_logger().info(
            "STeP-Cost policy bridge started: "
            f"detour_factor={self._detour_ratio_threshold:.3f}, "
            f"cooldown={self._detour_cooldown_s:.1f}s, "
            f"expiry_hz={self._maintain_rate_hz:.1f}, "
            f"review_mode={self._llm_decay_approval_mode}, "
            f"review_threshold={self._llm_decay_confidence_threshold:.2f}"
        )
        self.get_logger().info(f"[GMM]\n{self._gmm.summary()}")

    # ------------------------------------------------------------------
    # Basic state and file helpers
    # ------------------------------------------------------------------

    def _new_mission_template(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "mission_id": f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "started_at": now,
            "started_at_local": datetime.now().isoformat(),
            "events": [],
            "tag_groups": {},
        }

    def _current_mission_id(self) -> str:
        with self._mission_lock:
            return str(self._mission.get("mission_id") or "unknown_mission")

    def _make_tag_group_id(self, tag_key: str) -> str:
        key = normalize_tag_key(tag_key)
        return f"{self._current_mission_id()}::{key}" if key else ""

    def _rebuild_mission_tag_groups_locked(self) -> None:
        groups: Dict[str, Dict[str, Any]] = {}
        for ev in self._mission.get("events", []) or []:
            gid = ev.get("tag_group_id")
            if not gid:
                continue
            row = groups.setdefault(
                gid,
                {
                    "tag_key": ev.get("vlm_tag_key"),
                    "count": 0,
                    "events": [],
                },
            )
            row["count"] += 1
            row["events"].append(ev.get("event"))
        self._mission["tag_groups"] = groups

    def _save_mission(self) -> None:
        try:
            with self._mission_lock:
                snapshot = json.loads(json.dumps(self._mission))
            _atomic_json_write(self._mission_summary_path, snapshot)
        except Exception as e:
            self.get_logger().error(f"[MISSION SAVE ERROR] {e}")

    def _start_new_mission(self, save: bool = True) -> None:
        with self._mission_lock:
            self._mission = self._new_mission_template()
        if save:
            self._save_mission()

    def _mission_append_event(self, ev: Dict[str, Any]) -> None:
        with self._mission_lock:
            self._mission.setdefault("events", []).append(ev)
            self._rebuild_mission_tag_groups_locked()
        self._save_mission()

    def _mission_update_event(self, event_id: str, patch: Dict[str, Any]) -> None:
        with self._mission_lock:
            for ev in self._mission.get("events", []) or []:
                if ev.get("event") == event_id:
                    ev.update(patch)
                    break
            self._rebuild_mission_tag_groups_locked()
        self._save_mission()

    def _count_tag_group_in_mission(self, tag_group_id: str) -> int:
        if not tag_group_id:
            return 0
        with self._mission_lock:
            return sum(
                1
                for ev in self._mission.get("events", []) or []
                if ev.get("tag_group_id") == tag_group_id
            )

    def _clamp_base_ttl(self, ttl_s: float) -> float:
        return min(
            self._base_ttl_max_s,
            max(self._base_ttl_min_s, float(ttl_s)),
        )

    def _load_decay_table(self) -> None:
        raw = _read_json(self._vlm_decay_table, {})
        table: Dict[str, Dict[str, float]] = {}

        if isinstance(raw, dict):
            for raw_key, raw_value in raw.items():
                key = normalize_tag_key(raw_key)
                if not key:
                    continue
                try:
                    if isinstance(raw_value, dict):
                        ttl = raw_value.get("ttl", self._default_cost_ttl_s)
                    else:
                        ttl = raw_value
                    table[key] = {"ttl": self._clamp_base_ttl(float(ttl))}
                except Exception:
                    continue

        self._decay_table = table

        if not os.path.exists(self._vlm_decay_table):
            try:
                _atomic_json_write(self._vlm_decay_table, {})
            except Exception:
                pass

        self.get_logger().info(
            f"[DECAY] loaded tags={len(self._decay_table)} from {self._vlm_decay_table}"
        )

    def _lookup_decay_ttl(self, tag_key: str) -> float:
        key = normalize_tag_key(tag_key)
        row = self._decay_table.get(key)
        if isinstance(row, dict):
            try:
                ttl = self._clamp_base_ttl(row.get("ttl", self._default_cost_ttl_s))
                self.get_logger().info(f"[DECAY] hit tag={key} ttl={ttl:.2f}s")
                return ttl
            except Exception:
                pass

        # No semantic-base fallback: the policy is compound-tag indexed.
        ttl = self._clamp_base_ttl(self._default_cost_ttl_s)
        self.get_logger().warn(
            f"[DECAY] miss compound tag={key!r} -> default base ttl={ttl:.2f}s"
        )
        return ttl

    # ------------------------------------------------------------------
    # ROS callbacks: map / pose / image / scan
    # ------------------------------------------------------------------

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self._map_msg = msg

    def _pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        self._current_pose_xy = (float(p.x), float(p.y))

    def _image_cb(self, msg: Image) -> None:
        if not (_CAPTURE_OK and self._cv_bridge and (self._enable_capture or self._vlm_enable)):
            return

        now = time.time()
        if self._sample_hz > 0.0 and now - self._last_store_time < 1.0 / self._sample_hz:
            return
        self._last_store_time = now

        try:
            img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().debug(f"[IMAGE] conversion failed: {e}")
            return

        with self._frame_lock:
            self._frame_buf.append((now, img))

    def _scan_cb(self, msg: LaserScan) -> None:
        if not self._enabled or self._map_msg is None:
            return

        self._last_scan_stamp = msg.header.stamp

        try:
            stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            trans = self.tf_buffer.lookup_transform(
                self._map_frame,
                msg.header.frame_id,
                stamp,
                timeout=Duration(seconds=0.1),
            )
        except Exception:
            return

        tx = float(trans.transform.translation.x)
        ty = float(trans.transform.translation.y)
        yaw = _yaw_from_quaternion(trans.transform.rotation)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        unexpected_points: List[Tuple[float, float]] = []
        angle = float(msg.angle_min)

        for r in msg.ranges:
            if math.isfinite(r) and self._min_range <= r <= self._max_range:
                lx = float(r) * math.cos(angle)
                ly = float(r) * math.sin(angle)
                wx = tx + cy * lx - sy * ly
                wy = ty + sy * lx + cy * ly
                if self._is_unexpected_point(wx, wy):
                    unexpected_points.append((wx, wy))
            angle += float(msg.angle_increment)

        centroids = self._cluster_centroids(unexpected_points)
        rx, ry = self._current_pose_xy
        centroids = [
            c
            for c in centroids
            if math.hypot(c[0] - rx, c[1] - ry) >= self._min_centroid_robot_dist
        ]

        with self._latest_unexpected_lock:
            self._latest_unexpected_centroids = list(centroids)

        if self._speed_classifier_enable:
            self._update_centroid_tracks(centroids, time.time())

    # ------------------------------------------------------------------
    # Unexpected-obstacle extraction and map-frame centroid tracking
    # ------------------------------------------------------------------

    def _cluster_centroids(self, pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not pts:
            return []

        clusters: List[List[Tuple[float, float]]] = []
        current = [pts[0]]

        for p in pts[1:]:
            if math.hypot(p[0] - current[-1][0], p[1] - current[-1][1]) <= self._cluster_dist:
                current.append(p)
            else:
                if len(current) >= self._cluster_min_points:
                    clusters.append(current)
                current = [p]

        if len(current) >= self._cluster_min_points:
            clusters.append(current)

        result: List[Tuple[float, float]] = []
        for cluster in clusters:
            sx = sum(p[0] for p in cluster)
            sy = sum(p[1] for p in cluster)
            n = float(len(cluster))
            result.append((sx / n, sy / n))
        return result

    def _update_centroid_tracks(self, centroids: List[Tuple[float, float]], now: float) -> None:
        with self._centroid_track_lock:
            matched_tracks: set[int] = set()

            for cx, cy in centroids:
                if not self._is_in_corridor(cx, cy):
                    continue

                best_id: Optional[int] = None
                best_dist = float("inf")

                for tid, track in self._centroid_tracks.items():
                    if tid in matched_tracks:
                        continue
                    d = math.hypot(cx - track.x, cy - track.y)
                    if d < best_dist:
                        best_dist = d
                        best_id = tid

                if best_id is not None and best_dist <= self._centroid_match_dist:
                    track = self._centroid_tracks[best_id]
                    dt = now - track.t
                    if dt >= self._speed_sample_min_dt_s:
                        speed = math.hypot(cx - track.x, cy - track.y) / dt
                        if math.isfinite(speed) and speed >= 0.0:
                            track.speed_samples.append(float(speed))
                            if len(track.speed_samples) > self._speed_sample_window:
                                track.speed_samples = track.speed_samples[-self._speed_sample_window :]
                    track.x = float(cx)
                    track.y = float(cy)
                    track.t = float(now)
                    matched_tracks.add(best_id)
                else:
                    self._centroid_track_id += 1
                    tid = self._centroid_track_id
                    self._centroid_tracks[tid] = TrackState(
                        track_id=tid,
                        x=float(cx),
                        y=float(cy),
                        t=float(now),
                        speed_samples=[],
                    )
                    matched_tracks.add(tid)

            dead = [
                tid
                for tid, track in self._centroid_tracks.items()
                if now - track.t > self._centroid_track_max_age_s
            ]
            for tid in dead:
                self._centroid_tracks.pop(tid, None)

    def _tracked_speed_near(self, x: float, y: float, max_dist: float = 2.0) -> float:
        if not self._speed_classifier_enable:
            return -1.0

        with self._centroid_track_lock:
            best: Optional[TrackState] = None
            best_dist = float("inf")

            for track in self._centroid_tracks.values():
                d = math.hypot(track.x - x, track.y - y)
                if d < best_dist:
                    best_dist = d
                    best = track

            if best is None or best_dist > max_dist or not best.speed_samples:
                return -1.0

            samples = [
                s for s in best.speed_samples if math.isfinite(s) and s >= 0.0
            ]

        return _median(samples)

    def _is_unexpected_point(self, wx: float, wy: float) -> bool:
        grid = self._map_msg
        if grid is None:
            return False

        cell = self._world_to_map(wx, wy)
        if cell is None:
            return False
        mx, my = cell

        idx = my * int(grid.info.width) + mx
        value = int(grid.data[idx])

        if value < 0:
            return not self._exclude_unknown
        if value >= self._occupied_threshold:
            return False
        if self._occupied_margin_cells > 0 and self._near_occupied(
            mx, my, self._occupied_margin_cells
        ):
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
                if int(grid.data[row + xx]) >= self._occupied_threshold:
                    return True
        return False

    def _world_to_map(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        grid = self._map_msg
        if grid is None:
            return None

        res = float(grid.info.resolution)
        if res <= 0.0:
            return None

        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        mx = int((wx - ox) / res)
        my = int((wy - oy) / res)

        if mx < 0 or my < 0 or mx >= int(grid.info.width) or my >= int(grid.info.height):
            return None
        return mx, my

    # ------------------------------------------------------------------
    # Detour gate and event creation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_path_length(path_msg: Path) -> float:
        poses = path_msg.poses
        if len(poses) < 2:
            return 0.0

        total = 0.0
        for i in range(len(poses) - 1):
            p1 = poses[i].pose.position
            p2 = poses[i + 1].pose.position
            total += math.hypot(p2.x - p1.x, p2.y - p1.y)
        return float(total)

    def _is_in_corridor(self, x: float, y: float) -> bool:
        x_min = min(self._corridor_start_x, self._corridor_end_x) - self._corridor_x_margin
        x_max = max(self._corridor_start_x, self._corridor_end_x) + self._corridor_x_margin
        if not (x_min <= x <= x_max):
            return False
        return any(
            abs(y - cy) <= self._corridor_y_half_width
            for cy in self._corridor_y_centers
        )

    def _make_event_id(self) -> str:
        return f"event_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _depth_ratio_for_event(self, event_id: str, obstacle_x: float) -> float:
        if not self._depth_correction_enable:
            return 0.0

        bounds = self._event_depth_bounds.get(event_id)
        if bounds is None:
            return 0.0

        entry_x, exit_x = bounds
        denom = exit_x - entry_x
        if abs(denom) <= 1e-6:
            return 0.0

        rho = (float(obstacle_x) - entry_x) / denom
        return max(0.0, min(1.0, rho))

    def _plan_cb(self, msg: Path) -> None:
        current_length = self._compute_path_length(msg)
        old_length = self._previous_path_length
        self._previous_path_length = current_length

        if not self._gate_on_detour_only:
            return
        if old_length < self._detour_min_previous_length:
            return

        denom = max(old_length, 1e-6)
        path_ratio_factor = current_length / denom
        detour_ratio = (current_length - old_length) / denom

        if path_ratio_factor < self._detour_ratio_threshold:
            return

        now = time.time()
        if now - self._last_detour_trigger < self._detour_cooldown_s:
            return

        self._last_detour_trigger = now
        self._detour_active_until = now + self._detour_hold_s

        robot_x, robot_y = self._current_pose_xy
        event_id = self._make_event_id()
        self._current_event_id = event_id
        self._event_tag_map[event_id] = ""

        corridor_lo = min(
            self._corridor_start_x,
            self._corridor_end_x,
        )
        corridor_hi = max(
            self._corridor_start_x,
            self._corridor_end_x,
        )
        
        entry_x = None
        exit_x = None

        with self._latest_unexpected_lock:
            instant_centroids = list(self._latest_unexpected_centroids)

        obstacle_xy: Optional[Tuple[float, float]] = None
        obstacle_speed_mps = -1.0
        obstacle_dist_m: Optional[float] = None
        obstacle_in_corridor = False

        if instant_centroids:
            closest = min(
                instant_centroids,
                key=lambda c: math.hypot(c[0] - robot_x, c[1] - robot_y),
            )
            obstacle_xy = (float(closest[0]), float(closest[1]))
            obstacle_dist_m = math.hypot(
                obstacle_xy[0] - robot_x,
                obstacle_xy[1] - robot_y,
            )
            obstacle_in_corridor = self._is_in_corridor(*obstacle_xy)

            if obstacle_in_corridor:
                if robot_x <= obstacle_xy[0]:
                    entry_x = corridor_lo
                    exit_x = corridor_hi
                else:
                    entry_x = corridor_hi
                    exit_x = corridor_lo
            
                self._event_depth_bounds[event_id] = (
                    float(entry_x),
                    float(exit_x),
                )
            
                obstacle_speed_mps = self._tracked_speed_near(
                    *obstacle_xy
                )
            
                if obstacle_speed_mps >= 0.0:
                    self._event_speed_mps[event_id] = float(
                        obstacle_speed_mps
                    )
            
                self._insert_temporary_cost(
                    event_id,
                    obstacle_xy,
                    now,
                )
        else:
            self.get_logger().info(
                f"[EVENT] {event_id}: no unexpected centroid available at detour time"
            )

        trigger_text = (
            f"detour_detected old={old_length:.2f} new={current_length:.2f} "
            f"ratio={detour_ratio:.4f} factor={path_ratio_factor:.4f} "
            f"pose=({robot_x:.2f},{robot_y:.2f}) event={event_id} "
            f"active_for={self._detour_hold_s:.1f}s"
        )
        if obstacle_speed_mps >= 0.0:
            trigger_text += f" speed={obstacle_speed_mps:.3f}m/s"
        if obstacle_dist_m is not None:
            trigger_text += f" obstacle_dist={obstacle_dist_m:.2f}m"

        self.get_logger().info(trigger_text)

        self._mission_append_event(
            {
                "event": event_id,
                "timestamp_unix": float(now),
                "timestamp_local": datetime.now().isoformat(),
                "trigger_text": trigger_text,
                "pose_xy": {"x": float(robot_x), "y": float(robot_y)},
                "plan_prev_len": float(old_length),
                "plan_new_len": float(current_length),
                # Relative increase, matching the manuscript's mission-summary field.
                "ratio": float(detour_ratio),
                # Diagnostic factor used by the actual trigger.
                "path_ratio_factor": float(path_ratio_factor),
                "vlm": None,
                "vlm_tag_key": None,
                "vlm_base_tag": None,
                "speed_class": None,
                "obstacle_speed_mps": (
                    float(obstacle_speed_mps) if obstacle_speed_mps >= 0.0 else None
                ),
                "obstacle_dist_m": (
                    float(obstacle_dist_m) if obstacle_dist_m is not None else None
                ),
                "obstacle_xy": (
                    {"x": float(obstacle_xy[0]), "y": float(obstacle_xy[1])}
                    if obstacle_xy is not None
                    else None
                ),
                "obstacle_in_corridor": bool(obstacle_in_corridor),
                "corridor_entry_x": (
                    float(entry_x)
                    if entry_x is not None
                    else None
                ),
                "corridor_exit_x": (
                    float(exit_x)
                    if exit_x is not None
                    else None
                ),
                "tag_group_id": None,
                "tag_repeat_count_in_mission": 0,
            }
        )

        if self._enable_capture or self._vlm_enable:
            self._capture_detour_event(
                event_id=event_id,
                trigger_text=trigger_text,
                old_length=old_length,
                new_length=current_length,
                x=robot_x,
                y=robot_y,
                event_time=now,
            )

    # ------------------------------------------------------------------
    # Residual cost lifecycle
    # ------------------------------------------------------------------

    def _insert_temporary_cost(
        self,
        event_id: str,
        obstacle_xy: Tuple[float, float],
        now: float,
    ) -> None:
        x, y = obstacle_xy
        if not self._is_in_corridor(x, y):
            return

        ttl = max(self._applied_ttl_min_s, float(self._default_cost_ttl_s))
        with self._active_cost_lock:
            cid = self._next_cost_id
            self._next_cost_id += 1
            self._active_costs[cid] = ActiveCost(
                cost_id=cid,
                x=float(x),
                y=float(y),
                ttl_s=float(ttl),
                expires_at=float(now + ttl),
                inserted_at=float(now),
                event_id=event_id,
            )

        self.get_logger().info(
            f"[COST ADD] event={event_id} id={cid} pos=({x:.2f},{y:.2f}) "
            f"temporary_ttl={ttl:.2f}s"
        )
        self._publish_active_costs()

    def _update_event_cost_after_vlm(
        self,
        event_id: str,
        tag_key: str,
        tag_group_id: str,
        ttl_base: float,
        vlm_dt_s: float,
    ) -> Optional[Dict[str, float]]:
        now = time.time()

        with self._active_cost_lock:
            event_costs = [
                cost for cost in self._active_costs.values() if cost.event_id == event_id
            ]
            if not event_costs:
                return None

            cost = event_costs[0]
            if not self._is_in_corridor(cost.x, cost.y):
                return None

            depth_ratio = self._depth_ratio_for_event(event_id, cost.x)
            depth_corrected = float(ttl_base) * (1.0 - depth_ratio)
            applied_ttl = max(
                self._applied_ttl_min_s,
                depth_corrected - float(vlm_dt_s),
            )

            # Reset expiry at the post-VLM update time. Latency is subtracted once.
            cost.tag_key = tag_key
            cost.tag_group_id = tag_group_id
            cost.ttl_s = float(applied_ttl)
            cost.expires_at = float(now + applied_ttl)

        self.get_logger().info(
            f"[TTL APPLY] event={event_id} tag={tag_key} base={ttl_base:.2f}s "
            f"rho={depth_ratio:.3f} depth_ttl={depth_corrected:.2f}s "
            f"vlm_dt={vlm_dt_s:.2f}s applied={applied_ttl:.2f}s"
        )
        self._publish_active_costs()

        return {
            "depth_ratio": float(depth_ratio),
            "depth_corrected_ttl_s": float(depth_corrected),
            "vlm_dt_used_s": float(vlm_dt_s),
            "applied_ttl_s": float(applied_ttl),
        }

    def _active_cost_centroids(self) -> List[Tuple[float, float]]:
        with self._active_cost_lock:
            return [(c.x, c.y) for c in self._active_costs.values()]

    def _publish_pose_array(
        self, centroids: List[Tuple[float, float]], stamp_msg: Any = None
    ) -> None:
        msg = PoseArray()
        if stamp_msg is not None:
            msg.header.stamp = stamp_msg
        msg.header.frame_id = self._map_frame

        for x, y in centroids:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = 0.0
            p.orientation.w = 1.0
            msg.poses.append(p)

        self.object_pub.publish(msg)
        self._last_publish_empty = len(centroids) == 0

    def _publish_active_costs(self) -> None:
        self._publish_pose_array(self._active_cost_centroids(), self._last_scan_stamp)

    def _remove_expired_costs(self, now: float) -> int:
        with self._active_cost_lock:
            expired = [
                cid for cid, cost in self._active_costs.items() if cost.expires_at <= now
            ]
            for cid in expired:
                self._active_costs.pop(cid, None)
        return len(expired)

    def _maintain_cb(self) -> None:
        if not self._enabled:
            return

        expired_count = self._remove_expired_costs(time.time())
        alive = self._active_cost_centroids()

        if expired_count > 0:
            self.get_logger().info(
                f"[EXPIRE] expired={expired_count} survivors={len(alive)} "
                f"global_clear={self._enable_global_clear_on_expire}"
            )

            if self._enable_global_clear_on_expire:
                self._call_clear_service_and_republish()
            else:
                self._publish_pose_array(alive, self._last_scan_stamp)
            return

        if alive:
            self._publish_pose_array(alive, self._last_scan_stamp)
        elif self._publish_empty_when_no_active and not self._last_publish_empty:
            self._publish_pose_array([], self._last_scan_stamp)

    def _call_clear_service_and_republish(self) -> None:
        if self._clear_inflight:
            self._clear_pending_republish = True
            return

        if not self.clear_client.wait_for_service(timeout_sec=self._clear_service_wait_s):
            self.get_logger().warn(f"[CLEAR] unavailable: {self._clear_service_name}")
            self._publish_active_costs()
            return

        self._clear_inflight = True
        future = self.clear_client.call_async(ClearEntireCostmap.Request())
        future.add_done_callback(self._on_clear_done)

    def _on_clear_done(self, future: Any) -> None:
        self._clear_inflight = False
        try:
            future.result()
        except Exception as e:
            self.get_logger().error(f"[CLEAR] failed: {e}")

        self._publish_active_costs()

        if self._clear_pending_republish:
            self._clear_pending_republish = False
            self._call_clear_service_and_republish()

    # ------------------------------------------------------------------
    # Event-time camera capture and VLM semantic tagging
    # ------------------------------------------------------------------

    def _capture_detour_event(
        self,
        event_id: str,
        trigger_text: str,
        old_length: float,
        new_length: float,
        x: float,
        y: float,
        event_time: float,
    ) -> None:
        if not (_CAPTURE_OK and self._cv_bridge and (self._enable_capture or self._vlm_enable)):
            return

        with self._frame_lock:
            if not self._frame_buf:
                self.get_logger().warn(
                    f"[CAPTURE] no camera frame available for event={event_id}"
                )
                return
            frame_ts, frame_img = min(
                self._frame_buf,
                key=lambda item: abs(item[0] - event_time),
            )

        try:
            event_dir = os.path.join(self._save_root, event_id)
            os.makedirs(event_dir, exist_ok=True)
            frame_path = os.path.join(event_dir, "frame_event.jpg")
            if not cv2.imwrite(frame_path, frame_img):
                raise RuntimeError("cv2.imwrite returned false")

            ratio = (new_length - old_length) / max(old_length, 1e-6)
            meta = {
                "event": event_id,
                "event_timestamp_unix": float(event_time),
                "frame_timestamp_unix": float(frame_ts),
                "frame_event_offset_s": float(frame_ts - event_time),
                "timestamp_local": datetime.now().isoformat(),
                "trigger_text": trigger_text,
                "pose_xy": {"x": float(x), "y": float(y)},
                "plan_prev_len": float(old_length),
                "plan_new_len": float(new_length),
                "ratio": float(ratio),
                "camera_topic": self._camera_topic,
                "frames_saved": 1,
            }
            _atomic_json_write(os.path.join(event_dir, "meta.json"), meta)
            self._mission_update_event(
                event_id,
                {
                    "event_dir": event_dir,
                    "frames_saved": 1,
                    "event_frame_offset_s": float(frame_ts - event_time),
                },
            )

            if self._vlm_enable:
                self._run_vlm_async(event_dir, frame_path, trigger_text)
        except Exception as e:
            self.get_logger().error(f"[CAPTURE] event={event_id} failed: {e}")

    def _run_vlm_async(self, event_dir: str, frame_path: str, trigger_text: str) -> None:
        def runner() -> None:
            if self._vlm_singleflight:
                # Serialize requests instead of dropping later events.
                with self._vlm_guard:
                    self._run_vlm_and_save(event_dir, frame_path, trigger_text)
            else:
                self._run_vlm_and_save(event_dir, frame_path, trigger_text)

        threading.Thread(target=runner, daemon=True).start()

    def _run_vlm_and_save(self, event_dir: str, frame_path: str, trigger_text: str) -> None:
        event_id = os.path.basename(event_dir)
        t0 = time.time()

        prompt = (
            f"{self._vlm_prompt}\n\n"
            "Use the event-time camera frame only.\n"
            "Identify the OBJECT TYPE only.\n"
            "Do not append :slow or :fast to tag_key.\n"
            "Return concise visual evidence for the semantic category.\n"
        )

        cmd = [
            self._vlm_python,
            self._vlm_script,
            "--images",
            frame_path,
            "--prompt",
            prompt,
            "--model",
            self._vlm_model[7:]
            if self._vlm_model.startswith("models/")
            else self._vlm_model,
            "--decay_table_path",
            self._vlm_decay_table,
        ]

        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._vlm_timeout_sec,
                env=os.environ.copy(),
            )
            dt = time.time() - t0
            stdout = (r.stdout or "").strip()
            stderr = (r.stderr or "").strip()

            if stdout:
                self.get_logger().info(
                    f"[VLM] stdout(head): {stdout[:self._debug_vlm_stdout_chars]}"
                )
            if stderr:
                self.get_logger().warn(
                    f"[VLM] stderr(head): {stderr[:self._debug_vlm_stderr_chars]}"
                )

            result_json: Optional[Dict[str, Any]] = None
            try:
                obj = json.loads(stdout)
                if isinstance(obj, dict):
                    result_json = obj
            except Exception:
                result_json = None

            _atomic_json_write(
                os.path.join(event_dir, "vlm_result.json"),
                {
                    "event": event_id,
                    "model": self._vlm_model,
                    "prompt": prompt,
                    "trigger_text": trigger_text,
                    "returncode": int(r.returncode),
                    "stdout": stdout,
                    "stderr": stderr,
                    "result_json": result_json,
                    "dt_sec": float(dt),
                },
            )

            if r.returncode != 0 or not isinstance(result_json, dict):
                self._mission_update_event(
                    event_id,
                    {
                        "vlm_returncode": int(r.returncode),
                        "vlm_dt_sec": float(dt),
                        "vlm_error": stderr or "invalid_json_output",
                    },
                )
                return

            base_tag = extract_vlm_tag_key(result_json)
            event_speed = self._event_speed_mps.get(event_id, -1.0)

            speed_class = ""
            if base_tag and event_speed >= 0.0 and self._speed_classifier_enable:
                if not self._gmm_freeze:
                    self._gmm.add_sample(base_tag, event_speed)

                if self._gmm.is_ready(base_tag):
                    speed_class = self._gmm.predict(base_tag, event_speed)
                else:
                    # Readiness is checked before predict; this is the runtime fallback.
                    speed_class = (
                        "fast" if event_speed >= self._speed_threshold_mps else "slow"
                    )

            if base_tag and speed_class in ("slow", "fast"):
                tag_key = f"{base_tag}:{speed_class}"
            else:
                tag_key = ""

            tag_group_id = self._make_tag_group_id(tag_key) if tag_key else ""
            if tag_key:
                self._event_tag_map[event_id] = tag_key

            self._mission_update_event(
                event_id,
                {
                    "vlm": result_json,
                    "vlm_tag_key": tag_key or None,
                    "vlm_base_tag": base_tag or None,
                    "speed_class": speed_class or None,
                    "obstacle_speed_mps": (
                        float(event_speed) if event_speed >= 0.0 else None
                    ),
                    "tag_group_id": tag_group_id or None,
                    "vlm_returncode": int(r.returncode),
                    "vlm_dt_sec": float(dt),
                },
            )

            repeat_count = self._count_tag_group_in_mission(tag_group_id)
            if tag_group_id:
                self._mission_update_event(
                    event_id,
                    {"tag_repeat_count_in_mission": int(repeat_count)},
                )

            if tag_key:
                ttl_base = self._lookup_decay_ttl(tag_key)
                ttl_fields = self._update_event_cost_after_vlm(
                    event_id=event_id,
                    tag_key=tag_key,
                    tag_group_id=tag_group_id,
                    ttl_base=ttl_base,
                    vlm_dt_s=dt,
                )

                patch: Dict[str, Any] = {"ttl_base_s": float(ttl_base)}
                if ttl_fields:
                    patch.update(ttl_fields)
                self._mission_update_event(event_id, patch)
            else:
                self.get_logger().warn(
                    f"[TAG] event={event_id} incomplete compound tag: "
                    f"base={base_tag!r}, speed={event_speed:.3f}"
                )

            out = String()
            out.data = json.dumps(
                {
                    "event": event_id,
                    "vlm": result_json,
                    "base_tag": base_tag or None,
                    "speed_class": speed_class or None,
                    "tag_key": tag_key or None,
                    "dt_sec": float(dt),
                },
                ensure_ascii=False,
            )
            self.vlm_pub.publish(out)

        except Exception as e:
            dt = time.time() - t0
            self.get_logger().error(f"[VLM] event={event_id} failed after {dt:.2f}s: {e}")
            try:
                _atomic_json_write(
                    os.path.join(event_dir, "vlm_result.json"),
                    {
                        "event": event_id,
                        "model": self._vlm_model,
                        "prompt": prompt,
                        "trigger_text": trigger_text,
                        "error": str(e),
                        "dt_sec": float(dt),
                    },
                )
            except Exception:
                pass
            self._mission_update_event(
                event_id,
                {"vlm_error": str(e), "vlm_dt_sec": float(dt)},
            )

    # ------------------------------------------------------------------
    # LLM proposal parsing and local application
    # ------------------------------------------------------------------

    def _parse_llm_proposal_updates(
        self, stdout: str
    ) -> List[Tuple[str, float, float]]:
        if not stdout:
            return []

        self._llm_reason_map = {}
        parsed: List[Tuple[str, float, float]] = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("[PROPOSAL_JSON]"):
                continue

            try:
                obj = json.loads(line[len("[PROPOSAL_JSON]") :].strip())
            except Exception:
                continue

            for update in obj.get("updates", []) or []:
                tag = normalize_tag_key(update.get("tag_key", ""))
                ttl_raw = update.get("ttl_base_s", update.get("ttl_s"))
                if not tag or ttl_raw is None:
                    continue

                try:
                    ttl = self._clamp_base_ttl(float(ttl_raw))
                    conf = min(1.0, max(0.0, float(update.get("confidence", 0.5))))
                except Exception:
                    continue

                reason = str(update.get("reason", "") or "")
                if reason:
                    self._llm_reason_map[tag] = reason
                parsed.append((tag, ttl, conf))

        if parsed:
            return parsed

        # Backward-compatible textual fallback for the current helper script.
        pattern = re.compile(
            r"^\s*-\s+([A-Za-z0-9_\-:]+):\s+"
            r"(?:ttl_base_s|ttl_s)=([0-9]+(?:\.[0-9]+)?)",
            re.MULTILINE,
        )
        for m in pattern.finditer(stdout):
            tag = normalize_tag_key(m.group(1))
            if not tag:
                continue
            parsed.append((tag, self._clamp_base_ttl(float(m.group(2))), 0.5))
        return parsed

    def _apply_llm_proposal_updates_local(
        self, proposal_updates: List[Tuple[str, float]]
    ) -> int:
        raw = _read_json(self._vlm_decay_table, {})
        if not isinstance(raw, dict):
            raw = {}

        updated = 0
        for tag, new_ttl in proposal_updates:
            key = normalize_tag_key(tag)
            if not key:
                continue
            ttl = self._clamp_base_ttl(new_ttl)
            cur = raw.get(key)
            if isinstance(cur, dict):
                cur = dict(cur)
                cur["ttl"] = ttl
                raw[key] = cur
            else:
                raw[key] = {"ttl": ttl}
            updated += 1

        if updated:
            _atomic_json_write(self._vlm_decay_table, raw)
            self._load_decay_table()
        return updated

    def _prompt_optional_feedback(self) -> str:
        try:
            return input("Optional feedback (press Enter to skip): ").strip()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Archive handling
    # ------------------------------------------------------------------

    def _normalize_feedback_text(self, text: Optional[str]) -> str:
        s = (text or "").strip().lower()
        if not s:
            return ""
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-z0-9_ ./:\-]", "", s)
        return s[:300]

    def _append_archive_records(
        self,
        proposal_updates: List[Tuple[str, float, float]],
        approval_mode: str,
        approval_status: str,
        human_feedback: str = "",
    ) -> None:
        """Append/update proposal cases using the Appendix-D-style archive list."""
        if not (self._llm_decay_rag_enable and self._llm_decay_append_to_archive):
            return
        if not proposal_updates:
            return

        proposal_map = {
            normalize_tag_key(tag): (float(ttl), float(conf))
            for tag, ttl, conf in proposal_updates
        }

        with self._mission_lock:
            mission = json.loads(json.dumps(self._mission))

        mission_id = str(mission.get("mission_id") or "unknown_mission")
        feedback = (human_feedback or "").strip()
        feedback_norm = self._normalize_feedback_text(feedback)

        archive = _read_json(self._llm_decay_retrieval_archive_path, [])
        if isinstance(archive, dict):
            # Backward-compatible migration from older {"cases": [...]} archives.
            rows = archive.get("cases", [])
        else:
            rows = archive
        if not isinstance(rows, list):
            rows = []

        index: Dict[Tuple[str, str, str], int] = {}
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            index[(
                str(row.get("mission_id") or ""),
                str(row.get("event_id") or ""),
                normalize_tag_key(row.get("tag_key")),
            )] = i

        for ev in mission.get("events", []) or []:
            if not isinstance(ev, dict):
                continue

            tag = normalize_tag_key(ev.get("vlm_tag_key"))
            if tag not in proposal_map:
                continue

            event_id = str(ev.get("event") or "")
            if not event_id:
                continue

            ttl, llm_conf = proposal_map[tag]
            vlm = ev.get("vlm") if isinstance(ev.get("vlm"), dict) else {}
            try:
                vlm_conf = (
                    float(vlm.get("confidence"))
                    if vlm.get("confidence") is not None
                    else None
                )
            except Exception:
                vlm_conf = None

            try:
                repeat_count = int(ev.get("tag_repeat_count_in_mission") or 1)
            except Exception:
                repeat_count = 1
            repeat_count = max(1, repeat_count)

            row = {
                "mission_id": mission_id,
                "event_id": event_id,
                "tag_key": tag,
                "detour_ratio": ev.get("ratio"),
                "old_len": ev.get("plan_prev_len"),
                "new_len": ev.get("plan_new_len"),
                "vlm_confidence": vlm_conf,
                "evidence": vlm.get("evidence"),
                "applied_ttl_s": ev.get("applied_ttl_s"),
                "timestamp": ev.get("timestamp_unix"),
                "tag_group_id": ev.get("tag_group_id"),
                "repeat_count_in_mission": repeat_count,
                "same_obstacle_reencountered": bool(repeat_count >= 2),
                "depth_ratio": ev.get("depth_ratio"),
                "depth_corrected_ttl_s": ev.get("depth_corrected_ttl_s"),
                "vlm_dt_used_s": ev.get("vlm_dt_used_s"),
                "ttl_base_s": ev.get("ttl_base_s"),
                "approval_mode": approval_mode,
                "approval_status": approval_status,
                "proposed_ttl_s": float(ttl),
                "llm_proposal_confidence": float(llm_conf),
                "proposal_reason": self._llm_reason_map.get(tag, ""),
                "approval_timestamp": time.time(),
                "human_feedback": feedback,
                "human_feedback_norm": feedback_norm,
                "human_feedback_present": bool(feedback),
            }

            key = (mission_id, event_id, tag)
            if key in index:
                rows[index[key]] = row
            else:
                index[key] = len(rows)
                rows.append(row)

        max_cases = max(1, self._llm_decay_archive_max_cases)
        if len(rows) > max_cases:
            rows = rows[-max_cases:]

        _atomic_json_write(self._llm_decay_retrieval_archive_path, rows)

    def _record_llm_decision(
        self,
        updates: List[Tuple[str, float, float]],
        status: str,
        mode: str,
    ) -> None:
        with self._mission_lock:
            block = self._mission.setdefault("llm_policy_update", {})
            decisions = block.setdefault("decisions", [])
            decisions.append(
                {
                    "timestamp": time.time(),
                    "mode": mode,
                    "status": status,
                    "updates": [
                        {
                            "tag_key": tag,
                            "ttl_base_s": ttl,
                            "confidence": conf,
                            "reason": self._llm_reason_map.get(tag, ""),
                        }
                        for tag, ttl, conf in updates
                    ],
                }
            )
        self._save_mission()

    # ------------------------------------------------------------------
    # Post-mission LLM update and selective review
    # ------------------------------------------------------------------

    def _run_llm_preview(self) -> Tuple[subprocess.CompletedProcess, float]:
        cmd = [
            self._llm_decay_python,
            self._llm_decay_script,
            "--input",
            self._mission_summary_path,
            "--output",
            self._mission_summary_out_path,
            "--decay_table_path",
            self._vlm_decay_table,
            "--init_table_if_missing",
            "--update_mission_summary",
            self._mission_summary_path,
            "--model",
            self._llm_decay_model[7:]
            if self._llm_decay_model.startswith("models/")
            else self._llm_decay_model,
            "--yes",
            "--dry_run",
        ]

        if self._llm_decay_rag_enable:
            cmd += [
                "--retrieval_archive_path",
                self._llm_decay_retrieval_archive_path,
                "--retrieval_max_repeat1_cases",
                str(max(1, self._llm_decay_retrieval_max_repeat1_cases)),
                "--archive_max_cases",
                str(max(1, self._llm_decay_archive_max_cases)),
            ]

        t0 = time.time()
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300.0,
            env=os.environ.copy(),
        )
        return r, time.time() - t0

    def _publish_llm_result(
        self,
        returncode: int,
        dt_s: float,
        mode: str,
        stdout: str,
        stderr: str,
    ) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "returncode": int(returncode),
                "dt_sec": float(dt_s),
                "mode": mode,
                "stdout": stdout[: self._llm_stdout_log_chars],
                "stderr": stderr[: self._llm_stderr_log_chars],
                "mission_summary": self._mission_summary_path,
                "decay_table": self._vlm_decay_table,
                "rag": {
                    "enabled": self._llm_decay_rag_enable,
                    "archive_path": (
                        self._llm_decay_retrieval_archive_path
                        if self._llm_decay_rag_enable
                        else ""
                    ),
                },
            },
            ensure_ascii=False,
        )
        self.llm_decay_pub.publish(msg)

    def _run_postrun_llm_async(self) -> bool:
        if not self._llm_decay_enable:
            self.get_logger().info("[LLM] disabled")
            return False

        if self._llm_decay_singleflight:
            with self._llm_decay_guard:
                if self._llm_decay_running:
                    self.get_logger().warn("[LLM] already running")
                    return False
                self._llm_decay_running = True

        def runner() -> None:
            try:
                self._run_postrun_llm_once()
            finally:
                if self._llm_decay_singleflight:
                    with self._llm_decay_guard:
                        self._llm_decay_running = False

        threading.Thread(target=runner, daemon=True).start()
        return True

    def _run_postrun_llm_once(self) -> None:
        with self._mission_lock:
            event_count = len(self._mission.get("events", []) or [])
        if event_count == 0:
            self.get_logger().info("[LLM] no mission events; nothing to update")
            return

        self._save_mission()

        try:
            r, dt = self._run_llm_preview()
        except Exception as e:
            self.get_logger().error(f"[LLM] preview failed: {e}")
            return

        stdout = r.stdout or ""
        stderr = r.stderr or ""
        self._publish_llm_result(
            r.returncode,
            dt,
            self._llm_decay_approval_mode,
            stdout,
            stderr,
        )

        if stdout:
            filtered = "\n".join(
                line
                for line in stdout.splitlines()
                if not line.strip().startswith("[PROPOSAL_JSON]")
            )
            self.get_logger().info(
                f"[LLM] stdout(head): {filtered[:self._llm_stdout_log_chars]}"
            )
        if stderr:
            self.get_logger().warn(
                f"[LLM] stderr(head): {stderr[:self._llm_stderr_log_chars]}"
            )

        if r.returncode != 0:
            self.get_logger().error(f"[LLM] helper failed rc={r.returncode}")
            return

        proposals = self._parse_llm_proposal_updates(stdout)
        if not proposals:
            self.get_logger().info("[LLM] no applicable proposal")
            if self._start_new_mission_after_llm:
                self._start_new_mission(save=True)
            return

        mode = self._llm_decay_approval_mode
        if mode == "auto":
            auto_updates = proposals
            human_updates: List[Tuple[str, float, float]] = []
        elif mode == "human_all":
            auto_updates = []
            human_updates = proposals
        else:  # ours
            auto_updates = [
                u for u in proposals if u[2] >= self._llm_decay_confidence_threshold
            ]
            human_updates = [
                u for u in proposals if u[2] < self._llm_decay_confidence_threshold
            ]

        if auto_updates:
            self._apply_llm_proposal_updates_local(
                [(tag, ttl) for tag, ttl, _ in auto_updates]
            )
            self._append_archive_records(
                auto_updates,
                approval_mode=mode,
                approval_status="auto",
            )
            self._record_llm_decision(auto_updates, status="auto", mode=mode)

        review_status = ""
        human_feedback = ""

        if human_updates:
            lines = []
            for tag, ttl, conf in human_updates:
                old = self._lookup_decay_ttl(tag)
                reason = self._llm_reason_map.get(tag, "")
                line = (
                    f"  - {tag}: old={old:.2f}s -> proposed={ttl:.2f}s "
                    f"confidence={conf:.2f}"
                )
                if reason:
                    line += f"\n    reason: {reason}"
                lines.append(line)

            self.get_logger().info(
                "[LLM] human review required:\n" + "\n".join(lines)
            )

            approved = False
            try:
                while True:
                    ans = input("[LLM] Apply pending updates? [y/N] ").strip().lower()
                    if ans in ("y", "yes"):
                        approved = True
                        review_status = "approved"
                        break
                    if ans in ("", "n", "no"):
                        approved = False
                        review_status = "rejected"
                        break
            except EOFError:
                # Manuscript behavior: pending proposal remains unapplied.
                approved = False
                review_status = "unapplied_eof"
                self.get_logger().info(
                    "[LLM] EOF: pending proposal left unapplied; TTL unchanged"
                )

            if review_status != "unapplied_eof":
                human_feedback = self._prompt_optional_feedback()

            if approved:
                self._apply_llm_proposal_updates_local(
                    [(tag, ttl) for tag, ttl, _ in human_updates]
                )

            self._append_archive_records(
                human_updates,
                approval_mode=mode,
                approval_status=review_status,
                human_feedback=human_feedback,
            )
            self._record_llm_decision(
                human_updates,
                status=review_status,
                mode=mode,
            )

        with self._mission_lock:
            self._mission["llm_postrun_dt_s"] = float(dt)
            self._mission["llm_postrun_mode"] = mode
            self._mission.setdefault("llm_policy_update", {})["timestamp"] = time.time()
        self._save_mission()

        try:
            with self._mission_lock:
                snapshot = json.loads(json.dumps(self._mission))
            _atomic_json_write(self._mission_summary_out_path, snapshot)
        except Exception as e:
            self.get_logger().warn(f"[LLM] failed to write mission summary output: {e}")

        if self._start_new_mission_after_llm:
            self._start_new_mission(save=True)

    # ------------------------------------------------------------------
    # Mission lifecycle services / optional goal-success trigger
    # ------------------------------------------------------------------

    def _reset_mission_cb(self, request: Trigger.Request, response: Trigger.Response):
        del request
        self._start_new_mission(save=True)
        response.success = True
        response.message = f"New mission started: {self._current_mission_id()}"
        return response

    def _run_postrun_llm_cb(self, request: Trigger.Request, response: Trigger.Response):
        del request
        started = self._run_postrun_llm_async()
        response.success = bool(started)
        response.message = "LLM postrun started" if started else "LLM postrun not started"
        return response

    def _goal_status_cb(self, msg: GoalStatusArray) -> None:
        if not self._postrun_on_goal_success:
            return

        now = time.time()
        newly_succeeded = []
        for st in msg.status_list:
            if int(st.status) != int(GoalStatus.STATUS_SUCCEEDED):
                continue
            gid = "".join(f"{b:02x}" for b in st.goal_info.goal_id.uuid)
            if gid not in self._seen_goal_success_ids:
                self._seen_goal_success_ids.add(gid)
                newly_succeeded.append(gid)

        if not newly_succeeded:
            return
        if now - self._last_goal_success_time < self._goal_success_cooldown_s:
            return

        self._last_goal_success_time = now
        self._run_postrun_llm_async()


def main() -> None:
    rclpy.init()
    node = UnexpectedObstacleDetector()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
