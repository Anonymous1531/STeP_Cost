#!/usr/bin/env python3
import math
import json
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String

class KalmanTrack:
    _id_counter = 0

    def __init__(self, x: float, y: float, dt: float = 0.2):
        KalmanTrack._id_counter += 1
        self.track_id = KalmanTrack._id_counter
        self.dt = dt
        self.miss_count = 0
        self.hit_count = 1

        self.x = np.array([x, y, 0.0, 0.0], dtype=float)

        self.P = np.eye(4) * 1.0

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        self.Q = np.eye(4) * 0.1   
        self.R = np.eye(2) * 0.5 

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.miss_count = 0
        self.hit_count += 1

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return math.hypot(self.x[2], self.x[3])

class OnlineKNN:
    def __init__(self, k: int = 5, threshold_mps: float = 0.3, min_samples: int = 10):
        self.k = k
        self.threshold_mps = threshold_mps
        self.min_samples = min_samples
        self._X: List[float] = []
        self._y: List[int]   = []

    @property
    def n_samples(self) -> int:
        return len(self._X)

    @property
    def is_ready(self) -> bool:
        return self.n_samples >= self.min_samples

    def add_sample(self, speed_mps: float):
        label = 1 if speed_mps >= self.threshold_mps else 0
        self._X.append(speed_mps)
        self._y.append(label)

    def predict(self, speed_mps: float) -> str:
        if not self.is_ready:
            return "fast"
        dists = [abs(speed_mps - x) for x in self._X]
        k_idx = sorted(range(len(dists)), key=lambda i: dists[i])[:self.k]
        votes = sum(self._y[i] for i in k_idx)
        return "fast" if votes >= (self.k / 2) else "slow"

class ObstacleSpeedClassifier(Node):
    def __init__(self):
        super().__init__("obstacle_speed_classifier")
        self.declare_parameter("scan_topic",          "/scan")
        self.declare_parameter("odom_topic",          "/odom")
        self.declare_parameter("output_topic",        "/obstacle_speed_class")
        self.declare_parameter("cluster_dist_thresh", 0.3)
        self.declare_parameter("cluster_min_points",  3)
        self.declare_parameter("max_range_m",         6.0)
        self.declare_parameter("min_range_m",         0.15)
        self.declare_parameter("publish_hz",          5.0)
        self.declare_parameter("knn_k",               5)
        self.declare_parameter("knn_min_samples",     10)
        self.declare_parameter("speed_threshold_mps", 0.3)
        self.declare_parameter("kalman_dt",           0.2)
        self.declare_parameter("track_max_miss",      5)

        scan_topic    = self.get_parameter("scan_topic").value
        odom_topic    = self.get_parameter("odom_topic").value
        output_topic  = self.get_parameter("output_topic").value
        self._cluster_dist  = float(self.get_parameter("cluster_dist_thresh").value)
        self._cluster_min   = int(self.get_parameter("cluster_min_points").value)
        self._max_range     = float(self.get_parameter("max_range_m").value)
        self._min_range     = float(self.get_parameter("min_range_m").value)
        self._kalman_dt     = float(self.get_parameter("kalman_dt").value)
        self._track_max_miss = int(self.get_parameter("track_max_miss").value)
        thresh = float(self.get_parameter("speed_threshold_mps").value)
        k      = int(self.get_parameter("knn_k").value)
        min_s  = int(self.get_parameter("knn_min_samples").value)

        self._knn = OnlineKNN(k=k, threshold_mps=thresh, min_samples=min_s)

        self._tracks: Dict[int, KalmanTrack] = {}
        self._track_lock = threading.Lock()

        self._robot_vx: float = 0.0
        self._robot_vy: float = 0.0
        self._odom_lock = threading.Lock()

        self._speed_history: List[Tuple[float, float]] = []
        self._history_window_s: float = 3.0 
        self._history_lock = threading.Lock()
        self._latest: Optional[dict] = None

        self.create_subscription(LaserScan, scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._pub = self.create_publisher(String, output_topic, 10)

        hz = float(self.get_parameter("publish_hz").value)
        self.create_timer(1.0 / hz, self._publish_cb)

        self.create_timer(self._kalman_dt, self._kalman_predict_cb)

        self.get_logger().info(
            f"ObstacleSpeedClassifier started "
            f"scan={scan_topic} odom={odom_topic} out={output_topic} "
            f"knn_k={k} min_samples={min_s} threshold={thresh}m/s"
        )

    def _odom_cb(self, msg: Odometry):
        with self._odom_lock:
            self._robot_vx = msg.twist.twist.linear.x
            self._robot_vy = msg.twist.twist.linear.y

    def _scan_cb(self, msg: LaserScan):
        points = self._scan_to_points(msg)
        if not points:
            return

        clusters = self._cluster_points(points)
        if not clusters:
            return

        centroids = [self._centroid(c) for c in clusters]

        with self._odom_lock:
            rvx, rvy = self._robot_vx, self._robot_vy

        with self._track_lock:
            self._associate_and_update(centroids)

            best_track = self._closest_track()
            if best_track is None:
                return

            abs_vx = best_track.vel[0] - rvx
            abs_vy = best_track.vel[1] - rvy
            speed_mps = math.hypot(abs_vx, abs_vy)

            self._knn.add_sample(speed_mps)
            speed_class = self._knn.predict(speed_mps)

            now = time.time()
            with self._history_lock:
                self._speed_history.append((now, speed_mps))
                cutoff = now - self._history_window_s
                self._speed_history = [
                    (t, s) for t, s in self._speed_history if t >= cutoff
                ]

            self._latest = {
                "speed_mps":   round(speed_mps, 3),
                "speed_class": speed_class,
                "track_id":    best_track.track_id,
                "knn_ready":   self._knn.is_ready,
                "n_samples":   self._knn.n_samples,
            }

    def _kalman_predict_cb(self):
        with self._track_lock:
            dead = []
            for tid, track in self._tracks.items():
                track.predict()
                track.miss_count += 1
                if track.miss_count > self._track_max_miss:
                    dead.append(tid)
            for tid in dead:
                del self._tracks[tid]

    def _publish_cb(self):
        if self._latest is None:
            return
        msg = String()
        msg.data = json.dumps(self._latest, ensure_ascii=False)
        self._pub.publish(msg)

    def _scan_to_points(self, msg: LaserScan) -> List[Tuple[float, float]]:
        pts = []
        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and self._min_range <= r <= self._max_range:
                pts.append((r * math.cos(angle), r * math.sin(angle)))
            angle += msg.angle_increment
        return pts

    def _cluster_points(self, points: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        if not points:
            return []
        clusters: List[List[Tuple[float, float]]] = []
        current = [points[0]]
        for p in points[1:]:
            if math.hypot(p[0] - current[-1][0], p[1] - current[-1][1]) <= self._cluster_dist:
                current.append(p)
            else:
                if len(current) >= self._cluster_min:
                    clusters.append(current)
                current = [p]
        if len(current) >= self._cluster_min:
            clusters.append(current)
        return clusters

    def _centroid(self, cluster: List[Tuple[float, float]]) -> Tuple[float, float]:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _associate_and_update(self, centroids: List[Tuple[float, float]]):
        matched_tracks = set()
        for cx, cy in centroids:
            best_tid, best_dist = None, float("inf")
            for tid, track in self._tracks.items():
                if tid in matched_tracks:
                    continue
                tx, ty = track.pos
                d = math.hypot(cx - tx, cy - ty)
                if d < best_dist:
                    best_dist = d
                    best_tid = tid

            assoc_thresh = self._cluster_dist * 3
            if best_tid is not None and best_dist < assoc_thresh:
                self._tracks[best_tid].update(np.array([cx, cy]))
                matched_tracks.add(best_tid)
            else:
                t = KalmanTrack(cx, cy, dt=self._kalman_dt)
                self._tracks[t.track_id] = t

    def _query_cb(self, msg: String):
        try:
            q = json.loads(msg.data)
            detour_time   = float(q.get("detour_time", time.time()))
            window_before = float(q.get("window_before_s", 2.0))
            window_after  = float(q.get("window_after_s", 0.5))
        except Exception as e:
            self.get_logger().warn(f"[QUERY] parse error: {e}")
            return

        result = self.get_speed_at_detour(detour_time, window_before, window_after)
        self._latest = {
            "speed_mps":           result["speed_mps"],
            "speed_class":         result["speed_class"],
            "track_id":            -1,
            "knn_ready":           result["knn_ready"],
            "n_samples":           self._knn.n_samples,
            "n_samples_in_window": result["n_samples_in_window"],
            "window_s":            result["window_s"],
            "detour_query":        True,
        }
        pub_msg = String()
        pub_msg.data = json.dumps(self._latest, ensure_ascii=False)
        self._pub.publish(pub_msg)
        speed = result["speed_mps"]
        cls = result["speed_class"]
        n = result["n_samples_in_window"]
        self.get_logger().info(
            f"[QUERY] detour_time={detour_time:.2f} speed={speed:.3f}m/s class={cls} n_window={n}"
        )

    def _closest_track(self) -> Optional[KalmanTrack]:
        if not self._tracks:
            return None
        return min(self._tracks.values(), key=lambda t: math.hypot(*t.pos))

    def get_speed_at_detour(
        self,
        detour_time: Optional[float] = None,
        window_before_s: float = 2.0,
        window_after_s: float = 0.5,
    ) -> dict:
        t_ref = detour_time if detour_time is not None else time.time()
        t_start = t_ref - window_before_s
        t_end   = t_ref + window_after_s

        with self._history_lock:
            window_samples = [
                s for t, s in self._speed_history
                if t_start <= t <= t_end
            ]

        if not window_samples:
            if self._latest:
                return {
                    "speed_mps": self._latest["speed_mps"],
                    "speed_class": self._latest["speed_class"],
                    "n_samples_in_window": 0,
                    "knn_ready": self._knn.is_ready,
                    "window_s": window_before_s + window_after_s,
                }
            return {
                "speed_mps": 0.0,
                "speed_class": "fast", 
                "n_samples_in_window": 0,
                "knn_ready": self._knn.is_ready,
                "window_s": window_before_s + window_after_s,
            }

        avg_speed = sum(window_samples) / len(window_samples)
        speed_class = self._knn.predict(avg_speed)

        return {
            "speed_mps": round(avg_speed, 3),
            "speed_class": speed_class,
            "n_samples_in_window": len(window_samples),
            "knn_ready": self._knn.is_ready,
            "window_s": window_before_s + window_after_s,
        }

def main():
    rclpy.init()
    node = ObstacleSpeedClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
