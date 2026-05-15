#!/usr/bin/env python3
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from action_msgs.msg import GoalStatusArray, GoalStatus

from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from gazebo_msgs.msg import EntityState
from tf_transformations import quaternion_from_euler


class MovingObstacleController(Node):
    def __init__(self):
        super().__init__('factory6_moving_obstacle_controller')

        self.declare_parameter('entity_name', 'exp_box')
        self.declare_parameter('sdf_path', '')
        self.declare_parameter('start_delay_s', 3.0)
        self.declare_parameter('loop_delay_s', 0.0) 
        self.declare_parameter('speed_mps', 0.05)
        self.declare_parameter('update_hz', 10.0)
        self.declare_parameter('despawn_on_finish', False)
        self.declare_parameter('loop_forever', False)
        self.declare_parameter('goal_status_topic', '/navigate_to_pose/_action/status')
        self.declare_parameter('speed_approach_mps', 0.0)  
        self.declare_parameter('mission_summary_out_path', '')
        self.declare_parameter('spawn_mode', 'standby')
        self.declare_parameter('spawn_ratio_min', 0.0)
        self.declare_parameter('spawn_ratio_max', 0.3)
        self.declare_parameter('standby_x', 16.14)
        self.declare_parameter('standby_y', -11.50)
        self.declare_parameter('standby_z', 0.0)
        self.declare_parameter('enter_x', 16.14)
        self.declare_parameter('enter_y', -10.00)
        self.declare_parameter('enter_z', 0.0)
        self.declare_parameter('exit_x', 16.14)
        self.declare_parameter('exit_y', -7.90)
        self.declare_parameter('exit_z', 0.0)
        self.declare_parameter('exit2_x', 16.14)
        self.declare_parameter('exit2_y', -7.90)
        self.declare_parameter('exit2_z', 0.0)
        self.declare_parameter('model_yaw_offset_deg', 0.0)
        self.declare_parameter('pair_mode', False)
        self.declare_parameter('pair_entity_name', '')
        self.declare_parameter('pair_sdf_path', '')
        self.declare_parameter('pair_offset_m', 1.2)


        self.model_yaw_offset = math.radians(
            float(self.get_parameter('model_yaw_offset_deg').value))
        self.entity_name   = str(self.get_parameter('entity_name').value)
        self.sdf_path      = str(self.get_parameter('sdf_path').value)
        self.start_delay_s = float(self.get_parameter('start_delay_s').value)
        self.loop_delay_s  = float(self.get_parameter('loop_delay_s').value)
        self.mission_summary_out_path = str(self.get_parameter('mission_summary_out_path').value).strip()
        self.spawn_mode      = str(self.get_parameter('spawn_mode').value).strip()
        self.spawn_ratio_min = float(self.get_parameter('spawn_ratio_min').value)
        self.spawn_ratio_max = float(self.get_parameter('spawn_ratio_max').value)
        self.speed_mps     = max(1e-3, float(self.get_parameter('speed_mps').value))
        self.update_hz     = max(1.0,  float(self.get_parameter('update_hz').value))
        self.despawn_on_finish = bool(self.get_parameter('despawn_on_finish').value)
        self.loop_forever      = bool(self.get_parameter('loop_forever').value)
        _approach = float(self.get_parameter('speed_approach_mps').value)
        self.speed_approach_mps = _approach if _approach > 1e-4 else self.speed_mps * 2.0
        self._goal_status_topic = str(self.get_parameter('goal_status_topic').value)
        self._robot_navigating = False  
        self.create_subscription(
            GoalStatusArray,
            self._goal_status_topic,
            self._goal_status_cb,
            10
        )

        self.standby = (float(self.get_parameter('standby_x').value),
                        float(self.get_parameter('standby_y').value),
                        float(self.get_parameter('standby_z').value))
        self.enter   = (float(self.get_parameter('enter_x').value),
                        float(self.get_parameter('enter_y').value),
                        float(self.get_parameter('enter_z').value))
        self.exit    = (float(self.get_parameter('exit_x').value),
                        float(self.get_parameter('exit_y').value),
                        float(self.get_parameter('exit_z').value))
        self.exit2   = (float(self.get_parameter('exit2_x').value),
                        float(self.get_parameter('exit2_y').value),
                        float(self.get_parameter('exit2_z').value))

        self.pair_mode    = bool(self.get_parameter('pair_mode').value)
        self.pair_offset  = float(self.get_parameter('pair_offset_m').value)

        pair_name = str(self.get_parameter('pair_entity_name').value)
        self.pair_entity_name = pair_name if pair_name else self.entity_name + '_2'

        pair_sdf = str(self.get_parameter('pair_sdf_path').value)
        self.pair_sdf_path = pair_sdf if pair_sdf else self.sdf_path

        dx = self.enter[0] - self.standby[0]
        dy = self.enter[1] - self.standby[1]
        dist = math.hypot(dx, dy) or 1.0
        self._perp = (-dy / dist, dx / dist)

        self.spawn_cli     = self.create_client(SpawnEntity,    '/spawn_entity')
        self.delete_cli    = self.create_client(DeleteEntity,   '/delete_entity')
        self.set_state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        self._wait_for_service(self.spawn_cli,     '/spawn_entity')
        self._wait_for_service(self.delete_cli,    '/delete_entity')
        self._wait_for_service(self.set_state_cli, '/gazebo/set_entity_state')

        self.started = False


    def _wait_for_service(self, cli, name: str):
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Waiting for {name}...')

    @staticmethod
    def quat_from_yaw(yaw: float):
        return quaternion_from_euler(0.0, 0.0, yaw)

    def _offset_point(self, pt: tuple, lateral: float) -> tuple:
        px, py, pz = pt
        return (px + self._perp[0] * lateral,
                py + self._perp[1] * lateral,
                pz)

    def spawn_at(self, entity_name: str, sdf_path: str,
                 x: float, y: float, z: float, yaw: float = 0.0) -> bool:
        sdf_text = Path(sdf_path).expanduser().read_text(encoding='utf-8')
        req = SpawnEntity.Request()
        req.name = entity_name
        req.xml  = sdf_text
        req.robot_namespace  = ''
        req.reference_frame  = 'world'
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        q = self.quat_from_yaw(yaw)
        req.initial_pose.orientation.x = q[0]
        req.initial_pose.orientation.y = q[1]
        req.initial_pose.orientation.z = q[2]
        req.initial_pose.orientation.w = q[3]

        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if not future.done() or future.result() is None:
            self.get_logger().error(f'[SPAWN] No response: {entity_name}')
            return False
        resp = future.result()
        try:
            ok = bool(resp.success)
        except Exception:
            ok = True
        if not ok:
            self.get_logger().error(
                f'[SPAWN] Failed {entity_name}: {resp.status_message}')
            return False
        self.get_logger().info(
            f'[SPAWN] OK {entity_name} at ({x:.2f}, {y:.2f}, {z:.2f})')
        return True

    def delete_entity(self, name: str):
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def set_pose(self, entity_name: str,
                 x: float, y: float, z: float, yaw: float = 0.0):
        state = EntityState()
        state.name = entity_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        q = self.quat_from_yaw(yaw)
        state.pose.orientation.x = q[0]
        state.pose.orientation.y = q[1]
        state.pose.orientation.z = q[2]
        state.pose.orientation.w = q[3]
        state.reference_frame = 'world'
        req = SetEntityState.Request()
        req.state = state
        future = self.set_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)


    def _goal_status_cb(self, msg: GoalStatusArray):
        navigating = any(
            st.status in (
                GoalStatus.STATUS_ACCEPTED,
                GoalStatus.STATUS_EXECUTING,
            )
            for st in msg.status_list
        )
        self._robot_navigating = navigating

    def _wait_for_robot_to_finish(self, poll_hz: float = 2.0):
        if not self._robot_navigating:
            return
        self.get_logger().info('[LOOP] robot is navigating, waiting before teleport...')
        dt = 1.0 / poll_hz
        while self._robot_navigating and self.loop_forever:
            rclpy.spin_once(self, timeout_sec=dt)
        self.get_logger().info('[LOOP] robot finished, proceeding to teleport')

    def _wait_for_llm_completion(self, goal_reached_time: float,
                                  timeout_s: float = 120.0,
                                  poll_interval_s: float = 1.0) -> bool:
        if not self.mission_summary_out_path:
            return True

        import json as _json
        out_path = self.mission_summary_out_path
        t0 = time.time()
        self.get_logger().info(
            f'[LOOP] waiting for LLM completion (timeout={timeout_s:.0f}s)...'
        )
        while self.loop_forever:
            rclpy.spin_once(self, timeout_sec=poll_interval_s)
            try:
                with open(out_path, 'r') as f:
                    data = _json.load(f)
                llm_ts = data.get('llm_policy_update', {}).get('timestamp')
                if llm_ts is not None and float(llm_ts) > goal_reached_time:
                    self.get_logger().info('[LOOP] LLM completed, proceeding.')
                    return True
            except Exception:
                pass
            if time.time() - t0 > timeout_s:
                self.get_logger().warn(
                    f'[LOOP] LLM timeout ({timeout_s:.0f}s), proceeding anyway.'
                )
                return False
        return False

    def _has_mission_events(self) -> bool:
        if not self.mission_summary_out_path:
            return False
        import json as _json
        summary_path = self.mission_summary_out_path.replace('_out.json', '.json')
        try:
            with open(summary_path, 'r') as f:
                data = _json.load(f)
            return bool(data.get('events'))
        except Exception:
            return False

    def _handle_exit2(self, timeout_s: float = 120.0):
        if self._robot_navigating:
            self._wait_for_robot_to_finish()
            goal_reached_time = time.time()
            if self._has_mission_events():
                self._wait_for_llm_completion(goal_reached_time, timeout_s=timeout_s)
            else:
                self.get_logger().info('[LOOP] no obstacle events, skipping LLM wait.')
        else:
            self.get_logger().info('[LOOP] robot not navigating at exit2, teleport immediately.')


    def move_segment(self, p0, p1, label='segment', speed_mps=None):
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        _spd = speed_mps if speed_mps is not None else self.speed_mps
        dist     = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        duration = dist / _spd
        steps    = max(1, int(duration * self.update_hz))
        dt       = 1.0 / self.update_hz
        yaw_path  = math.atan2(y1-y0, x1-x0)
        yaw_model = yaw_path + self.model_yaw_offset

        for i in range(steps + 1):
            a = i / steps
            x = x0 + a * (x1 - x0)
            y = y0 + a * (y1 - y0)
            z = z0 + a * (z1 - z0)
            self.set_pose(self.entity_name, x, y, z, yaw_model)
            time.sleep(dt)

    def move_segment_pair(self, p0, p1, label='segment'):
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        dist     = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        duration = dist / self.speed_mps
        steps    = max(1, int(duration * self.update_hz))
        dt       = 1.0 / self.update_hz
        yaw_path  = math.atan2(y1-y0, x1-x0)
        yaw_model = yaw_path + self.model_yaw_offset
        half = self.pair_offset / 2.0

        self.get_logger().info(
            f'[PAIR] {label} dist={dist:.2f}m steps={steps} yaw={yaw_model:.2f}')

        for i in range(steps + 1):
            a = i / steps
            cx = x0 + a * (x1 - x0)
            cy = y0 + a * (y1 - y0)
            cz = z0 + a * (z1 - z0)

            x1a = cx + self._perp[0] *  half
            y1a = cy + self._perp[1] *  half
            x2a = cx + self._perp[0] * -half
            y2a = cy + self._perp[1] * -half

            self.set_pose(self.entity_name,      x1a, y1a, cz, yaw_model)
            self.set_pose(self.pair_entity_name, x2a, y2a, cz, yaw_model)
            time.sleep(dt)


    def _start_once(self):
        if self.started:
            return
        self.started = True

        try:
            self.delete_entity(self.entity_name)
            if self.pair_mode:
                self.delete_entity(self.pair_entity_name)
            time.sleep(0.2)
        except Exception:
            pass

        yaw0 = math.atan2(
            self.enter[1] - self.standby[1],
            self.enter[0] - self.standby[0]
        ) + self.model_yaw_offset

        half = self.pair_offset / 2.0

        if self.pair_mode:
            s1 = self._offset_point(self.standby,  half)
            s2 = self._offset_point(self.standby, -half)

            ok1 = self.spawn_at(self.entity_name,      self.sdf_path,
                                s1[0], s1[1], s1[2], yaw0)
            ok2 = self.spawn_at(self.pair_entity_name, self.pair_sdf_path,
                                s2[0], s2[1], s2[2], yaw0)
            if not (ok1 and ok2):
                self.get_logger().error('[START] pair spawn failed, aborting')
                return

            self.get_logger().info(
                f'[PAIR] Spawned {self.entity_name} & {self.pair_entity_name} '
                f'at standby with offset={self.pair_offset}m')

            if self.start_delay_s > 0:
                self.get_logger().info(
                    f'Waiting {self.start_delay_s:.1f}s before moving')
                time.sleep(self.start_delay_s)

            self.move_segment_pair(self.standby, self.enter, 'standby->enter')
            self.move_segment_pair(self.enter,   self.exit,  'enter->exit')

        else:
            import random as _random

            yaw0 = math.atan2(
                self.enter[1] - self.standby[1],
                self.enter[0] - self.standby[0]
            ) + self.model_yaw_offset

            if self.spawn_mode == 'random_in_corridor':
                ratio = _random.uniform(self.spawn_ratio_min, self.spawn_ratio_max)
                sx = self.standby[0] + (self.enter[0] - self.standby[0]) * ratio
                sy = self.standby[1] + (self.enter[1] - self.standby[1]) * ratio
                spawn_pos = (sx, sy, self.standby[2])
                self.get_logger().info(
                    f'[START] random_in_corridor spawn ratio={ratio:.2f} pos=({sx:.2f},{sy:.2f})'
                )
            else:
                spawn_pos = self.standby

            ok = self.spawn_at(self.entity_name, self.sdf_path, *spawn_pos, yaw0)
            if not ok:
                self.get_logger().error('[START] spawn failed, aborting')
                return

            if self.start_delay_s > 0:
                self.get_logger().info(f'Waiting {self.start_delay_s:.1f}s before moving')
                time.sleep(self.start_delay_s)

            self.move_segment(spawn_pos,    self.enter, 'spawn->enter', self.speed_mps)
            self.move_segment(self.enter,   self.exit,  'enter->exit',  self.speed_approach_mps)
            self.move_segment(self.exit,    self.exit2, 'exit->exit2',  self.speed_approach_mps)

            while self.loop_forever:
                self._handle_exit2(timeout_s=120.0)
                if not self.loop_forever:
                    break

                yaw_tp = math.atan2(
                    self.enter[1] - self.standby[1],
                    self.enter[0] - self.standby[0]
                ) + self.model_yaw_offset

                if self.spawn_mode == 'random_in_corridor':
                    ratio = _random.uniform(self.spawn_ratio_min, self.spawn_ratio_max)
                    sx = self.standby[0] + (self.enter[0] - self.standby[0]) * ratio
                    sy = self.standby[1] + (self.enter[1] - self.standby[1]) * ratio
                    next_pos = (sx, sy, self.standby[2])
                    self.get_logger().info(
                        f'[LOOP] teleport exit2->random ratio={ratio:.2f} pos=({sx:.2f},{sy:.2f})'
                    )
                else:
                    next_pos = self.standby
                    self.get_logger().info('[LOOP] teleport exit2->standby')

                self.set_pose(self.entity_name, next_pos[0], next_pos[1], next_pos[2], yaw_tp)

                self.move_segment(next_pos,   self.enter, 'teleport->enter', self.speed_mps)
                self.move_segment(self.enter,  self.exit,  'enter->exit',     self.speed_approach_mps)
                self.move_segment(self.exit,   self.exit2, 'exit->exit2',     self.speed_approach_mps)


def main():
    rclpy.init()
    node = MovingObstacleController()
    try:
        node.get_logger().info('controller started')
        node._start_once()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
