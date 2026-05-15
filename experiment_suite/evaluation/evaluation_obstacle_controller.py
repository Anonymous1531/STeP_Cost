#!/usr/bin/env python3
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from gazebo_msgs.msg import EntityState
from tf_transformations import quaternion_from_euler


class MovingObstacleController(Node):
    def __init__(self):
        super().__init__('factory6_moving_obstacle_controller')

        self.declare_parameter('entity_name', 'exp_box')
        self.declare_parameter('sdf_path', '')
        self.declare_parameter('start_delay_s', 3.0)
        self.declare_parameter('speed_mps', 0.05)
        self.declare_parameter('update_hz', 10.0)
        self.declare_parameter('despawn_on_finish', False)
        self.declare_parameter('loop_forever', False)
        self.declare_parameter('speed_approach_mps', 0.0)
        self.declare_parameter('despawn_on_exit2', False)  
        self.declare_parameter('ghost_sdf_path', '') 

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
        self.speed_mps     = max(1e-3, float(self.get_parameter('speed_mps').value))
        _approach = float(self.get_parameter('speed_approach_mps').value)
        self.speed_approach_mps = _approach if _approach > 1e-4 else self.speed_mps * 2.0
        self.update_hz     = max(1.0,  float(self.get_parameter('update_hz').value))
        self.despawn_on_finish = bool(self.get_parameter('despawn_on_finish').value)
        self.loop_forever      = bool(self.get_parameter('loop_forever').value)
        self.despawn_on_exit2  = bool(self.get_parameter('despawn_on_exit2').value)
        ghost_sdf = str(self.get_parameter('ghost_sdf_path').value).strip()
        self.ghost_sdf_path = ghost_sdf if ghost_sdf else None

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
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        time.sleep(0.5)  

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


    def move_segment(self, p0, p1, label='segment', speed: float = 0.0):

        spd = speed if speed > 1e-4 else self.speed_mps
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        dist     = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        duration = dist / spd
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
            ok = self.spawn_at(self.entity_name, self.sdf_path,
                               *self.standby, yaw0)
            if not ok:
                self.get_logger().error('[START] spawn failed, aborting')
                return

            if self.start_delay_s > 0:
                self.get_logger().info(
                    f'Waiting {self.start_delay_s:.1f}s before moving')
                time.sleep(self.start_delay_s)

            self.move_segment(self.standby, self.enter, 'standby->enter', self.speed_mps)

            if self.ghost_sdf_path:
                self.get_logger().info(
                    f'[GHOST] reached enter, respawning as ghost: {self.ghost_sdf_path}'
                )
                yaw_enter = math.atan2(
                    self.exit[1] - self.enter[1],
                    self.exit[0] - self.enter[0]
                ) + self.model_yaw_offset
                self.delete_entity(self.entity_name)
                time.sleep(0.3)
                ok = self.spawn_at(
                    self.entity_name, self.ghost_sdf_path,
                    self.enter[0], self.enter[1], self.enter[2], yaw_enter
                )
                if not ok:
                    self.get_logger().error('[GHOST] ghost respawn failed, continuing with original')

            self.move_segment(self.enter,   self.exit,  'enter->exit',    self.speed_approach_mps)
            self.move_segment(self.exit,    self.exit2, 'exit->exit2',    self.speed_approach_mps)

            if self.despawn_on_exit2:
                self.get_logger().info(f'[DESPAWN] {self.entity_name} reached exit2, deleting')
                self.delete_entity(self.entity_name)
                return

            while self.loop_forever:
                yaw_tp = math.atan2(
                    self.enter[1] - self.standby[1],
                    self.enter[0] - self.standby[0]
                ) + self.model_yaw_offset
                self.set_pose(self.entity_name,
                              self.standby[0], self.standby[1], self.standby[2], yaw_tp)
                self.get_logger().info('[LOOP] teleport exit2->standby')
                self.move_segment(self.standby, self.enter, 'standby->enter', self.speed_mps)

                if self.ghost_sdf_path:
                    yaw_enter = math.atan2(
                        self.exit[1] - self.enter[1],
                        self.exit[0] - self.enter[0]
                    ) + self.model_yaw_offset
                    self.delete_entity(self.entity_name)
                    time.sleep(0.3)
                    self.spawn_at(
                        self.entity_name, self.ghost_sdf_path,
                        self.enter[0], self.enter[1], self.enter[2], yaw_enter
                    )

                self.move_segment(self.enter,   self.exit,  'enter->exit',    self.speed_approach_mps)
                self.move_segment(self.exit,    self.exit2, 'exit->exit2',    self.speed_approach_mps)

                if self.despawn_on_exit2:
                    self.get_logger().info(f'[DESPAWN] {self.entity_name} reached exit2, deleting')
                    self.delete_entity(self.entity_name)
                    return


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
