"""Simulator true-map A* planner and one-shot action compiler for SSA."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import maps as habitat_maps


GridPoint = Tuple[int, int]


@dataclass
class SSAPlan:
    actions: List[int]
    target_position: np.ndarray
    target_yaw_deg: float
    error: str = ""


class SimulatorAStarPlanner:
    """Plan an SSA takeover sequence on the simulator true topdown map."""

    def __init__(
        self,
        *,
        map_resolution: int = 1024,
        forward_step_m: float = 0.25,
        turn_angle_deg: float = 30.0,
    ):
        self.map_resolution = int(map_resolution)
        self.forward_step_m = float(forward_step_m)
        self.turn_angle_deg = float(turn_angle_deg)

    def build_plan(self, env, pose: Dict[str, float]) -> SSAPlan:
        agent_state = env.sim.get_agent_state()
        start_position = np.array(agent_state.position, dtype=np.float32)
        target_position = self._target_world_position(agent_state, pose)
        topdown = habitat_maps.get_topdown_map_from_sim(
            env.sim,
            map_resolution=self.map_resolution,
            draw_border=False,
            meters_per_pixel=None,
        )
        free_mask = np.asarray(topdown) == 1
        start_rc = self._to_grid(env.sim, topdown.shape[:2], start_position)
        goal_rc = self._to_grid(env.sim, topdown.shape[:2], target_position)
        goal_rc = self._snap_to_free(free_mask, goal_rc)
        if goal_rc is None:
            return SSAPlan([], target_position, float(pose.get("yaw_deg", 0.0) or 0.0), error="goal_not_reachable_on_map")

        path = self._astar(free_mask, start_rc, goal_rc)
        if not path:
            return SSAPlan([], target_position, float(pose.get("yaw_deg", 0.0) or 0.0), error="astar_path_not_found")

        sparse_world_points = self._path_to_world_points(env.sim, topdown.shape[:2], path)
        actions = self._compile_actions(
            start_position=start_position,
            start_rotation=agent_state.rotation,
            sparse_world_points=sparse_world_points,
            final_yaw_deg=float(pose.get("yaw_deg", 0.0) or 0.0),
        )
        return SSAPlan(actions, target_position, float(pose.get("yaw_deg", 0.0) or 0.0), error="")

    def reached_target(self, env, target_position: np.ndarray, target_yaw_deg: float) -> bool:
        agent_state = env.sim.get_agent_state()
        position = np.array(agent_state.position, dtype=np.float32)
        distance = float(np.linalg.norm(position[[0, 2]] - target_position[[0, 2]]))
        yaw_error = abs(self._angle_diff_deg(self._current_yaw_deg(agent_state.rotation), target_yaw_deg))
        return distance <= 0.5 and yaw_error <= 30.0

    def _compile_actions(
        self,
        *,
        start_position: np.ndarray,
        start_rotation,
        sparse_world_points: Sequence[np.ndarray],
        final_yaw_deg: float,
    ) -> List[int]:
        actions: List[int] = []
        sim_position = np.array(start_position, dtype=np.float32)
        sim_yaw = self._current_yaw_deg(start_rotation)

        for waypoint in sparse_world_points:
            delta = np.array([waypoint[0] - sim_position[0], 0.0, waypoint[2] - sim_position[2]], dtype=np.float32)
            planar_dist = float(np.linalg.norm(delta[[0, 2]]))
            if planar_dist < 1e-3:
                continue
            desired_yaw = self._yaw_to_target(sim_position, waypoint)
            actions.extend(self._turn_actions(sim_yaw, desired_yaw))
            sim_yaw = self._snap_yaw(desired_yaw)
            forward_steps = max(1, int(round(planar_dist / self.forward_step_m)))
            actions.extend([1] * forward_steps)
            rad = math.radians(sim_yaw)
            sim_position = np.array(
                [
                    sim_position[0] + math.sin(rad) * forward_steps * self.forward_step_m,
                    sim_position[1],
                    sim_position[2] - math.cos(rad) * forward_steps * self.forward_step_m,
                ],
                dtype=np.float32,
            )

        actions.extend(self._turn_actions(sim_yaw, final_yaw_deg))
        return actions

    def _path_to_world_points(self, sim, shape: Tuple[int, int], path: Sequence[GridPoint]) -> List[np.ndarray]:
        if not path:
            return []
        sparse: List[np.ndarray] = []
        last_point: Optional[np.ndarray] = None
        for idx, point in enumerate(path):
            world = self._grid_to_world(sim, shape, point)
            if last_point is None:
                last_point = world
                continue
            dist = float(np.linalg.norm(world[[0, 2]] - last_point[[0, 2]]))
            direction_changed = False
            if 0 < idx < len(path) - 1:
                prev = path[idx - 1]
                curr = path[idx]
                nxt = path[idx + 1]
                direction_changed = (curr[0] - prev[0], curr[1] - prev[1]) != (nxt[0] - curr[0], nxt[1] - curr[1])
            if dist >= self.forward_step_m * 2.0 or direction_changed or idx == len(path) - 1:
                sparse.append(world)
                last_point = world
        if not sparse and path:
            sparse.append(self._grid_to_world(sim, shape, path[-1]))
        return sparse

    def _target_world_position(self, agent_state, pose: Dict[str, float]) -> np.ndarray:
        x_forward = float(pose.get("x_forward_m", 0.0) or 0.0)
        right_m = float(pose.get("right_m", 0.0) or 0.0)
        forward = quaternion_rotate_vector(agent_state.rotation, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        right = quaternion_rotate_vector(agent_state.rotation, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        forward[1] = 0.0
        right[1] = 0.0
        forward = forward / max(1e-6, float(np.linalg.norm(forward)))
        right = right / max(1e-6, float(np.linalg.norm(right)))
        target = np.array(agent_state.position, dtype=np.float32) + x_forward * forward + right_m * right
        target[1] = agent_state.position[1]
        return target.astype(np.float32)

    def _to_grid(self, sim, shape: Tuple[int, int], position: np.ndarray) -> GridPoint:
        row, col = habitat_maps.to_grid(
            float(position[2]),
            float(position[0]),
            shape,
            sim=sim,
        )
        return int(row), int(col)

    def _grid_to_world(self, sim, shape: Tuple[int, int], point: GridPoint) -> np.ndarray:
        lower, upper = sim.pathfinder.get_bounds()
        row, col = int(point[0]), int(point[1])
        row_scale = abs(float(upper[2]) - float(lower[2])) / max(1, int(shape[0]))
        col_scale = abs(float(upper[0]) - float(lower[0])) / max(1, int(shape[1]))
        world_z = float(lower[2]) + (row + 0.5) * row_scale
        world_x = float(lower[0]) + (col + 0.5) * col_scale
        return np.array([world_x, float(lower[1]), world_z], dtype=np.float32)

    def _turn_actions(self, current_yaw_deg: float, target_yaw_deg: float) -> List[int]:
        delta = self._angle_diff_deg(target_yaw_deg, current_yaw_deg)
        if abs(delta) < self.turn_angle_deg * 0.5:
            return []
        steps = int(round(abs(delta) / self.turn_angle_deg))
        action = 2 if delta > 0.0 else 3
        return [action] * max(1, steps)

    @staticmethod
    def _astar(free_mask: np.ndarray, start: GridPoint, goal: GridPoint) -> List[GridPoint]:
        if start == goal:
            return [start]
        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.4142),
            (-1, 1, 1.4142),
            (1, -1, 1.4142),
            (1, 1, 1.4142),
        )
        open_heap = [(0.0, start)]
        came_from: Dict[GridPoint, GridPoint] = {}
        g_score: Dict[GridPoint, float] = {start: 0.0}
        closed = set()
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return SimulatorAStarPlanner._reconstruct(came_from, current)
            closed.add(current)
            row, col = current
            for d_row, d_col, cost in neighbors:
                n_row, n_col = row + d_row, col + d_col
                if not (0 <= n_row < free_mask.shape[0] and 0 <= n_col < free_mask.shape[1]):
                    continue
                if not free_mask[n_row, n_col]:
                    continue
                neighbor = (n_row, n_col)
                cand = g_score[current] + cost
                if cand >= g_score.get(neighbor, 1e18):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = cand
                heuristic = float(math.hypot(goal[0] - n_row, goal[1] - n_col))
                heapq.heappush(open_heap, (cand + heuristic, neighbor))
        return []

    @staticmethod
    def _reconstruct(came_from: Dict[GridPoint, GridPoint], current: GridPoint) -> List[GridPoint]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _snap_to_free(free_mask: np.ndarray, goal: GridPoint, max_radius: int = 20) -> Optional[GridPoint]:
        row, col = goal
        if 0 <= row < free_mask.shape[0] and 0 <= col < free_mask.shape[1] and free_mask[row, col]:
            return row, col
        best = None
        best_dist = 1e18
        for radius in range(1, max_radius + 1):
            for d_row in range(-radius, radius + 1):
                for d_col in range(-radius, radius + 1):
                    n_row, n_col = row + d_row, col + d_col
                    if not (0 <= n_row < free_mask.shape[0] and 0 <= n_col < free_mask.shape[1]):
                        continue
                    if not free_mask[n_row, n_col]:
                        continue
                    dist = d_row * d_row + d_col * d_col
                    if dist < best_dist:
                        best = (n_row, n_col)
                        best_dist = dist
            if best is not None:
                return best
        return None

    @staticmethod
    def _current_yaw_deg(rotation) -> float:
        heading_vector = quaternion_rotate_vector(rotation.inverse(), np.array([0.0, 0.0, -1.0], dtype=np.float32))
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return float(np.rad2deg(np.array(phi) + np.pi))

    @staticmethod
    def _yaw_to_target(start_position: np.ndarray, target_position: np.ndarray) -> float:
        delta_x = float(target_position[0] - start_position[0])
        delta_z = float(target_position[2] - start_position[2])
        return float((math.degrees(math.atan2(delta_x, -delta_z)) + 360.0) % 360.0)

    def _snap_yaw(self, yaw_deg: float) -> float:
        steps = round(float(yaw_deg) / self.turn_angle_deg)
        return float((steps * self.turn_angle_deg) % 360.0)

    @staticmethod
    def _angle_diff_deg(target_deg: float, current_deg: float) -> float:
        delta = (float(target_deg) - float(current_deg) + 180.0) % 360.0 - 180.0
        return float(delta)
