"""Discrete PointNav environment over a CompositeNavMesh.

Motion model
------------
``forward`` advances ``profile.step_size`` metres along the current heading;
``left`` / ``right`` rotate by ``profile.turn_deg`` degrees; ``stop`` ends the
episode.  On a floor, the movement is delegated to that floor's Recast
``PathFinder.try_step``: walls, door clearances and ledge / void edges are
therefore enforced by the same bake that the planner uses (agent radius,
height, step climb, slope).  Stairs and ramps are followed as their spec walk
track: a forward action near a gate with the heading aligned to the track
climbs step by step, so the body actually changes height rather than teleporting
between storeys.

Observation
-----------
:meth:`PointNavEnv.observation` returns position, yaw / quaternion, collision,
step, done / success, distance and floor.  RGB-D is deliberately **not**
implemented here: attach a renderer with
``env.set_observation_provider(fn)``; ``fn(env)`` may return ``rgb`` and
``depth`` arrays (or any sensor dict), which are merged into the observation
under ``rgb`` / ``depth`` / ``sensor``.  Without a provider those fields are
``None`` and ``sensor.provider == "external"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from roomgen_nav.pathfinding import (
    CompositePath,
    angle_delta_deg,
    densify_polyline,
    heading_deg,
    point_distance,
    polyline_length,
    xy_distance,
)

ACTION_FORWARD = "forward"
ACTION_LEFT = "left"
ACTION_RIGHT = "right"
ACTION_STOP = "stop"
ACTIONS = (ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP)
_ACTION_ALIASES = {
    "move_forward": ACTION_FORWARD,
    "forward": ACTION_FORWARD,
    "turn_left": ACTION_LEFT,
    "left": ACTION_LEFT,
    "turn_right": ACTION_RIGHT,
    "right": ACTION_RIGHT,
    "stop": ACTION_STOP,
}


@dataclass
class _LinkTraversal:
    link: object
    direction: str  # "forward" | "reverse"
    nodes: List[tuple]
    index: int


class PointNavEnv:
    """One deterministic PointNav episode runner."""

    def __init__(self, scene_id: str, spec, profile, nav,
                 step_size: Optional[float] = None,
                 turn_deg: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 success_radius: Optional[float] = None,
                 link_enter_radius: Optional[float] = None,
                 link_enter_angle_deg: float = 60.0,
                 observation_provider: Optional[Callable] = None,
                 auto_success: bool = True):
        if scene_id != nav.entry.scene_id:
            raise ValueError(f"env scene '{scene_id}' != nav scene '{nav.entry.scene_id}'")
        self.scene_id = scene_id
        self.spec = spec
        self.profile = profile
        self.nav = nav
        self.step_size = float(step_size if step_size is not None else profile.step_size)
        self.turn_deg = float(turn_deg if turn_deg is not None else profile.turn_deg)
        self.max_steps = int(max_steps if max_steps is not None else profile.max_steps)
        self.success_radius = float(
            success_radius if success_radius is not None else profile.success_radius)
        self.link_enter_radius = float(
            link_enter_radius if link_enter_radius is not None
            else max(0.35, 1.4 * self.step_size))
        self.link_enter_angle_deg = float(link_enter_angle_deg)
        self.observation_provider = observation_provider
        #: When True (scripted oracle) entering ``success_radius`` ends the
        #: episode immediately.  A closed-loop policy sets this False so it can
        #: observe the arrival and issue the explicit ``stop`` action.
        self.auto_success = bool(auto_success)
        #: set by the evaluator: fn(scene_id, episode_id, seed) -> Episode-like
        self.episode_provider: Optional[Callable] = None
        self.episode = None
        self._reset_runtime(None)

    # ------------------------------------------------------------------ #
    # reset / episode state
    # ------------------------------------------------------------------ #
    def _reset_runtime(self, episode) -> None:
        self.episode = episode
        self.episode_id: Optional[int] = None
        self.seed: Optional[int] = None
        self.start_floor = ""
        self.goal_floor = ""
        self.start_position = (0.0, 0.0, 0.0)
        self.goal_position = (0.0, 0.0, 0.0)
        self.start_yaw_deg = 0.0
        self.position = self.start_position
        self.yaw_deg = self.start_yaw_deg
        self.active_floor = ""
        self.link_state: Optional[_LinkTraversal] = None
        self.step_count = 0
        self.collisions = 0
        self.consecutive_collisions = 0
        self.traveled_distance = 0.0
        self.last_action = ""
        self.last_action_collision = False
        self.done = False
        self.success = False
        self.failure_reason: Optional[str] = None
        self.actions: List[str] = []
        if episode is not None:
            self._load_episode(episode)

    def _load_episode(self, episode) -> None:
        self.episode = episode
        self.episode_id = int(getattr(episode, "episode_id"))
        self.seed = int(getattr(episode, "seed"))
        self.start_position = tuple(float(v) for v in episode.start_position)
        self.goal_position = tuple(float(v) for v in episode.goal_position)
        self.start_floor = str(episode.start_floor)
        self.goal_floor = str(episode.goal_floor)
        self.start_yaw_deg = float(episode.start_yaw_deg)
        self.position = self.start_position
        self.yaw_deg = self.start_yaw_deg
        self.active_floor = self.start_floor
        snapped = self.nav.snap(self.active_floor, self.position, tolerance=1.0)
        if snapped is None:
            raise RuntimeError(
                f"episode {self.episode_id}: start {self.start_position} is not on "
                f"floor '{self.active_floor}'")
        self.position = snapped

    def reset(self, scene_id: str, episode_id: int, seed: Optional[int] = None,
              episode=None) -> dict:
        """Reset to an episode; reproducible from ``(scene_id, episode_id, seed)``."""
        if scene_id != self.scene_id:
            raise ValueError(f"env scene '{self.scene_id}', reset requested '{scene_id}'")
        if episode is None:
            if self.episode_provider is None:
                raise RuntimeError("no episode_provider set; pass episode= or attach a catalog")
            episode = self.episode_provider(scene_id, int(episode_id), seed)
        if seed is not None and int(getattr(episode, "seed")) != int(seed):
            # The catalog regenerated the episode; the stored seed must win so a
            # reset cannot silently mix two random streams.
            seed = int(getattr(episode, "seed"))
        self._reset_runtime(episode)
        return self.observation()

    # ------------------------------------------------------------------ #
    # stepping
    # ------------------------------------------------------------------ #
    def step(self, action: str) -> dict:
        if self.done:
            return self.observation()
        name = str(action).strip().lower()
        if name not in _ACTION_ALIASES:
            self.done = True
            self.failure_reason = "invalid_action"
            return self.observation()
        name = _ACTION_ALIASES[name]
        self.last_action = name
        self.last_action_collision = False
        self.actions.append(name)

        if name == ACTION_LEFT:
            self.yaw_deg = _wrap_degrees(self.yaw_deg + self.turn_deg)
        elif name == ACTION_RIGHT:
            self.yaw_deg = _wrap_degrees(self.yaw_deg - self.turn_deg)
        elif name == ACTION_STOP:
            self.done = True
            self.success = self._at_goal()
            if not self.success:
                self.failure_reason = "stopped_short"
        else:
            self._forward()

        self.step_count += 1
        if not self.done:
            if self.auto_success:
                self.success = self._at_goal()
                if self.success:
                    self.done = True
                elif self.consecutive_collisions >= 10:
                    self.done = True
                    self.failure_reason = "stuck"
                elif self.step_count >= self.max_steps:
                    self.done = True
                    self.failure_reason = "timeout"
            else:
                # The policy sees the arrival in its next observation and emits
                # ``stop`` itself.  Landing on the goal exactly at the budget is
                # still a success, not a timeout.
                if self._at_goal() and self.step_count >= self.max_steps:
                    self.success = True
                    self.done = True
                elif self.consecutive_collisions >= 10:
                    self.done = True
                    self.failure_reason = "stuck"
                elif self.step_count >= self.max_steps:
                    self.done = True
                    self.failure_reason = "timeout"
        return self.observation()

    def _forward(self) -> None:
        if self.link_state is not None:
            moved = self._advance_link(self.step_size)
            self.last_action_collision = moved <= 1e-6
            if self.last_action_collision:
                self._note_collision()
            else:
                self.consecutive_collisions = 0
            return

        traversal = self._link_at_current_pose()
        if traversal is not None:
            gate = traversal.nodes[0]
            approach = xy_distance(self.position, gate)
            if approach > 0.0:
                self.traveled_distance += approach
            self.position = gate
            self.link_state = traversal
            moved = self._advance_link(max(0.0, self.step_size - approach))
            self.last_action_collision = moved <= 1e-6
            self.consecutive_collisions = 0
            return

        dx, dz = math.cos(math.radians(self.yaw_deg)), -math.sin(math.radians(self.yaw_deg))
        end = (self.position[0] + dx * self.step_size, self.position[1],
               self.position[2] + dz * self.step_size)
        stepped = self.nav.floor(self.active_floor).try_step(self.position, end)
        if stepped is None:
            self._note_collision()
            return
        moved = point_distance(self.position, stepped)
        if moved <= max(1e-6, 0.2 * self.step_size):
            self._note_collision()
            # try_step may still return a small slide; keep it only if it moved.
            if moved > 1e-6:
                self.traveled_distance += moved
                self.position = stepped
            return
        self.traveled_distance += moved
        self.position = stepped
        self.last_action_collision = False
        self.consecutive_collisions = 0

    def _note_collision(self) -> None:
        self.last_action_collision = True
        self.collisions += 1
        self.consecutive_collisions += 1

    def _link_at_current_pose(self) -> Optional[_LinkTraversal]:
        for link in self.nav.valid_links:
            options = []
            if link.from_floor == self.active_floor:
                options.append(("forward", link.track))
            if link.to_floor == self.active_floor:
                options.append(("reverse", list(reversed(link.track))))
            for direction, nodes in options:
                if len(nodes) < 2:
                    continue
                gate = nodes[0]
                if xy_distance(self.position, gate) > self.link_enter_radius:
                    continue
                desired = heading_deg(nodes[0], nodes[1])
                error = abs(angle_delta_deg(desired, self.yaw_deg))
                if error > self.link_enter_angle_deg:
                    continue
                return _LinkTraversal(link=link, direction=direction,
                                      nodes=[tuple(p) for p in nodes], index=1)
        return None

    def _advance_link(self, distance: float) -> float:
        state = self.link_state
        if state is None:
            return 0.0
        remaining = float(distance)
        moved = 0.0
        while remaining > 1e-9 and state.index < len(state.nodes):
            target = state.nodes[state.index]
            segment = point_distance(self.position, target)
            if segment <= remaining + 1e-12:
                self.position = target
                moved += segment
                remaining -= segment
                state.index += 1
            else:
                t = remaining / segment if segment > 1e-12 else 0.0
                self.position = (self.position[0] + (target[0] - self.position[0]) * t,
                                 self.position[1] + (target[1] - self.position[1]) * t,
                                 self.position[2] + (target[2] - self.position[2]) * t)
                moved += remaining
                remaining = 0.0
        self.traveled_distance += moved
        if state.index >= len(state.nodes):
            self.position = state.nodes[-1]
            if state.direction == "forward":
                self.active_floor = state.link.to_floor
            else:
                self.active_floor = state.link.from_floor
            self.link_state = None
        return moved

    # ------------------------------------------------------------------ #
    # observations / evaluation
    # ------------------------------------------------------------------ #
    def distance_to_goal(self) -> float:
        return point_distance(self.position, self.goal_position)

    def _at_goal(self) -> bool:
        return (self.active_floor == self.goal_floor
                and self.distance_to_goal() <= self.success_radius)

    def rotation_quaternion(self) -> List[float]:
        half = math.radians(self.yaw_deg) * 0.5
        return [0.0, math.sin(half), 0.0, math.cos(half)]

    def set_observation_provider(self, provider: Optional[Callable]) -> None:
        """Attach the external RGB-D renderer (``None`` disables it)."""
        self.observation_provider = provider

    def attach_catalog(self, catalog) -> None:
        """Make ``reset(scene_id, episode_id, seed)`` look episodes up itself."""
        self.episode_provider = (
            lambda scene_id, episode_id, seed: catalog.get(episode_id, seed))

    def observation(self) -> dict:
        distance = self.distance_to_goal()
        sensor = {"provider": "external", "rgb": None, "depth": None}
        obs = {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "position": [float(v) for v in self.position],
            "heading_deg": float(self.yaw_deg),
            "rotation": self.rotation_quaternion(),
            "collision": bool(self.last_action_collision),
            "step": int(self.step_count),
            "done": bool(self.done),
            "success": bool(self.success),
            "distance_to_goal": float(distance),
            "floor_id": self.active_floor,
            "on_link": self.link_state is not None,
            "traveled_distance": float(self.traveled_distance),
            "failure_reason": self.failure_reason,
            "rgb": None,
            "depth": None,
            "sensor": sensor,
        }
        if self.observation_provider is not None:
            extra = self.observation_provider(self)
            if isinstance(extra, dict):
                sensor.update(extra)
                if "rgb" in extra:
                    obs["rgb"] = extra["rgb"]
                if "depth" in extra:
                    obs["depth"] = extra["depth"]
        return obs

    def evaluate(self) -> dict:
        """Metrics of the finished episode (success / SPL / collision / reason)."""
        success = bool(self.success and self._at_goal())
        shortest = float(getattr(self.episode, "shortest_distance", 0.0) or 0.0)
        length = max(float(self.traveled_distance), 1e-9)
        spl = (shortest / max(length, shortest)) if success and shortest > 0.0 else 0.0
        return {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "success": success,
            "spl": float(spl),
            "collisions": int(self.collisions),
            "steps": int(self.step_count),
            "timeout": bool(self.failure_reason == "timeout"),
            "final_distance": float(self.distance_to_goal()),
            "path_length": float(length),
            "shortest_distance": shortest,
            "failure_reason": self.failure_reason,
            "start_floor": self.start_floor,
            "goal_floor": self.goal_floor,
            "cross_floor": self.start_floor != self.goal_floor,
            "links_used": list(getattr(self.episode, "links_used", []) or []),
        }

    def rollout(self, actions: Sequence[str]) -> dict:
        for action in actions:
            if self.done:
                break
            self.step(action)
        return self.observation()

    def replay_episode(self) -> dict:
        """Replay the episode's own action list from its stored initial state."""
        if self.episode is None:
            raise RuntimeError("no episode loaded")
        self.reset(self.scene_id, self.episode.episode_id, self.episode.seed,
                   episode=self.episode)
        return self.rollout(getattr(self.episode, "actions", []) or [])


# --------------------------------------------------------------------------- #
# closed-loop path policy
# --------------------------------------------------------------------------- #
def trajectory_row(observation: dict, action: str, collision=None) -> dict:
    """One replay-compatible trajectory row (position, yaw, action, collision)."""
    return {
        "position": [float(value) for value in observation["position"]],
        "yaw": float(observation["heading_deg"]),
        "action": action,
        "collision": bool(observation["collision"] if collision is None else collision),
        "floor_id": observation.get("floor_id"),
    }


class ShortestPathPolicy:
    """Closed-loop PointNav policy that follows the reference path waypoints.

    The policy never reads ``episode.actions`` (the stored oracle sequence).
    On every step it locates the current pose on the reference polyline, takes a
    look-ahead point on that polyline and emits a single discrete action:

    * if the goal floor and ``success_radius`` are reached -> ``stop``;
    * while a stair / ramp link is active the link track drives the body, so
      -> ``forward``;
    * heading error larger than the alignment threshold -> ``left`` / ``right``;
    * otherwise -> ``forward``.

    Motion parameters come from the env / :class:`AgentProfile`: ``step_size``
    and ``turn_deg`` are the profile values, the alignment threshold defaults to
    ``0.75 * turn_deg`` and the look-ahead defaults to ``0.6 * step_size``.
    Collisions are counted by :class:`PointNavEnv`; the policy only reacts to
    the position it observes after each action.
    """

    name = "path"

    def __init__(self, env: PointNavEnv, align_deg: Optional[float] = None,
                 lookahead: Optional[float] = None):
        profile = env.profile
        self.env = env
        self.profile = profile
        self.step_size = float(env.step_size)
        self.turn_deg = float(env.turn_deg)
        self.align_deg = float(
            align_deg if align_deg is not None else max(0.5, self.turn_deg * 0.75))
        self.lookahead = float(
            lookahead if lookahead is not None else max(0.05, self.step_size * 0.6))
        if self.align_deg <= 0.0:
            raise ValueError("alignment threshold must be positive")
        if self.lookahead <= 0.0:
            raise ValueError("look-ahead distance must be positive")
        self.episode = None
        self.path_points: List[tuple] = []
        self.dense: List[tuple] = []
        self._cursor = 0
        self._was_on_link = False

    def describe(self) -> dict:
        return {
            "name": self.name,
            "source": "reference_path waypoints",
            "step_size": self.step_size,
            "turn_deg": self.turn_deg,
            "align_deg": round(self.align_deg, 4),
            "lookahead": round(self.lookahead, 4),
        }

    # -- episode setup ------------------------------------------------------ #
    def reset(self, episode=None, path=None) -> None:
        if episode is not None:
            self.episode = episode
        if self.episode is None:
            raise RuntimeError("reset the env to an episode before running the policy")
        if path is None:
            path = getattr(self.episode, "reference_path", None) or []
        if hasattr(path, "points"):
            path = path.points
        points = [tuple(float(value) for value in point) for point in path or []]
        if len(points) < 2:
            points = [tuple(self.env.start_position), tuple(self.env.goal_position)]
        start = tuple(float(value) for value in self.env.position)
        goal = tuple(float(value) for value in self.env.goal_position)
        if point_distance(start, points[0]) > 1e-6:
            points.insert(0, start)
        if point_distance(goal, points[-1]) > 1e-6:
            points.append(goal)
        self.path_points = points
        self.dense = densify_polyline(
            points, max_spacing=max(0.02, min(0.08, self.step_size * 0.4)))
        self._cursor = 0
        self._was_on_link = False

    # -- control ------------------------------------------------------------ #
    def act(self) -> Optional[str]:
        """Return the next discrete action, or None when the episode is over."""
        env = self.env
        if env.done:
            return None
        if env._at_goal():
            return ACTION_STOP
        if env.link_state is not None:
            self._was_on_link = True
            return ACTION_FORWARD
        if self._was_on_link:
            # The link moved the body along its track without the policy looking
            # at the polyline; re-locate the cursor once on the destination floor.
            self._was_on_link = False
            self._cursor = min(
                range(len(self.dense)),
                key=lambda index: point_distance(env.position, self.dense[index]))
        target = self._lookahead_target(env.position)
        horizontal = xy_distance(env.position, target)
        vertical = abs(float(env.position[1]) - float(target[1]))
        if horizontal < 1e-6 and vertical < 1e-6:
            if env._at_goal():
                return ACTION_STOP
            target = env.goal_position
        error = angle_delta_deg(heading_deg(env.position, target), env.yaw_deg)
        if abs(error) > self.align_deg:
            return ACTION_LEFT if error > 0.0 else ACTION_RIGHT
        return ACTION_FORWARD

    def _lookahead_target(self, position: Sequence[float]) -> tuple:
        dense = self.dense
        if not dense:
            return tuple(position)
        count = len(dense)
        low = max(0, self._cursor - 16)
        high = min(count, self._cursor + 65)
        closest = min(range(low, high),
                      key=lambda index: point_distance(position, dense[index]))
        if closest == low and low > 0:
            closest = min(range(count),
                          key=lambda index: point_distance(position, dense[index]))
        self._cursor = closest
        target = closest
        while target < count - 1 and \
                point_distance(position, dense[target]) < self.lookahead:
            target += 1
        return dense[target]

    def run(self, episode=None) -> dict:
        """Run the closed loop from a fresh reset and return actions + trajectory."""
        if episode is not None:
            self.env.reset(self.env.scene_id, int(episode.episode_id),
                           int(episode.seed), episode=episode)
        self.reset(episode)
        env = self.env
        actions: List[str] = []
        trajectory = [trajectory_row(env.observation(), "start", False)]
        while not env.done:
            action = self.act()
            if action is None:
                break
            observation = env.step(action)
            actions.append(action)
            trajectory.append(trajectory_row(observation, action))
        return {
            "policy": self.name,
            "actions": actions,
            "trajectory": trajectory,
            "result": env.evaluate(),
        }


def run_action_sequence(env: PointNavEnv, actions: Sequence[str]) -> dict:
    """Execute a stored action list and record the same replay trajectory shape."""
    executed: List[str] = []
    trajectory = [trajectory_row(env.observation(), "start", False)]
    for action in actions:
        if env.done:
            break
        observation = env.step(action)
        executed.append(action)
        trajectory.append(trajectory_row(observation, action))
    return {
        "policy": "oracle",
        "actions": executed,
        "trajectory": trajectory,
        "result": env.evaluate(),
    }


# --------------------------------------------------------------------------- #
# reference action scripting
# --------------------------------------------------------------------------- #
def script_actions(env: PointNavEnv, path: Optional[CompositePath] = None,
                   max_actions: Optional[int] = None) -> List[str]:
    """Drive ``env`` along ``path`` and return the executed action list.

    The controller is closed-loop during generation (it reads the real position
    after each action), but the returned list is open-loop: replaying it from the
    stored initial state is deterministic.  The list always ends with ``stop``.
    """
    if env.episode is None:
        raise RuntimeError("reset the env to an episode before scripting")
    points = list(path.points) if path is not None else []
    if not points:
        points = [tuple(env.start_position), tuple(env.goal_position)]
    dense = densify_polyline(points, max_spacing=max(0.08, env.step_size * 0.5))
    if not dense:
        return [ACTION_STOP]
    actions: List[str] = []
    limit = int(max_actions if max_actions is not None else env.max_steps)
    stuck = 0
    lookahead = max(0.10, env.step_size * 0.6)
    while len(actions) < limit and not env.done:
        if env._at_goal():
            break
        # Track the path by the closest point, not a monotone index: a body
        # moving 0.25 m per action can pass several 0.125 m samples at once,
        # and a switchback legitimately comes back towards its entry.
        closest = min(range(len(dense)),
                      key=lambda i: point_distance(env.position, dense[i]))
        target_index = closest
        while target_index < len(dense) - 1 and \
                point_distance(env.position, dense[target_index]) < lookahead:
            target_index += 1
        if target_index == closest and point_distance(env.position, dense[closest]) < 0.02:
            target_index = min(len(dense) - 1, closest + 1)
        target = dense[target_index]
        horizontal = xy_distance(env.position, target)
        vertical = abs(float(env.position[1]) - float(target[1]))
        if horizontal < 0.02 and vertical < 0.02:
            if env._at_goal() or closest >= len(dense) - 1:
                break
            # On the spot and the path continues: face the next sample by
            # nudging the target one step further.
            target = dense[min(len(dense) - 1, target_index + 1)]
        desired = heading_deg(env.position, target)
        error = angle_delta_deg(desired, env.yaw_deg)
        if abs(error) > env.turn_deg * 0.75:
            action = ACTION_LEFT if error > 0 else ACTION_RIGHT
            env.step(action)
            actions.append(action)
            stuck = 0
            continue
        before = env.position
        env.step(ACTION_FORWARD)
        actions.append(ACTION_FORWARD)
        if env.last_action_collision:
            stuck += 1
            if stuck > 6:
                break
        else:
            stuck = 0
        if point_distance(before, env.position) < 1e-6:
            # A collision can leave the target unchanged; turn one notch so the
            # next attempt slides along the wall instead of ramming it.
            env.step(ACTION_LEFT)
            actions.append(ACTION_LEFT)
    if not env.done:
        env.step(ACTION_STOP)
        actions.append(ACTION_STOP)
    elif not actions or actions[-1] != ACTION_STOP:
        actions.append(ACTION_STOP)
    return actions


def _wrap_degrees(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle
