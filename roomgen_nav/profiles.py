"""Agent profiles that decide what a PointNav walker can traverse.

The profile is the single source of truth for both the Recast bake
(:class:`~roomgen_nav.navmesh.CompositeNavMesh`) and the discrete motion model
(:class:`~roomgen_nav.env.PointNavEnv`):

* ``radius`` / ``height`` become Recast's agent radius and height, and the same
  radius sets the clearance a sampled start/goal must keep from walls;
* ``max_step`` becomes Recast's ``agent_max_climb`` and the largest riser the
  walker may climb on a stair track;
* ``max_slope_deg`` becomes Recast's ``agent_max_slope`` and the steepest ramp
  the walker may use;
* ``locomotion`` documents the class of body (``humanoid`` / ``wheeled``) and is
  used for reporting only -- reachability always comes from the numbers.

Two profiles ship: ``humanoid`` (climbs the generated stairs) and ``wheeled``
(cannot climb a stair riser and only accepts near-flat ramps).  Add a profile by
constructing :class:`AgentProfile` and passing it to the builder; caches are
keyed by :meth:`AgentProfile.fingerprint`, so a new profile never reuses an old
NavMesh.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass(frozen=True)
class AgentProfile:
    """Reachability and motion parameters of one PointNav agent."""

    name: str
    radius: float
    height: float
    max_step: float
    max_slope_deg: float
    locomotion: str = "humanoid"
    #: distance of one discrete ``forward`` action (m)
    step_size: float = 0.25
    #: yaw change of one ``left`` / ``right`` action (deg)
    turn_deg: float = 15.0
    #: episode budget in actions
    max_steps: int = 500
    #: a goal is reached when the 3D distance is at most this (m)
    success_radius: float = 0.5
    #: Recast cell size / height for the bake
    cell_size: float = 0.05
    cell_height: float = 0.02

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("agent radius must be positive")
        if self.height <= 0.0:
            raise ValueError("agent height must be positive")
        if self.max_step < 0.0:
            raise ValueError("agent max_step must be non-negative")
        if not (0.0 < self.max_slope_deg < 90.0):
            raise ValueError("agent max_slope_deg must be in (0, 90)")
        if self.locomotion not in ("humanoid", "wheeled"):
            raise ValueError("locomotion must be 'humanoid' or 'wheeled'")
        if self.step_size <= 0.0 or self.turn_deg <= 0.0:
            raise ValueError("step_size and turn_deg must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.success_radius <= 0.0:
            raise ValueError("success_radius must be positive")
        if self.cell_size <= 0.0 or self.cell_height <= 0.0:
            raise ValueError("cell size / height must be positive")

    @property
    def max_slope(self) -> float:
        """Maximum ramp slope as a rise/run ratio (tan of the angle)."""
        return math.tan(math.radians(self.max_slope_deg))

    def can_climb_riser(self, rise: float, tolerance: float = 0.01) -> bool:
        """True when a stair riser is within this agent's step height."""
        return float(rise) <= self.max_step + tolerance

    def can_traverse_slope(self, rise: float, run: float,
                           tolerance: float = 0.01) -> bool:
        """True when a ramp is within this agent's slope limit."""
        if run <= 0.0:
            return False
        return float(rise) / float(run) <= self.max_slope + tolerance

    def navmesh_settings(self) -> Dict[str, float]:
        """Recast parameters derived from this profile (no habitat import)."""
        return {
            "agent_radius": float(self.radius),
            "agent_height": float(self.height),
            "agent_max_climb": float(self.max_step),
            "agent_max_slope": float(self.max_slope_deg),
            "cell_size": float(self.cell_size),
            "cell_height": float(self.cell_height),
        }

    def fingerprint(self) -> str:
        """Stable cache key for the full profile (bake + motion fields)."""
        payload = {
            "name": self.name,
            "radius": self.radius,
            "height": self.height,
            "max_step": self.max_step,
            "max_slope_deg": self.max_slope_deg,
            "locomotion": self.locomotion,
            "step_size": self.step_size,
            "turn_deg": self.turn_deg,
            "max_steps": self.max_steps,
            "success_radius": self.success_radius,
            "cell_size": self.cell_size,
            "cell_height": self.cell_height,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def navmesh_fingerprint(self) -> str:
        """Cache key for the NavMesh bake (only fields Recast consumes).

        Two profiles that differ in step size or episode budget share a bake;
        they must still get separate episode catalogs, which is why
        :meth:`fingerprint` also covers the motion fields.
        """
        payload = {
            "radius": self.radius,
            "height": self.height,
            "max_step": self.max_step,
            "max_slope_deg": self.max_slope_deg,
            "cell_size": self.cell_size,
            "cell_height": self.cell_height,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint()
        data["navmesh_fingerprint"] = self.navmesh_fingerprint()
        data["max_slope"] = round(self.max_slope, 6)
        return data


#: A person: clears a 0.9 m door with a 0.22 m radius, climbs the generated
#: 0.18 m stair risers (0.25 m step budget), uses ramps up to 45 degrees.
HUMANOID = AgentProfile(
    name="humanoid",
    radius=0.22,
    height=1.65,
    max_step=0.25,
    max_slope_deg=45.0,
    locomotion="humanoid",
    step_size=0.25,
    turn_deg=15.0,
    max_steps=500,
    success_radius=0.5,
)

#: A wheelchair / service robot: wider footprint, no stair climbing (4 cm
#: kerb), and a 5 degree ramp limit (roughly the 1:12 accessible ramp).
WHEELED = AgentProfile(
    name="wheeled",
    radius=0.35,
    height=1.20,
    max_step=0.04,
    max_slope_deg=5.0,
    locomotion="wheeled",
    step_size=0.20,
    turn_deg=15.0,
    max_steps=500,
    success_radius=0.5,
)

PROFILES: Dict[str, AgentProfile] = {
    HUMANOID.name: HUMANOID,
    WHEELED.name: WHEELED,
}


def get_profile(name: str) -> AgentProfile:
    """Return a shipped profile by name, or a profile stored as JSON.

    ``name`` may also be a path to a JSON object with the profile fields, which
    lets an evaluation pin a custom agent without touching the package.
    """
    if name in PROFILES:
        return PROFILES[name]
    if name.endswith(".json"):
        with open(name, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        allowed = set(AgentProfile.__dataclass_fields__)
        return AgentProfile(**{key: value for key, value in data.items()
                               if key in allowed})
    raise KeyError(f"unknown agent profile '{name}'; known: {sorted(PROFILES)}")


def available_profiles() -> Dict[str, dict]:
    return {name: profile.to_dict() for name, profile in sorted(PROFILES.items())}
