"""RoomGen PointNav navigation core.

Pure geometry / profiles / metrics import without Habitat-Sim; the baked
NavMesh layer imports it lazily, so the host Python can run the unit tests::

    from roomgen_nav.profiles import HUMANOID
    from roomgen_nav.navmesh import build_composite_navmesh  # needs nav_env

See ``roomgen_nav/eval_pointnav.py`` for the benchmark entry point.
"""

from roomgen_nav.profiles import (  # noqa: F401
    AgentProfile,
    HUMANOID,
    WHEELED,
    PROFILES,
    available_profiles,
    get_profile,
)
from roomgen_nav.pathfinding import (  # noqa: F401
    CompositePath,
    NavLink,
    plan_composite_path,
    validate_composite_path,
)

__all__ = [
    "AgentProfile",
    "HUMANOID",
    "WHEELED",
    "PROFILES",
    "available_profiles",
    "get_profile",
    "CompositePath",
    "NavLink",
    "plan_composite_path",
    "validate_composite_path",
]
