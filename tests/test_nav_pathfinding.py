"""Pure path-planning and validation tests (no Habitat-Sim needed)."""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import spec as spec_mod  # noqa: E402
from roomgen_nav import pathfinding as pf  # noqa: E402
from roomgen_nav.profiles import HUMANOID, WHEELED  # noqa: E402


class FakeFloor:
    """A 1D corridor on one floor with an optional wall in the middle."""

    def __init__(self, floor_id, elevation=0.0, wall_at=None, length=10.0):
        self.floor_id = floor_id
        self.elevation = elevation
        self.wall_at = wall_at
        self.length = length

    def snap_point(self, point):
        x = min(max(float(point[0]), 0.0), self.length)
        return (x, self.elevation, 0.0)

    def find_path(self, start, end):
        if self.wall_at is not None:
            a, b = sorted((float(start[0]), float(end[0])))
            if a < self.wall_at < b:
                return None
        distance = abs(float(end[0]) - float(start[0]))
        return pf.FloorPath([(float(start[0]), self.elevation, 0.0),
                             (float(end[0]), self.elevation, 0.0)], distance)


def _link(link_id="stair_0", from_floor="f0", to_floor="f1", valid=True,
          from_x=8.0, to_x=8.0, kind="stair", rise=2.8):
    track = [(from_x, 0.0, 0.0), (from_x, rise, 0.0)] if to_floor != from_floor else \
        [(from_x, 0.0, 0.0), (to_x, 0.0, 0.0)]
    return pf.NavLink(
        link_id=link_id, kind=kind, from_room="a", to_room="b",
        from_floor=from_floor, to_floor=to_floor,
        from_endpoint=pf.LinkEndpoint(from_floor, track[0], "base"),
        to_endpoint=pf.LinkEndpoint(to_floor, track[-1], "head"),
        track=track, valid=valid,
    )


class CompositePlannerTest(unittest.TestCase):
    def test_cross_floor_route_uses_the_link(self):
        floors = {"f0": FakeFloor("f0", 0.0), "f1": FakeFloor("f1", 2.8)}
        path = pf.plan_composite_path(floors, [_link()], "f0", (1.0, 0.0, 0.0),
                                      "f1", (1.0, 2.8, 0.0))
        self.assertIsNotNone(path)
        self.assertAlmostEqual(path.distance, 7.0 + 2.8 + 7.0, places=5)
        self.assertEqual(path.links_used, ["stair_0"])
        self.assertTrue(path.crosses_floors)
        self.assertAlmostEqual(path.points[0][0], 1.0)
        self.assertAlmostEqual(path.points[-1][0], 1.0)
        self.assertAlmostEqual(path.points[-1][1], 2.8)

    def test_invalid_link_removes_cross_floor_reachability(self):
        floors = {"f0": FakeFloor("f0", 0.0), "f1": FakeFloor("f1", 2.8)}
        path = pf.plan_composite_path(floors, [_link(valid=False)], "f0",
                                      (1.0, 0.0, 0.0), "f1", (1.0, 2.8, 0.0))
        self.assertIsNone(path)

    def test_direct_floor_path_wins_over_a_stair_detour(self):
        floors = {"f0": FakeFloor("f0", 0.0), "f1": FakeFloor("f1", 2.8)}
        path = pf.plan_composite_path(floors, [], "f0", (1.0, 0.0, 0.0),
                                      "f0", (9.0, 0.0, 0.0))
        self.assertIsNotNone(path)
        self.assertAlmostEqual(path.distance, 8.0, places=5)
        self.assertEqual(path.links_used, [])

    def test_same_floor_link_bridges_a_wall(self):
        floors = {"f0": FakeFloor("f0", 0.0, wall_at=5.0)}
        bridge = _link(link_id="ramp_0", from_floor="f0", to_floor="f0",
                       kind="ramp", from_x=4.8, to_x=5.2)
        path = pf.plan_composite_path(floors, [bridge], "f0", (1.0, 0.0, 0.0),
                                      "f0", (9.0, 0.0, 0.0))
        self.assertIsNotNone(path)
        self.assertEqual(path.links_used, ["ramp_0"])
        self.assertAlmostEqual(path.distance, 3.8 + 0.4 + 3.8, places=5)

    def test_unreachable_rooms_return_none(self):
        floors = {"f0": FakeFloor("f0", 0.0, wall_at=5.0)}
        path = pf.plan_composite_path(floors, [], "f0", (1.0, 0.0, 0.0),
                                      "f0", (9.0, 0.0, 0.0))
        self.assertIsNone(path)

    def test_snap_tolerance_rejects_an_offmesh_start(self):
        floors = {"f0": FakeFloor("f0", 0.0)}
        # FakeFloor clamps any x to the corridor, so ask for a far-away point:
        # snap succeeds at 0.0 but the distance is over the tolerance.
        path = pf.plan_composite_path(floors, [], "f0", (-100.0, 0.0, 0.0),
                                      "f0", (9.0, 0.0, 0.0))
        self.assertIsNone(path)


class LinkProfileTest(unittest.TestCase):
    def test_humanoid_climbs_the_generated_riser(self):
        link = _link()
        link.track = [(0.0, 0.0, 0.0), (0.0, 0.18, 0.0), (0.0, 0.36, 0.0)]
        self.assertIsNone(pf.link_profile_problem(link, HUMANOID))
        self.assertIn("riser", pf.link_profile_problem(link, WHEELED))

    def test_ramp_slope_separates_the_profiles(self):
        ramp = _link(link_id="ramp_0", from_floor="f0", to_floor="f0",
                     kind="ramp")
        ramp.track = [(0.0, 0.0, 0.0), (2.0, 0.375, 0.0)]
        self.assertIsNone(pf.link_profile_problem(ramp, HUMANOID))
        self.assertIn("slope", pf.link_profile_problem(ramp, WHEELED))
        gentle = _link(link_id="ramp_1", from_floor="f0", to_floor="f0",
                       kind="ramp")
        gentle.track = [(0.0, 0.0, 0.0), (2.0, 0.15, 0.0)]
        self.assertIsNone(pf.link_profile_problem(gentle, WHEELED))


def _spec_with_void():
    data = {
        "unit": "meter", "wall_height": 2.7, "wall_thickness": 0.1,
        "schema_version": 2,
        "floors": [
            {"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
            {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15},
        ],
        "rooms": [
            {"id": "a", "type": "room", "floor_id": "f0",
             "polygon": [[0, 0], [6, 0], [6, 6], [0, 6]]},
            {"id": "b", "type": "room", "floor_id": "f1",
             "polygon": [[0, 0], [6, 0], [6, 6], [0, 6]]},
        ],
        "cameras": [{"name": "cam0", "room": "a", "pos": [2.0, 2.0, 1.5]}],
        "openings": [],
        "voids": [{"floor": "f0", "outline": [[2, 2], [4, 2], [4, 4], [2, 4]]}],
        "slab_openings": [{"outline": [[2, 2], [4, 2], [4, 4], [2, 4]],
                           "floors": ["f1"]}],
    }
    return spec_mod.parse_spec(data)


class PathValidationTest(unittest.TestCase):
    def test_a_path_across_a_void_is_reported(self):
        spec = _spec_with_void()
        path = pf.CompositePath(
            points=[(1.0, 2.86, -3.0), (5.0, 2.86, -3.0)],
            distance=4.0,
            segments=[pf.PathSegment("floor", "f1",
                                     [(1.0, 2.86, -3.0), (5.0, 2.86, -3.0)], 4.0)])
        violations = pf.validate_composite_path(spec, path, HUMANOID)
        self.assertIn("crosses_void", [item.code for item in violations])

    def test_a_split_level_jump_is_reported(self):
        spec = _spec_with_void()
        path = pf.CompositePath(
            points=[(1.0, 0.0, -1.0), (1.0, 0.5, -1.0)],
            distance=0.5,
            segments=[pf.PathSegment("floor", "f0",
                                     [(1.0, 0.0, -1.0), (1.0, 0.5, -1.0)], 0.5)])
        violations = pf.validate_composite_path(spec, path, HUMANOID)
        self.assertIn("step_exceeds_max_step", [item.code for item in violations])

    def test_a_normal_path_has_no_violation(self):
        spec = _spec_with_void()
        path = pf.CompositePath(
            points=[(1.0, 0.02, -1.0), (5.0, 0.02, -1.0)],
            distance=4.0,
            segments=[pf.PathSegment("floor", "f0",
                                     [(1.0, 0.02, -1.0), (5.0, 0.02, -1.0)], 4.0)])
        self.assertEqual(pf.validate_composite_path(spec, path, HUMANOID), [])


class GeometryTest(unittest.TestCase):
    def test_heading_and_turns(self):
        self.assertAlmostEqual(pf.heading_deg((0, 0, 0), (1, 0, 0)), 0.0)
        self.assertAlmostEqual(pf.heading_deg((0, 0, 0), (0, 0, -1)), 90.0)
        self.assertAlmostEqual(pf.angle_delta_deg(10, 350), 20.0)
        dx, dz = pf.forward_vector(90.0)
        self.assertAlmostEqual(dx, 0.0, places=6)
        self.assertAlmostEqual(dz, -1.0, places=6)

    def test_densify_keeps_the_endpoints_and_bounds_segments(self):
        dense = pf.densify_polyline([(0, 0, 0), (1, 0, 0)], max_spacing=0.2)
        self.assertEqual(dense[0], (0.0, 0.0, 0.0))
        self.assertEqual(dense[-1], (1.0, 0.0, 0.0))
        self.assertTrue(all(pf.point_distance(dense[i], dense[i + 1]) <= 0.2 + 1e-9
                            for i in range(len(dense) - 1)))

    def test_polygon_sampling_is_inside_and_deterministic(self):
        polygon = [[0, 0], [4, 0], [4, 4], [0, 4]]
        a = pf.sample_point_in_polygon(random.Random(7), polygon, margin=0.5)
        b = pf.sample_point_in_polygon(random.Random(7), polygon, margin=0.5)
        self.assertEqual(a, b)
        self.assertGreaterEqual(pf.distance_to_polygon_boundary(a, polygon), 0.5)

    def test_segment_crosses_a_polygon(self):
        square = [[2, 2], [4, 2], [4, 4], [2, 4]]
        # The helper reads (x, z) from a 3D point (Y-up); this segment enters
        # the square from below (z=1) and exits above it (z=5).
        self.assertTrue(pf.segment_crosses_polygon_2d((3, 0, 1), (3, 0, 5), square))
        # A segment at z=-5 stays outside the square.
        self.assertFalse(pf.segment_crosses_polygon_2d((0, 0, -5), (6, 0, -5), square))


if __name__ == "__main__":
    unittest.main()
