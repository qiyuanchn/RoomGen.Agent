"""Diversity metrics: the room-instance graph, union areas and camera spreads.

Run::

    python tests/test_diversity.py            # or: python tests/test_diversity.py -v

The second review found `tools/diversity_report.py` reducing every connection to
a room *type* pair before hashing and only reading `openings`: a chain, a star
and a four-room ring of the same room type shared one fingerprint
(`b7272cbca310`), every ring reported zero cycles, and a connectors-only duplex
fixture with seven connections reported an empty edge set.  `area_m2` was the
bounding rectangle and camera spacing ignored the storey.

These tests pin the three counter-examples and the properties the metrics now
promise: room instances keep their function and storey, edges keep their
connection kind, the cycle rank is `E - V + C` of that graph, the fingerprint is
canonical (renaming rooms, reordering any list or feeding the nodes in another
order keeps it; chain/star/ring differ; storey count and connection kinds are
part of it), areas are unions of the outlines instead of bounding rectangles,
and camera gaps come in 3D / within-storey / navigation flavours.  The hash only
groups candidates -- an artificially colliding hash must still leave different
fingerprints in different groups.

The chain, star and ring specs are built with their opening centres on the
shared wall, and the two-storey cases use `tests/fixtures/two_floor_demo.json`,
so the positive cases go through `blender_agent.spec` exactly like a generated
spec does; the tolerant raw reader is exercised separately (malformed specs and
a reader-agreement test).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, REPO_ROOT)

from tools import diversity_report as dr  # noqa: E402

TOOL = os.path.join(REPO_ROOT, "tools", "diversity_report.py")
TWO_FLOOR_FIXTURE = os.path.join(TESTS_DIR, "fixtures", "two_floor_demo.json")

FLOOR_0 = {"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15}
FLOOR_1 = {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15}


def rect(x0: float, y0: float, x1: float, y1: float):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def wall_center(first, second):
    """Midpoint of the wall two touching rectangles share."""
    (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = first, second
    if abs(ax1 - bx0) < 1e-9 or abs(bx1 - ax0) < 1e-9:
        x = ax1 if abs(ax1 - bx0) < 1e-9 else ax0
        return [x, (max(ay0, by0) + min(ay1, by1)) / 2.0]
    y = ay1 if abs(ay1 - by0) < 1e-9 else ay0
    return [(max(ax0, bx0) + min(ax1, bx1)) / 2.0, y]


def scene_spec(rooms, connections, kinds=None, floors=None, cameras=None, furniture=None,
               connectors=False, room_types=None):
    """One spec from rooms ``(id, bounds)`` / ``(id, bounds, floor_id)`` and links.

    Every connection's centre lands on the shared wall of the two rooms, so the
    spec is contract-valid and the report reads it through `blender_agent.spec`;
    `connectors=True` writes schema version 2 with a `connectors` list, the other
    mode writes the legacy `openings`.
    """
    bounds = {room[0]: room[1] for room in rooms}
    floor_of = {room[0]: (room[2] if len(room) > 2 else None) for room in rooms}
    room_list = []
    for room_id, box in [(room[0], room[1]) for room in rooms]:
        entry = {"id": room_id, "type": (room_types or {}).get(room_id, "bedroom"),
                 "polygon": rect(*box)}
        if floor_of[room_id] or floors:
            entry["floor_id"] = floor_of[room_id] or floors[0]["id"]
        room_list.append(entry)
    spec = {
        "wall_height": floors[0]["height"] if floors else 2.7,
        "wall_thickness": 0.12,
        "rooms": room_list,
        "windows": [],
        "cameras": cameras or [{"name": "0000", "room": rooms[0][0], "pos": [1.0, 1.0, 1.5]}],
        "style": {"sun_azimuth_deg": 210.0, "sun_elevation_deg": 35.0, "exposure": 0.0},
    }
    if connectors:
        spec["schema_version"] = 2
        spec["floors"] = floors
        spec["connectors"] = [
            {"kind": (kinds or {}).get(link, "door"), "from_room": link[0], "to_room": link[1],
             "from_floor": floor_of[link[0]] or floors[0]["id"],
             "to_floor": floor_of[link[1]] or floors[0]["id"],
             "center": wall_center(bounds[link[0]], bounds[link[1]])}
            for link in connections]
    else:
        spec["openings"] = [
            {"between": list(link), "center": wall_center(bounds[link[0]], bounds[link[1]]),
             "width": 0.9, "height": 2.05, "kind": (kinds or {}).get(link, "door")}
            for link in connections]
    if furniture is not None:
        spec["furniture"] = furniture
    return spec


#: four identical rooms, three different connections -- the review's counter-example
ROW_LAYOUT = [(f"r{index}", (3.0 * index, 0.0, 3.0 * index + 3.0, 3.0)) for index in range(4)]
RING_LAYOUT = [("r0", (0.0, 0.0, 3.0, 3.0)), ("r1", (3.0, 0.0, 6.0, 3.0)),
               ("r2", (3.0, 3.0, 6.0, 6.0)), ("r3", (0.0, 3.0, 3.0, 6.0))]
STAR_LAYOUT = [("hub", (0.0, 0.0, 3.0, 9.0)), ("s0", (3.0, 0.0, 6.0, 3.0)),
               ("s1", (3.0, 3.0, 6.0, 6.0)), ("s2", (3.0, 6.0, 6.0, 9.0))]
CHAIN = [("r0", "r1"), ("r1", "r2"), ("r2", "r3")]
RING = CHAIN + [("r3", "r0")]
STAR = [("hub", "s0"), ("hub", "s1"), ("hub", "s2")]

TOPOLOGIES = (("chain", ROW_LAYOUT, CHAIN), ("star", STAR_LAYOUT, STAR),
              ("ring", RING_LAYOUT, RING))


def chain_spec(**overrides):
    return scene_spec(ROW_LAYOUT, CHAIN, **overrides)


class TopologyFingerprintTest(unittest.TestCase):
    """The review's first two counter-examples: one hash and zero cycles for all."""

    def test_the_positive_specs_go_through_the_contract(self):
        for name, layout, connections in TOPOLOGIES:
            spec = scene_spec(layout, connections)
            self.assertEqual(dr.scene_graph(spec)["source"], "spec", name)

    def test_chain_star_ring_are_three_topologies(self):
        keys, hashes, cycles, edges = {}, {}, {}, {}
        for name, layout, connections in TOPOLOGIES:
            metrics = dr.scene_metrics(scene_spec(layout, connections))
            keys[name] = metrics["structure_key"]
            hashes[name] = metrics["structure_hash"]
            cycles[name] = metrics["structure"]["cycles"]
            edges[name] = metrics["connections"]["count"]
        self.assertEqual(len(set(keys.values())), 3, keys)
        self.assertEqual(len(set(hashes.values())), 3, hashes)
        # E - V + C on the room graph: only the ring closes a loop.
        self.assertEqual(cycles, {"chain": 0, "star": 0, "ring": 1})
        self.assertEqual(edges, {"chain": 3, "star": 3, "ring": 4})

    def test_edge_kind_is_part_of_the_fingerprint(self):
        doors = dr.scene_metrics(scene_spec(RING_LAYOUT, RING))
        passages = dr.scene_metrics(scene_spec(
            RING_LAYOUT, RING, kinds={link: "open_passage" for link in RING}))
        self.assertEqual(passages["structure"]["edge_kinds"], {"open_passage": 4})
        self.assertNotEqual(passages["structure_key"], doors["structure_key"])
        # Stair and ramp are the vertical kinds; no valid geometry is needed to
        # assert that the fingerprint sees the difference.
        attrs = [("hall", 0), ("landing", 1)]
        _, exact = dr.canonical_graph_key(attrs, [(0, 1, "stair")], 2)
        self.assertTrue(exact)
        self.assertNotEqual(dr.canonical_graph_key(attrs, [(0, 1, "stair")], 2)[0],
                            dr.canonical_graph_key(attrs, [(0, 1, "ramp")], 2)[0])

    def test_fingerprint_ignores_names_order_furniture_and_size(self):
        metrics = dr.scene_metrics(scene_spec(RING_LAYOUT, RING))
        renaming = {f"r{index}": name for index, name in enumerate(
            ["entry_hall", "big_room", "sleep_spot", "wash"])}
        renamed = scene_spec([(renaming[room_id], box) for room_id, box in RING_LAYOUT],
                             [(renaming[first], renaming[second]) for first, second in RING])
        self.assertEqual(dr.scene_metrics(renamed)["structure_key"], metrics["structure_key"])

        reordered = scene_spec(RING_LAYOUT, RING)
        reordered["rooms"] = list(reversed(reordered["rooms"]))
        reordered["openings"] = list(reversed(reordered["openings"]))
        self.assertEqual(dr.scene_metrics(reordered)["structure_key"], metrics["structure_key"])

        scaled = scene_spec([(room_id, tuple(value * 1.5 for value in box))
                             for room_id, box in RING_LAYOUT], RING)
        self.assertAlmostEqual(dr.scene_metrics(scaled)["shape"]["bbox_area_m2"], 6.0 * 6.0 * 2.25,
                               places=1)
        self.assertEqual(dr.scene_metrics(scaled)["structure_key"], metrics["structure_key"])

        furnished = scene_spec(RING_LAYOUT, RING,
                               furniture=[{"room": "r0", "asset": "builtin:bed"}])
        other = dr.scene_metrics(furnished)
        self.assertEqual(other["structure_key"], metrics["structure_key"])
        self.assertNotEqual(other["furniture"]["combo_hash"], metrics["furniture"]["combo_hash"])

    def test_canonical_key_is_node_order_invariant(self):
        attrs = [("bedroom", 0)] * 4
        edges = [(0, 1, "door"), (1, 2, "door"), (2, 3, "door"), (3, 0, "door")]
        baseline, exact = dr.canonical_graph_key(attrs, edges, 1)
        self.assertTrue(exact)
        rotation = {0: 3, 1: 0, 2: 1, 3: 2}
        shuffled_edges = [(rotation[a], rotation[b], kind) for a, b, kind in reversed(edges)]
        rotated, exact = dr.canonical_graph_key(
            [attrs[rotation[node]] for node in range(4)], shuffled_edges, 1)
        self.assertTrue(exact)
        self.assertEqual(baseline, rotated)
        # The storey count is part of the structural expression.
        self.assertNotEqual(baseline, dr.canonical_graph_key(attrs, edges, 2)[0])

    def test_storey_count_enters_the_metrics(self):
        """Built raw: the contract rejects a declared storey without rooms."""
        one = scene_spec(ROW_LAYOUT, CHAIN)
        two = scene_spec(ROW_LAYOUT, CHAIN)
        two["schema_version"] = 2
        two["floors"] = [FLOOR_0, FLOOR_1]
        with mock.patch.object(dr, "_spec_module", lambda: None):
            self.assertEqual(dr.scene_graph(one)["source"], "raw")
            self.assertEqual(dr.scene_metrics(two)["floors"], 2)
            self.assertNotEqual(dr.scene_metrics(one)["structure_key"],
                                dr.scene_metrics(two)["structure_key"])


class ConnectionSourceTest(unittest.TestCase):
    """The third counter-example: connectors only, edges empty."""

    def test_connectors_only_duplex_reports_its_edges(self):
        with open(TWO_FLOOR_FIXTURE, encoding="utf-8") as handle:
            spec = json.load(handle)
        metrics = dr.scene_metrics(spec)
        self.assertEqual(spec.get("openings"), None)
        self.assertEqual(metrics["connections"]["count"], 7)
        self.assertEqual(metrics["connections"]["kinds"],
                         {"door": 5, "open_passage": 1, "stair": 1})
        self.assertEqual(metrics["connections"]["cross_floor"], 1)
        self.assertEqual(metrics["floors"], 2)
        self.assertEqual(len(metrics["structure"]["edges"]), 7)
        self.assertEqual(sorted(metrics["structure"]["edge_kinds"]),
                         ["door", "open_passage", "stair"])
        # 8 rooms, 7 connections, one component -> no loop; the stair is an edge.
        self.assertEqual(metrics["structure"]["cycles"], 0)
        self.assertTrue(any(edge["kind"] == "stair" for edge in metrics["structure"]["edges"]))
        self.assertTrue(metrics["structure"]["exact"])

    def test_legacy_openings_and_connectors_agree(self):
        legacy = scene_spec(RING_LAYOUT, RING)
        modern = scene_spec([(room_id, box, FLOOR_0["id"]) for room_id, box in RING_LAYOUT],
                            RING, floors=[FLOOR_0], connectors=True)
        self.assertEqual(dr.scene_graph(legacy)["source"], "spec")
        self.assertEqual(dr.scene_graph(modern)["source"], "spec")
        self.assertEqual(dr.scene_metrics(legacy)["structure_key"],
                         dr.scene_metrics(modern)["structure_key"])

    def test_both_readers_build_the_same_graph(self):
        with open(TWO_FLOOR_FIXTURE, encoding="utf-8") as handle:
            fixture = json.load(handle)
        specs = [scene_spec(layout, connections) for _name, layout, connections in TOPOLOGIES]
        specs.append(fixture)
        specs.append(scene_spec([(room_id, box, FLOOR_0["id"]) for room_id, box in RING_LAYOUT],
                                RING, floors=[FLOOR_0], connectors=True))
        for spec in specs:
            parsed = dr.scene_metrics(spec)
            with mock.patch.object(dr, "_spec_module", lambda: None):
                self.assertEqual(dr.scene_graph(spec)["source"], "raw")
                raw = dr.scene_metrics(spec)
            self.assertEqual(parsed["structure_key"], raw["structure_key"])
            self.assertEqual(parsed["connections"], raw["connections"])
            self.assertEqual(parsed["shape"], raw["shape"])

    def test_unparseable_spec_falls_back_to_raw_reading(self):
        # An unknown room reference and a zero storey height: blender_agent.spec
        # refuses the spec, the tolerant reader must not.
        spec = chain_spec()
        spec["wall_height"] = 0.0
        spec["openings"].append({"between": ["r0", "nope"], "center": [0.0, 0.0],
                                 "width": 0.9, "height": 2.05})
        graph = dr.scene_graph(spec)
        self.assertEqual(graph["source"], "raw")
        metrics = dr.scene_metrics(spec)
        self.assertEqual(metrics["connections"]["count"], 3)
        self.assertEqual(metrics["structure"]["cycles"], 0)

        # A cross-floor stair declared in `connectors` survives the same fallback.
        hall = [("hall", (0.0, 0.0, 3.0, 6.0), "f0"), ("landing", (1.0, 7.0, 4.0, 13.0), "f1")]
        duplex = scene_spec(hall, [], floors=[FLOOR_0, FLOOR_1], connectors=True,
                            room_types={"hall": "hall", "landing": "hall"})
        duplex["wall_height"] = 0.0
        duplex["connectors"] = [{"kind": "stair", "from_room": "hall", "to_room": "landing",
                                 "from_floor": "f0", "to_floor": "f1"}]
        metrics = dr.scene_metrics(duplex)
        self.assertEqual(metrics["connections"]["kinds"], {"stair": 1})
        self.assertEqual(metrics["connections"]["cross_floor"], 1)
        self.assertEqual(metrics["floors"], 2)

    def test_malformed_blocks_do_not_sink_the_scene(self):
        spec = chain_spec()
        spec.pop("wall_height")
        spec["openings"] = {"not": "a list"}
        spec["connectors"] = "nope"
        spec["furniture"] = {"room": "r0", "asset": "builtin:bed"}
        spec["windows"] = 7
        spec["style"] = "solarized"
        spec["cameras"] = [{"name": "0000", "room": "r0", "pos": [1.0, 1.0]}]
        metrics = dr.scene_metrics(spec)
        self.assertEqual(metrics["connections"]["count"], 0)
        self.assertEqual(metrics["furniture"]["count"], 0)
        self.assertEqual(metrics["lighting"]["windows"], 0)
        # The camera carried no z: it stands at eye height above its storey.
        self.assertAlmostEqual(metrics["cameras"]["count"], 1.0)
        self.assertEqual(metrics["structure"]["cycles"], 0)

    def test_load_specs_skips_a_non_object_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "spec.json"), "w", encoding="utf-8") as handle:
                json.dump(["not", "a", "spec"], handle)
            self.assertEqual(dr.load_specs([tmpdir]), [])


class ShapeAreaTest(unittest.TestCase):
    """``area_m2`` used to be the XY bounding rectangle of every room outline."""

    def test_footprint_is_a_union_not_a_bounding_box(self):
        rooms = [("living", (0.0, 0.0, 4.0, 4.0)), ("kitchen", (0.0, 4.0, 2.0, 6.0))]
        metrics = dr.scene_metrics(scene_spec(rooms, [],
                                             room_types={"living": "living_room",
                                                         "kitchen": "kitchen"}))
        shape = metrics["shape"]
        self.assertAlmostEqual(shape["footprint_area_m2"], 20.0, places=3)
        self.assertAlmostEqual(shape["bbox_area_m2"], 24.0, places=3)
        self.assertLess(shape["footprint_area_m2"], shape["bbox_area_m2"])
        self.assertAlmostEqual(shape["footprint_ratio"], 20.0 / 24.0, places=3)
        self.assertAlmostEqual(shape["total_floor_area_m2"], 20.0, places=3)

    def test_polygon_union_area_counts_an_overlap_once(self):
        overlapping = [rect(0.0, 0.0, 2.0, 2.0), rect(1.0, 1.0, 3.0, 3.0)]
        self.assertAlmostEqual(dr.polygon_union_area(overlapping), 7.0, places=6)
        self.assertAlmostEqual(dr.polygon_union_area([]), 0.0, places=6)

    def test_stacked_storeys_and_per_storey_areas(self):
        with open(TWO_FLOOR_FIXTURE, encoding="utf-8") as handle:
            metrics = dr.scene_metrics(json.load(handle))
        shape = metrics["shape"]
        self.assertAlmostEqual(shape["footprint_area_m2"], 48.0, places=2)
        self.assertEqual(sorted(shape["floor_areas_m2"]), ["f0", "f1"])
        self.assertAlmostEqual(shape["floor_areas_m2"]["f0"], 48.0, places=2)
        self.assertAlmostEqual(shape["floor_areas_m2"]["f1"], 48.0, places=2)
        # Two stacked storeys: the footprint is one storey, the floor area is both.
        self.assertAlmostEqual(shape["total_floor_area_m2"], 96.0, places=2)
        self.assertGreater(shape["total_floor_area_m2"], shape["footprint_area_m2"])


class CameraSpreadTest(unittest.TestCase):
    """Camera spacing used to be XY only, so two storeys at one XY were 0 m apart."""

    def test_same_xy_on_two_storeys_is_not_a_zero_gap(self):
        with open(TWO_FLOOR_FIXTURE, encoding="utf-8") as handle:
            spec = json.load(handle)
        metrics = dr.scene_metrics(spec)
        cameras = metrics["cameras"]
        # The fixture really does carry that pair: hall on f0 and landing on f1.
        positions = [(camera["room"], camera["pos"]) for camera in spec["cameras"]]
        hall = next(pos for room, pos in positions if room == "hall")
        landing = next(pos for room, pos in positions if room == "landing")
        self.assertAlmostEqual(hall[0], landing[0], places=6)
        self.assertAlmostEqual(hall[1], landing[1], places=6)
        xy_gap = min(
            min(((first[1][0] - second[1][0]) ** 2 + (first[1][1] - second[1][1]) ** 2) ** 0.5
                for second in positions if second is not first)
            for first in positions)
        self.assertAlmostEqual(xy_gap, 0.0, places=6)          # what the old metric said
        self.assertGreater(cameras["min_gap_3d_m"], 0.0)       # what it says now
        self.assertGreater(cameras["min_gap_same_floor_m"], 0.0)
        self.assertGreaterEqual(cameras["min_gap_nav_m"], cameras["min_gap_3d_m"])
        self.assertEqual(cameras["unreachable_pairs"], 0)
        self.assertEqual(cameras["floor_counts"], {"f0": 4, "f1": 4})

    def test_cameras_on_unconnected_storeys_are_reported_unreachable(self):
        rooms = [("a", (0.0, 0.0, 3.0, 3.0), "f0"), ("b", (3.0, 0.0, 6.0, 3.0), "f0"),
                 ("c", (0.0, 0.0, 3.0, 3.0), "f1")]
        spec = scene_spec(rooms, [("a", "b")], floors=[FLOOR_0, FLOOR_1], connectors=True,
                          room_types={"a": "bedroom", "b": "bedroom", "c": "study"},
                          cameras=[{"name": "0000", "room": "a", "pos": [1.0, 1.0, 1.5]},
                                   {"name": "0001", "room": "c", "pos": [1.0, 1.0, 4.35]}])
        self.assertEqual(dr.scene_graph(spec)["source"], "spec")
        metrics = dr.scene_metrics(spec)
        self.assertEqual(metrics["cameras"]["unreachable_pairs"], 1)
        self.assertAlmostEqual(metrics["cameras"]["min_gap_3d_m"], 2.85, places=3)
        self.assertIsNone(metrics["cameras"]["min_gap_nav_m"])


class SummaryTest(unittest.TestCase):
    """Aggregation: repeats are exact fingerprint matches, hashes only group."""

    def metrics(self):
        return [dr.scene_metrics(scene_spec(layout, connections))
                for _name, layout, connections in TOPOLOGIES] + \
               [dr.scene_metrics(scene_spec(RING_LAYOUT, RING))]

    def test_duplicate_rate_counts_topologies_not_scenes(self):
        summary = dr.diversity_summary(self.metrics())
        structures = summary["structures"]
        self.assertEqual(structures["unique"], 3)
        self.assertEqual(structures["duplicate_rate"], 0.25)
        self.assertEqual(structures["repeat_scenes"], 2)
        self.assertEqual(structures["groups"][0]["count"], 2)
        self.assertEqual(summary["cycles"]["histogram"], {"0": 2, "1": 2})
        self.assertEqual(summary["cycles"]["loops"], 2)
        self.assertEqual(summary["connections"]["total"], 3 + 3 + 4 + 4)
        self.assertEqual(summary["connections"]["scenes_without_connection"], 0)
        self.assertEqual(structures["hash_collisions"], 0)

    def test_hash_only_groups_candidates(self):
        metrics = self.metrics()
        with mock.patch.object(dr, "_hash", lambda payload: "000000000000"):
            groups, collisions = dr._exact_groups(metrics, "structure_key")
        self.assertEqual(len(groups), 3)
        self.assertEqual(collisions, 1)
        self.assertEqual(sorted(len(value) for value in groups.values()), [1, 1, 2])

    def test_sample_prefers_a_new_structure(self):
        metrics = self.metrics()
        for index, entry in enumerate(metrics):
            entry["seed"] = index
            entry["family"] = "row" if index < 3 else "block"
        sample = dr.choose_sample(metrics, 4)
        self.assertEqual(sample["seeds"], [0, 1, 2, 3])
        self.assertEqual(sample["reasons"]["1"], "new topology structure")


# --------------------------------------------------------------------------- #
# final products: request vs build vs delivery, fallbacks and bias
# --------------------------------------------------------------------------- #
BENCHMARK_ROOTS = os.path.join(REPO_ROOT, "outputs", "benchmark100")
#: arrangements that are not a topology *family*: layout_gen names them in
#: ``report["shape"]`` and leaves ``report["family"]`` empty
PLAIN_SHAPES = ("row", "block", "duplex")
ALL_OK = {"layout_ok": True, "render_ok": True, "validation_ok": True, "bundle_ok": True}


def stats_block(**overrides) -> dict:
    """A layout_report stats block, with the fields layout_gen really writes."""
    block = {"seed": 0, "attempt": 0, "attempts_used": 1, "floors": 1}
    block.update(overrides)
    return block


def forecast_block(**rooms) -> dict:
    """``door_visual_forecast``: per camera name, what it sees."""
    return {name: {"room": room, "coverage": {"visible_area_fraction": value},
                   "visible_fraction": 1.0}
            for name, (room, value) in rooms.items()}


def write_scene(root: str, seed: int, spec: dict, family: str, stats: dict = None,
                ok: dict = None, failure: dict = None, forecast: dict = None,
                review: dict = None, validation: dict = None) -> dict:
    """One scene directory of a synthetic batch, plus the batch row describing it."""
    scene = os.path.join(root, "scene_%04d" % seed)
    os.makedirs(scene, exist_ok=True)
    block = stats_block(seed=seed)
    block.update(stats or {})
    report = {"shape": family, "family": family if family not in PLAIN_SHAPES else None,
              "stats": block}
    if forecast is not None:
        report["door_visual_forecast"] = forecast
    levels = dict(ALL_OK if ok is None else ok)
    with open(os.path.join(scene, "spec.json"), "w", encoding="utf-8") as handle:
        json.dump(spec, handle)
    with open(os.path.join(scene, "layout_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    summary = {"kind": "layout_gen_run", "seed": seed, "stats": block, "ok": levels,
               "failure": failure, "render_requested": True,
               "stages": {"layout": "ok" if levels.get("layout_ok") else "failed",
                          "render": "ok" if levels.get("render_ok") else "failed",
                          "validation": "ok" if levels.get("validation_ok") else "failed"}}
    if review is not None:
        summary["review"] = review
    if validation is not None:
        summary["validation"] = validation
    with open(os.path.join(scene, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle)
    return {"seed": seed, "dir": "scene_%04d" % seed,
            "status": "bundle_ok" if levels.get("bundle_ok") else "failed",
            "ok": levels, "failure": failure, "resumed": False}


def write_root(root: str, rows, request: dict = None, render_requested: bool = True,
               review_requested: bool = True) -> None:
    summary = {"kind": "layout_gen_batch", "schema_version": 2, "batch_id": "test",
               "render_requested": render_requested, "review_requested": review_requested,
               "request": request, "levels": None, "scenes": rows}
    with open(os.path.join(root, "batch_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle)


def product_report(root: str, **kwargs):
    blocks = dr._set_role([dr.read_root(root, replay_draw=kwargs.pop("replay", True))],
                          kwargs.pop("product_root", None))
    return dr.success_payload(blocks, scenes=kwargs.pop("scenes", False))


class SuccessMixTest(unittest.TestCase):
    """The review's point: the delivered mix is not the requested mix."""

    def test_request_build_and_delivery_are_counted_apart(self):
        with tempfile.TemporaryDirectory() as root:
            # four scenes ask for u_shape / l_shape and two of them land on row
            plan = [(0, "u_shape", "row", {"auto_pick": [5, "u_shape"], "auto_fallback": True,
                                           "attempts_used": 17}),
                    (1, "u_shape", "u_shape", {"auto_pick": [5, "u_shape"],
                                               "auto_fallback": False, "attempts_used": 1}),
                    (2, "l_shape", "l_shape", {"auto_pick": [5, "l_shape"],
                                               "auto_fallback": False, "attempts_used": 2}),
                    (3, "l_shape", "row", {"auto_pick": [5, "l_shape"], "auto_fallback": True,
                                           "attempts_used": 18})]
            rows = [write_scene(root, seed, scene_spec(ROW_LAYOUT, CHAIN), family, stats=stats)
                    for seed, _drawn, family, stats in plan]
            write_root(root, rows, request={"layout": {"shape": "auto"}})
            report = product_report(root)

        requested = report["requested"]["generation"]
        self.assertEqual(requested["basis"], {"recorded": 4})
        self.assertEqual(requested["share"], {"l_shape": 0.5, "u_shape": 0.5})
        self.assertEqual(report["generated"]["family_share"], {"l_shape": 0.25, "row": 0.5,
                                                              "u_shape": 0.25})
        self.assertEqual(report["generated"]["count"], 4)
        self.assertEqual(report["success"]["count"], 4)
        # Half the request never reached a product: tv = |row .5| + |u/l -.25| => .5
        self.assertEqual(report["bias"]["generation"]["tv_distance"], 0.5)
        notes = " | ".join(report["bias"]["generation"]["notes"])
        self.assertIn("'row' grows", notes)
        survival = {row["family"]: row["survival"]
                    for row in report["bias"]["generation"]["families"]}
        self.assertEqual(survival, {"l_shape": 0.5, "row": None, "u_shape": 0.5})
        # 'row' was never asked for and built twice: the count of a family in
        # the built set is not its survival, so the two are reported apart
        rows = {row["family"]: row for row in report["bias"]["generation"]["families"]}
        self.assertEqual((rows["row"]["asked"], rows["row"]["built"]), (0, 2))
        self.assertEqual((rows["u_shape"]["asked"], rows["u_shape"]["kept"]), (2, 1))

    def test_fallbacks_are_measured_apart(self):
        with tempfile.TemporaryDirectory() as root:
            # seed 0 asked for u_shape, gave up after the sticky attempts and
            # landed on row; the other two built what they drew
            plan = [(0, "row", {"auto_pick": [5, "u_shape"], "auto_fallback": True,
                                "attempts_used": 17}),
                    (1, "u_shape", {"auto_pick": [5, "u_shape"], "auto_fallback": False,
                                    "attempts_used": 1}),
                    (2, "l_shape", {"auto_pick": [5, "l_shape"], "auto_fallback": False,
                                    "attempts_used": 3})]
            rows = [write_scene(root, seed, scene_spec(ROW_LAYOUT, CHAIN), family, stats=stats)
                    for seed, family, stats in plan]
            write_root(root, rows, request={"layout": {"shape": "auto"}})
            report = product_report(root)

        fallback = report["success"]["fallback"]
        self.assertEqual(fallback["attempt_histogram"], {"1": 1, "17": 1, "3": 1})
        self.assertEqual(fallback["first_attempt_rate"], 0.3333)
        self.assertEqual(fallback["retry_rate"], 0.6667)
        self.assertEqual(fallback["auto_fallback"], {"known": 3, "true": 1, "rate": 0.3333,
                                                     "unknown": 0})
        switch = fallback["family_switch"]
        self.assertEqual(switch["known"], 3)
        self.assertEqual(switch["switched"], 1)
        self.assertEqual(switch["pairs"], {"u_shape->row": 1})
        self.assertEqual(switch["by_basis"]["recorded"]["known"], 3)

    def test_the_pipeline_filters_the_success_set(self):
        """A product that failed its review must not count as a delivered house."""
        with tempfile.TemporaryDirectory() as root:
            rows = [
                write_scene(root, 0, scene_spec(ROW_LAYOUT, CHAIN), "row",
                            stats=stats_block(auto_pick=[4, "row"])),
                write_scene(root, 1, scene_spec(RING_LAYOUT, RING), "loop",
                            stats=stats_block(auto_pick=[4, "loop"])),
                write_scene(root, 2, scene_spec(STAR_LAYOUT, STAR), "u_shape",
                            stats=stats_block(auto_pick=[4, "u_shape"]),
                            ok=dict(ALL_OK, bundle_ok=False, review_ok=False),
                            failure={"stage": "review", "kind": "review_failed",
                                     "detail": "exposure"}),
            ]
            write_root(root, rows, request={"layout": {"shape": "auto"}})
            report = product_report(root)

        self.assertEqual(report["success"]["count"], 2)
        self.assertEqual(report["generated"]["count"], 3)
        self.assertEqual(report["filtered"]["dropped"], 1)
        self.assertEqual(report["filtered"]["drop_rate"], 0.3333)
        self.assertEqual(report["filtered"]["reasons"], {"review:review_failed": 1})
        self.assertEqual(report["filtered"]["dropped_families"], {"u_shape": 1})
        self.assertEqual(report["filtered"]["seeds"], [2])
        self.assertNotIn("u_shape", report["success"]["family_share"])
        self.assertIn("u_shape", report["generated"]["family_share"])
        notes = " | ".join(report["bias"]["product"]["notes"])
        self.assertIn("never delivered: u_shape", notes)

    def test_a_legacy_artifact_is_replayed_not_invented(self):
        """No stats.auto_pick: the drawn family is re-drawn, and reported as such."""
        with tempfile.TemporaryDirectory() as root:
            scene = os.path.join(root, "scene_0007")
            os.makedirs(scene)
            with open(os.path.join(scene, "spec.json"), "w", encoding="utf-8") as handle:
                json.dump(scene_spec(ROW_LAYOUT, CHAIN), handle)
            # exactly what a report from before the auto_pick contract holds:
            # stats.shape is the *requested* shape, report.shape the built one
            with open(os.path.join(scene, "layout_report.json"), "w", encoding="utf-8") as handle:
                json.dump({"shape": "row", "family": None,
                           "stats": stats_block(seed=7, shape="auto", attempts_used=2)}, handle)
            write_root(root, [{"seed": 7, "dir": "scene_0007", "status": "bundle_ok",
                               "ok": dict(ALL_OK), "failure": None, "resumed": False}],
                       request={"layout": {"shape": "auto"}})
            report = product_report(root, scenes=True)

        record = report["scenes"]["success"][0]
        self.assertEqual(record["family"], "row")
        self.assertEqual(record["requested_shape"], "auto")
        self.assertEqual(record["requested_family_basis"], "replayed")
        self.assertEqual(record["requested_family"], dr.draw_family(7)["family"])
        fallback = report["success"]["fallback"]
        # Nothing is known about this artifact's fallback -- and it says so.
        self.assertEqual(fallback["auto_fallback"]["known"], 0)
        self.assertEqual(fallback["auto_fallback"]["unknown"], 1)
        self.assertEqual(fallback["family_switch"]["known"], 0)
        self.assertEqual(fallback["family_switch"]["by_basis"]["replayed"]["known"], 1)
        self.assertEqual(report["requested"]["generation"]["basis"], {"replayed": 1})

        with tempfile.TemporaryDirectory() as root:
            write_root(root, [{"seed": 7, "dir": "scene_0007", "status": "bundle_ok",
                               "ok": dict(ALL_OK), "failure": None, "resumed": False}])
            os.makedirs(os.path.join(root, "scene_0007"))
            with open(os.path.join(root, "scene_0007", "spec.json"), "w", encoding="utf-8") as h:
                json.dump(scene_spec(ROW_LAYOUT, CHAIN), h)
            with open(os.path.join(root, "scene_0007", "layout_report.json"), "w",
                      encoding="utf-8") as h:
                json.dump({"shape": "row", "family": None,
                           "stats": stats_block(seed=7, shape="auto")}, h)
            report = product_report(root, replay=False, scenes=True)
        self.assertIsNone(report["scenes"]["success"][0]["requested_family"])
        self.assertEqual(report["scenes"]["success"][0]["requested_family_basis"], "unknown")
        self.assertEqual(report["requested"]["generation"]["unknown_family"], 1)
        self.assertTrue(any("record no drawn family" in note
                            for note in report["bias"]["generation"]["notes"]))


class SuccessMetricsTest(unittest.TestCase):
    """Asset reuse, view coverage, shape and the door graph of the products."""

    def build(self, root, duplex=False):
        """Two furnished scenes, plus the two-storey fixture when asked."""
        rows = [
            write_scene(root, 0, scene_spec(ROW_LAYOUT, CHAIN, furniture=[
                {"room": "r0", "asset": "gen:sofa"},
                {"room": "r1", "asset": "gen:sofa"},
                {"room": "r2", "asset": "builtin:bed"}]),
                "row", stats=stats_block(auto_pick=[4, "row"]),
                forecast=forecast_block(c0=("r0", 0.4), c1=("r1", 0.9))),
            write_scene(root, 1, scene_spec(RING_LAYOUT, RING, furniture=[
                {"room": "r0", "asset": "gen:sofa"},
                {"room": "r1", "asset": "gen:lamp"}]),
                "loop", stats=stats_block(auto_pick=[4, "loop"]),
                forecast=forecast_block(c0=("r0", 0.8))),
        ]
        if duplex:
            with open(TWO_FLOOR_FIXTURE, encoding="utf-8") as handle:
                rows.append(write_scene(root, 2, json.load(handle), "duplex",
                                        stats=stats_block(auto_pick=None, floors=2,
                                                          shape="duplex", attempts_used=1)))
        write_root(root, rows, request={"layout": {"shape": "auto", "floors": "1"}})
        return product_report(root)

    def test_asset_reuse_spans_and_crosses_scenes(self):
        with tempfile.TemporaryDirectory() as root:
            report = self.build(root)
        reuse = report["success"]["furniture_reuse"]
        self.assertEqual(reuse["scenes"], 2)
        self.assertEqual(reuse["placements"], 5)
        self.assertEqual(reuse["distinct_assets"], 3)
        self.assertEqual(reuse["reuse_rate"], round(1 - 3 / 5, 4))
        self.assertEqual(reuse["assets_in_many_scenes"], 1)      # gen:sofa in both
        self.assertEqual(reuse["top"][0], {"asset": "gen:sofa", "placements": 3, "scenes": 2})
        self.assertEqual(reuse["by_prefix"]["gen"]["distinct"], 2)
        self.assertEqual(reuse["by_prefix"]["builtin"]["placements"], 1)

    def test_view_coverage_is_per_scene_and_per_stand(self):
        with tempfile.TemporaryDirectory() as root:
            report = self.build(root)
        views = report["success"]["view_coverage"]
        self.assertEqual(views["views_with_forecast"], 3)
        self.assertEqual(views["views_below_half"], 1)           # the 0.4 stand
        self.assertEqual(views["scenes_without_forecast"], 0)
        self.assertAlmostEqual(views["scene_visible_area_fraction"]["min"], 0.65, places=3)
        self.assertEqual(views["cameras"], 2)
        self.assertEqual(views["rooms_without_camera"], 6)       # one camera per scene
        self.assertEqual(views["room_coverage"], 0.25)

    def test_floors_shape_and_door_graph_are_reported(self):
        with tempfile.TemporaryDirectory() as root:
            report = self.build(root, duplex=True)
        success = report["success"]
        self.assertEqual(success["count"], 3)
        self.assertEqual(success["floors"], {"1": 2, "2": 1})
        # the row plan is 4:1, the ring square, the duplex 1.33:1
        self.assertEqual(success["shape_bins"]["aspect"], {"1.0-1.5": 2, "1.5-2.0": 0,
                                                           "2.0-3.0": 0, ">=3.0": 1})
        graph = success["door_graph"]
        self.assertEqual(graph["kinds"], {"door": 12, "open_passage": 1, "stair": 1})
        self.assertEqual(graph["cross_floor"], 1)
        self.assertEqual(graph["scenes_with_cycles"], 1)
        self.assertEqual(graph["degree_histogram"], {"1": 6, "2": 8, "3": 2})
        self.assertEqual(graph["leaf_rooms"], 6)
        self.assertEqual(graph["topologies"]["unique"], 3)
        self.assertEqual(success["family_share"], {"duplex": round(1 / 3, 4),
                                                   "loop": round(1 / 3, 4),
                                                   "row": round(1 / 3, 4)})
        # the duplex carries no door_visual_forecast: reported, not zeroed
        self.assertEqual(success["view_coverage"]["scenes_without_forecast"], 1)

    def test_a_root_without_a_batch_summary_still_reads(self):
        with tempfile.TemporaryDirectory() as root:
            write_scene(root, 0, scene_spec(ROW_LAYOUT, CHAIN), "row",
                        stats=stats_block(auto_pick=[4, "row"], shape="auto"))
            block = dr.read_root(root)
        self.assertEqual(block["source"], "scene_dirs")
        self.assertEqual(len(block["records"]), 1)
        self.assertEqual(block["records"][0]["family"], "row")

    def test_every_product_failing_still_produces_a_report(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [write_scene(root, 0, scene_spec(ROW_LAYOUT, CHAIN), "row",
                                ok=dict(ALL_OK, bundle_ok=False, review_ok=False),
                                failure={"stage": "review", "kind": "review_failed",
                                         "detail": "exposure"})]
            write_root(root, rows, request={"layout": {"shape": "auto"}})
            report = product_report(root)
        self.assertEqual(report["success"]["count"], 0)
        self.assertEqual(report["success"]["family_share"], {})
        self.assertEqual(report["success"]["furniture_reuse"]["placements"], 0)
        self.assertIsNone(report["success"]["fallback"]["retry_rate"])
        self.assertIsNone(report["success"]["view_coverage"]["room_coverage"])
        self.assertEqual(report["filtered"]["kept"], 0)
        self.assertEqual(report["filtered"]["reasons"], {"review:review_failed": 1})
        # every requested house vanished: half the distribution moved away
        self.assertEqual(report["bias"]["product"]["tv_distance"], 0.5)

    def test_quality_is_reported_next_to_the_diversity(self):
        """A more varied mix of worse houses is not a better dataset: both sides."""
        with tempfile.TemporaryDirectory() as root:
            rows = [
                write_scene(root, 0, scene_spec(ROW_LAYOUT, CHAIN), "row",
                            stats=stats_block(auto_pick=[4, "row"]),
                            review={"pass": True, "score": 0.9, "failed_checks": []},
                            validation={"pass": True, "pass_at_resolution": True}),
                write_scene(root, 1, scene_spec(RING_LAYOUT, RING), "loop",
                            stats=stats_block(auto_pick=[4, "loop"]),
                            ok=dict(ALL_OK, bundle_ok=False, review_ok=False),
                            failure={"stage": "review", "kind": "review_failed"},
                            review={"pass": False, "score": 0.4, "failed_checks": ["exposure"]}),
            ]
            write_root(root, rows, request={"layout": {"shape": "auto"}})
            report = product_report(root)
        success = report["success"]["quality"]
        self.assertEqual((success["reviewed"], success["review_passed"]), (1, 1))
        self.assertEqual(success["review_score"]["mean"], 0.9)
        self.assertEqual(success["validation_passed_at_resolution"], 1)
        self.assertEqual(success["by_family"]["row"]["review_pass"], 1)
        # the generated set still holds the one that the review rejected
        generated = report["generated"]["quality"]
        self.assertEqual((generated["reviewed"], generated["review_passed"]), (2, 1))
        self.assertEqual(generated["review_score"]["mean"], 0.65)
        self.assertEqual(generated["review_failed_checks"], {"exposure": 1})

    def test_a_scene_directory_supplies_the_seed_when_the_summary_lacks_it(self):
        with tempfile.TemporaryDirectory() as root:
            write_scene(root, 42, scene_spec(ROW_LAYOUT, CHAIN), "row",
                        stats=stats_block(auto_pick=[4, "row"], shape="auto"))
            path = os.path.join(root, "scene_0042", "run_summary.json")
            with open(path, encoding="utf-8") as handle:
                summary = json.load(handle)
            summary.pop("seed")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle)
            block = dr.read_root(root)
        self.assertEqual([record["seed"] for record in block["records"]], [42])
        self.assertEqual(block["skipped_rows"], 0)

    def test_a_root_with_neither_summary_nor_scenes_is_an_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                dr.read_root(root)


class SuccessCommandLineTest(unittest.TestCase):
    """`success` runs standalone, writes its own file and never overwrites."""

    def test_the_command_writes_a_second_report_next_to_the_old_ones(self):
        with tempfile.TemporaryDirectory() as root:
            write_scene(root, 0, scene_spec(ROW_LAYOUT, CHAIN), "row",
                        stats=stats_block(auto_pick=[4, "row"], shape="auto"))
            write_root(root, [{"seed": 0, "dir": "scene_0000", "status": "bundle_ok",
                               "ok": dict(ALL_OK), "failure": None, "resumed": False}],
                       request={"layout": {"shape": "auto"}})
            old_path = os.path.join(root, "diversity.json")
            with open(old_path, "w", encoding="utf-8") as handle:
                handle.write('{"old": true}')
            out = os.path.join(root, "diversity_success.json")
            result = subprocess.run([sys.executable, TOOL, "success", "--root", root,
                                     "--out", out, "--scenes"],
                                    capture_output=True, text=True, cwd=REPO_ROOT)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertIn("success fam.", result.stdout)
            self.assertIn("auto fallback", result.stdout)
            with open(old_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"old": True})
            with open(out, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["kind"], "panoworld_success_diversity")
            self.assertEqual(payload["success"]["count"], 1)
            self.assertEqual(len(payload["scenes"]["success"]), 1)

    def test_a_root_that_cannot_be_read_exits_two(self):
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run([sys.executable, TOOL, "success", "--root", root],
                                    capture_output=True, text=True, cwd=REPO_ROOT)
            self.assertEqual(result.returncode, 2)
            self.assertIn("no batch_summary.json", result.stderr)

    @unittest.skipUnless(os.path.isdir(BENCHMARK_ROOTS), "benchmark100 artifacts are not present")
    def test_the_benchmark_products_recompute(self):
        """The shipped benchmark: a layout-only sweep plus a rendered sample."""
        argv = [sys.executable, TOOL, "success",
                "--root", os.path.join(BENCHMARK_ROOTS, "layout"),
                "--root", os.path.join(BENCHMARK_ROOTS, "sample")]
        result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("products       :", result.stdout)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "diversity_success.json")
            written = subprocess.run(argv + ["--out", out], capture_output=True, text=True,
                                     cwd=REPO_ROOT)
            self.assertEqual(written.returncode, 0, written.stderr[-2000:])
            with open(out, encoding="utf-8") as handle:
                payload = json.load(handle)
        # The sweep was built before stats.auto_pick existed: every drawn family
        # of it must come back marked as replayed, never as recorded.
        self.assertEqual(payload["requested"]["generation"]["basis"], {"replayed": 100})
        self.assertEqual(payload["generated"]["count"], 100)
        self.assertEqual(payload["filtered"]["requested"], 10)
        self.assertTrue(payload["filtered"]["kept"] < payload["filtered"]["products"])
        self.assertGreater(payload["bias"]["generation"]["tv_distance"], 0.3)
        # The products are a stratified sample of the sweep: the report says so
        # instead of reading the sample as the delivered mix of all 100 seeds.
        self.assertIn("the product batch covers", payload["bias"]["product"]["notes"][0])


class CommandLineTest(unittest.TestCase):
    """The CLI still runs standalone and prints the corrected sections."""

    def test_specs_command_reports_the_new_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for seed, (_name, layout, connections) in enumerate(TOPOLOGIES):
                scene = os.path.join(tmpdir, "scene_%04d" % seed)
                os.makedirs(scene)
                with open(os.path.join(scene, "spec.json"), "w", encoding="utf-8") as handle:
                    json.dump(scene_spec(layout, connections), handle)
            out = os.path.join(tmpdir, "diversity.json")
            result = subprocess.run([sys.executable, TOOL, "specs", tmpdir,
                                     "--out", out, "--scenes"],
                                    capture_output=True, text=True, cwd=REPO_ROOT)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertIn("connections", result.stdout)
            with open(out, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["count"], 3)
            summary = payload["summary"]
            self.assertEqual(summary["structures"]["unique"], 3)
            self.assertEqual(summary["connections"]["total"], 10)
            self.assertEqual(len(payload["scenes"]), 3)
            self.assertIn("structure_key", payload["scenes"][0])
            self.assertEqual(payload["scenes"][2]["structure"]["edge_kinds"], {"door": 4})
            self.assertEqual(payload["scenes"][2]["structure"]["cycles"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
