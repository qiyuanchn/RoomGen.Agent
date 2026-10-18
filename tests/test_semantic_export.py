#!/usr/bin/env python3
"""Stable semantic identity for the GLB export (host python, no bpy).

Run::

    python tests/test_semantic_export.py

The Blender-side exporter writes ``instance_id`` / ``category_id`` /
``room_id`` / ``floor_id`` as glTF ``node.extras`` and mirrors them in
``scene3d_manifest.json -> semantics.objects``.  These tests drive the pure
mapping helpers with Blender-like object stubs, then verify the extras checker
on a hand-built GLB document (including collection-instance children, which
inherit their parent's ids).
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import export_glb  # noqa: E402
from blender_agent import spec as spec_mod  # noqa: E402


class SceneObject:
    """The two attributes the resolver reads from a bpy object."""

    def __init__(self, name: str, type: str = "MESH", properties: dict = None):
        self.name = name
        self.type = type
        self._properties = properties or {}

    def get(self, key, default=None):
        return self._properties.get(key, default)


def duplex_spec() -> spec_mod.SceneSpec:
    data = {
        "schema_version": 2,
        "unit": "meter",
        "wall_height": 2.7,
        "wall_thickness": 0.12,
        "floors": [
            {"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
            {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15},
        ],
        "rooms": [
            {"id": "hall", "type": "living_room", "floor_id": "f0",
             "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
            {"id": "attic", "type": "bedroom", "floor_id": "f1",
             "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        ],
        "cameras": [
            {"name": "c0", "room": "hall", "pos": [2.0, 2.0, 1.5]},
            {"name": "c1", "room": "attic", "pos": [2.0, 2.0, 4.35]},
        ],
        "furniture": [
            {"asset": "ph:sofa_02", "room": "hall", "pos": [3.2, 3.2], "name": "sofa"},
            {"asset": "builtin:dining_chair", "room": "hall", "pos": [1.0, 1.0], "name": "chair_a"},
            {"asset": "builtin:dining_chair", "room": "hall", "pos": [2.0, 1.0], "name": "chair_b"},
            {"asset": "builtin:rug", "room": "hall", "pos": [2.0, 2.0],
             "size": [1.6, 1.0], "name": "rug"},
            {"asset": "builtin:table_lamp", "room": "hall", "pos": [0.5, 3.5], "name": "bedside_lamp"},
            {"asset": "ph:sofa_02", "room": "attic", "pos": [1.2, 1.2], "name": "sofa"},
            {"asset": "gen:bed_01", "room": "attic", "pos": [2.8, 2.6], "name": "bed"},
            {"asset": "builtin:dining_chair", "room": "attic", "pos": [1.0, 2.0], "name": "chair_a"},
            {"asset": "builtin:dining_chair", "room": "attic", "pos": [2.0, 2.0], "name": "chair_b"},
        ],
    }
    return spec_mod.parse_spec(data)


def simple_duplex_spec() -> spec_mod.SceneSpec:
    """Two identical sofas with the same placement name, one per storey."""
    data = {
        "schema_version": 2,
        "unit": "meter",
        "wall_height": 2.7,
        "wall_thickness": 0.12,
        "floors": [
            {"id": "f0", "elevation": 0.0, "height": 2.7, "slab_thickness": 0.15},
            {"id": "f1", "elevation": 2.85, "height": 2.7, "slab_thickness": 0.15},
        ],
        "rooms": [
            {"id": "hall", "type": "living_room", "floor_id": "f0",
             "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
            {"id": "attic", "type": "bedroom", "floor_id": "f1",
             "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        ],
        "cameras": [{"name": "c0", "room": "hall", "pos": [2.0, 2.0, 1.5]}],
        "furniture": [
            {"asset": "ph:sofa_02", "room": "hall", "pos": [3.2, 3.2], "name": "sofa"},
            {"asset": "ph:sofa_02", "room": "attic", "pos": [1.2, 1.2], "name": "sofa"},
        ],
    }
    return spec_mod.parse_spec(data)


def legacy_spec() -> spec_mod.SceneSpec:
    data = {
        "unit": "meter",
        "wall_height": 2.7,
        "wall_thickness": 0.12,
        "rooms": [
            {"id": "living", "type": "living_room",
             "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        ],
        "openings": [],
        "windows": [
            {"room": "living", "wall": [[0.8, 0.0], [2.2, 0.0]], "sill": 0.9, "height": 1.4},
        ],
        "cameras": [{"name": "c0", "room": "living", "pos": [2.0, 2.0, 1.5]}],
        "details": {"skirting": True, "casings": True, "glass": True},
        "furniture": [
            {"asset": "ph:sofa_02", "room": "living", "pos": [3.2, 3.2], "name": "sofa"},
        ],
    }
    return spec_mod.parse_spec(data)


def duplex_objects() -> list:
    objects = [
        SceneObject("furn_sofa", "EMPTY"),
        SceneObject("furn_sofa.001", "EMPTY"),
        SceneObject("furn_bed", "EMPTY"),
        SceneObject("rug_rug"),
        SceneObject("lamp_bedside_lamp_base"),
        SceneObject("lamp_bedside_lamp_stem"),
        SceneObject("lamp_bedside_lamp_shade"),
        SceneObject("floor_hall"),
        SceneObject("floor_attic"),
    ]
    for label in ("chair_a", "chair_b"):
        for suffix in ("", ".001"):
            for part in ("seat", "back"):
                objects.append(SceneObject(f"fixture_{label}_{part}{suffix}"))
    segments_f0 = spec_mod.build_wall_segments(
        [room for room in duplex_spec().rooms if room.floor_id == "f0"])
    for index, segment in enumerate(segments_f0):
        label = "_".join(segment.owners) or "wall"
        name = (f"wall_{index:02d}_{label}_{segment.p0[0]:.2f}_"
                f"{segment.p0[1]:.2f}").replace("-", "m")
        objects.append(SceneObject(name))
    objects.append(SceneObject("mystery_mesh"))
    return objects


def write_glb(path: str, document: dict) -> None:
    """Write a minimal binary glTF (12-byte header + JSON chunk)."""
    payload = json.dumps(document).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total = 12 + 8 + len(payload)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total))
        handle.write(struct.pack("<II", len(payload), 0x4E4F534A))
        handle.write(payload)


def extras_for(semantics: dict, name: str) -> dict:
    return {field: semantics["objects"][name][field]
            for field in export_glb.SEMANTIC_FIELDS}


class SemanticIdentityTest(unittest.TestCase):
    def setUp(self):
        self.spec = duplex_spec()
        self.objects = duplex_objects()
        self.semantics = export_glb.build_semantics(self.spec, self.objects)
        self.entries = self.semantics["objects"]

    def test_every_exported_object_has_four_ids(self):
        exported = {obj.name for obj in self.objects}
        self.assertEqual(set(self.entries), exported)
        for name, entry in self.entries.items():
            for field in export_glb.SEMANTIC_FIELDS:
                self.assertIn(field, entry, f"{name} is missing {field}")
                self.assertIsInstance(entry[field], str)
            self.assertTrue(entry["category_id"], f"{name} has an empty category")

    def test_two_storey_rooms_carry_their_own_floor(self):
        lower = self.entries["furn_sofa"]
        upper = self.entries["furn_sofa.001"]
        self.assertEqual((lower["room_id"], lower["floor_id"]), ("hall", "f0"))
        self.assertEqual((upper["room_id"], upper["floor_id"]), ("attic", "f1"))
        self.assertEqual(lower["category_id"], "sofa")
        self.assertEqual(upper["category_id"], "sofa")

    def test_repeated_same_category_instances_have_unique_ids(self):
        chairs = [entry for entry in self.entries.values() if entry["category_id"] == "chair"]
        instance_ids = {entry["instance_id"] for entry in chairs}
        self.assertEqual(len(chairs), 8)
        self.assertEqual(len(instance_ids), 4)
        self.assertEqual(
            self.semantics["categories"]["chair"]["instance_count"], 4)

    def test_fixture_parts_of_one_placement_share_an_instance(self):
        seat = self.entries["fixture_chair_a_seat"]
        back = self.entries["fixture_chair_a_back"]
        lower_again = self.entries["fixture_chair_a_seat.001"]
        self.assertEqual(seat["instance_id"], back["instance_id"])
        self.assertNotEqual(seat["instance_id"], lower_again["instance_id"])
        self.assertEqual(seat["floor_id"], "f0")
        self.assertEqual(lower_again["floor_id"], "f1")

    def test_builtin_and_generated_assets_get_categories(self):
        self.assertEqual(self.entries["rug_rug"]["category_id"], "rug")
        self.assertEqual(self.entries["lamp_bedside_lamp_base"]["category_id"], "lamp")
        self.assertEqual(self.entries["lamp_bedside_lamp_stem"]["instance_id"],
                         self.entries["lamp_bedside_lamp_base"]["instance_id"])
        self.assertEqual(self.entries["furn_bed"]["category_id"], "bed")
        self.assertEqual(self.entries["furn_bed"]["floor_id"], "f1")

    def test_unmatched_objects_still_get_ids(self):
        entry = self.entries["mystery_mesh"]
        self.assertEqual(entry["category_id"], "mystery")
        self.assertIn("mystery_mesh", self.semantics["fallback_objects"])
        self.assertEqual(entry["instance_id"], "object:mystery_mesh")

    def test_manifest_keeps_the_viewer_fields_and_adds_semantics(self):
        manifest = export_glb.viewer_manifest(
            self.spec, "scene.glb", {"meshes": 1}, None, [], semantics=self.semantics)
        for key in ("kind", "bounds", "floors", "cameras", "rooms", "stats", "excluded"):
            self.assertIn(key, manifest)
        self.assertIs(manifest["semantics"], self.semantics)
        self.assertEqual(manifest["rooms"][1]["floor_id"], "f1")

    def test_manifest_without_semantics_is_unchanged(self):
        manifest = export_glb.viewer_manifest(
            self.spec, "scene.glb", {}, None, [])
        self.assertNotIn("semantics", manifest)


class LegacyAndDetailTest(unittest.TestCase):
    def test_legacy_single_storey_uses_empty_floor_id(self):
        spec = legacy_spec()
        objects = [
            SceneObject("furn_sofa", "EMPTY"),
            SceneObject("floor_living"),
            SceneObject("ceiling_living"),
        ]
        semantics = export_glb.build_semantics(spec, objects)
        for name in ("furn_sofa", "floor_living", "ceiling_living"):
            self.assertEqual(semantics["objects"][name]["floor_id"], "")
        self.assertEqual(semantics["objects"]["furn_sofa"]["room_id"], "living")

    def test_detail_objects_map_to_their_room_and_window(self):
        spec = legacy_spec()
        objects = [
            SceneObject("skirting_living_00_0000"),
            SceneObject("window_casing_0_a"),
            SceneObject("window_casing_0_head"),
            SceneObject("window_sill_0"),
            SceneObject("window_glass_0"),
        ]
        semantics = export_glb.build_semantics(spec, objects)
        self.assertEqual(semantics["objects"]["skirting_living_00_0000"]["instance_id"],
                         "skirting:living")
        self.assertEqual(semantics["objects"]["window_sill_0"]["category_id"], "window_sill")
        self.assertEqual(semantics["objects"]["window_glass_0"]["room_id"], "living")
        self.assertEqual(semantics["objects"]["window_casing_0_a"]["instance_id"],
                         "detail:window:0:window_casing")


class GlbExtrasTest(unittest.TestCase):
    def setUp(self):
        self.spec = simple_duplex_spec()
        self.objects = [
            SceneObject("furn_sofa", "EMPTY"),
            SceneObject("furn_sofa.001", "EMPTY"),
            SceneObject("floor_hall"),
            SceneObject("floor_attic"),
        ]
        self.semantics = export_glb.build_semantics(self.spec, self.objects)
        self.manifest = export_glb.viewer_manifest(
            self.spec, "scene.glb", {}, None, [], semantics=self.semantics)
        self.document = {
            "asset": {"version": "2.0"},
            "nodes": [
                {"name": "furn_sofa", "extras": extras_for(self.semantics, "furn_sofa"),
                 "children": [1]},
                {"name": "asset_sofa_02", "mesh": 0},
                {"name": "furn_sofa.001",
                 "extras": extras_for(self.semantics, "furn_sofa.001"), "children": [3]},
                {"name": "asset_sofa_02", "mesh": 0},
                {"name": "floor_hall", "mesh": 0,
                 "extras": extras_for(self.semantics, "floor_hall")},
                {"name": "floor_attic", "mesh": 0,
                 "extras": extras_for(self.semantics, "floor_attic")},
            ],
        }

    def test_glb_parser_reads_the_json_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.glb")
            write_glb(path, self.document)
            loaded = export_glb.read_glb_document(path)
        self.assertEqual(loaded["nodes"][0]["name"], "furn_sofa")

    def test_extras_match_the_manifest_and_children_inherit(self):
        export_glb.attach_glb_nodes(self.document, self.semantics)
        failures, warnings, report = export_glb.verify_glb_semantics(
            self.document, self.manifest)
        self.assertEqual(failures, [])
        self.assertEqual(warnings, [])
        self.assertEqual(report["mesh_nodes"], 4)
        self.assertEqual(report["resolved_mesh_nodes"], 4)
        # each parent lists its own instanced child mesh
        self.assertEqual(self.semantics["objects"]["furn_sofa"]["glb_nodes"], [0, 1])
        self.assertEqual(self.semantics["objects"]["furn_sofa.001"]["glb_nodes"], [2, 3])
        # navigation can read the ids straight from a GLB node index
        self.assertEqual(self.semantics["nodes"]["0"]["source"], "self")
        self.assertEqual(self.semantics["nodes"]["1"]["source"], "ancestor")
        self.assertEqual(self.semantics["nodes"]["1"]["instance_id"], "furniture:0000")

    def test_a_wrong_floor_in_extras_is_a_failure(self):
        export_glb.attach_glb_nodes(self.document, self.semantics)
        self.document["nodes"][0]["extras"]["floor_id"] = "f1"
        failures, _warnings, _report = export_glb.verify_glb_semantics(
            self.document, self.manifest)
        self.assertTrue(any("floor_id 'f1'" in failure for failure in failures))

    def test_a_child_without_semantic_parent_is_a_failure(self):
        export_glb.attach_glb_nodes(self.document, self.semantics)
        self.document["nodes"][0].pop("extras")
        failures, _warnings, _report = export_glb.verify_glb_semantics(
            self.document, self.manifest)
        self.assertTrue(any("mesh node 1" in failure and "no instance" in failure
                            for failure in failures))

    def test_legacy_manifest_without_semantics_is_only_a_warning(self):
        failures, warnings, report = export_glb.verify_glb_semantics(
            self.document, {"rooms": [], "floors": []})
        self.assertEqual(failures, [])
        self.assertEqual(report, {})
        self.assertTrue(any("legacy" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
