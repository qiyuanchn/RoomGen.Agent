#!/usr/bin/env python3
"""The viewer manifest carries the storey a room / camera belongs to.

`blender_agent/export_glb.py` runs inside Blender, but its manifest helper is
plain Python; these tests import it on the host and check the glTF Y values for
a two-storey spec (f0/f1 room centres at their own elevation + eye height) and
the unchanged single-storey numbers.  The `serve.py` fallback converter is
covered with the same geometry.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from blender_agent import export_glb, spec as spec_mod  # noqa: E402

SERVE_PATH = os.path.join(REPO, "tools", "scene3d_viewer", "serve.py")


def load_serve():
    module_spec = importlib.util.spec_from_file_location("scene3d_viewer_serve", SERVE_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


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
        "cameras": [
            {"name": "c0", "room": "living", "pos": [2.0, 2.0, 1.5]},
        ],
    }
    return spec_mod.parse_spec(data)


class ViewerManifestTest(unittest.TestCase):
    def test_two_storey_room_centres_use_their_own_elevation(self):
        manifest = export_glb.viewer_manifest(duplex_spec(), "scene.glb", {}, None, [])
        rooms = {room["id"]: room for room in manifest["rooms"]}
        self.assertEqual(rooms["hall"]["floor_id"], "f0")
        self.assertEqual(rooms["attic"]["floor_id"], "f1")
        self.assertAlmostEqual(rooms["hall"]["floor_elevation"], 0.0)
        self.assertAlmostEqual(rooms["attic"]["floor_elevation"], 2.85)
        self.assertAlmostEqual(rooms["hall"]["center"][1], 1.5)
        self.assertAlmostEqual(rooms["attic"]["center"][1], 4.35)
        self.assertEqual([floor["id"] for floor in manifest["floors"]], ["f0", "f1"])
        self.assertEqual(manifest["floors"][1]["rooms"], ["attic"])
        self.assertAlmostEqual(manifest["floors"][1]["slab_bottom"], 2.7)

    def test_cameras_report_the_storey_elevation(self):
        manifest = export_glb.viewer_manifest(duplex_spec(), "scene.glb", {}, None, [])
        cameras = {camera["name"]: camera for camera in manifest["cameras"]}
        self.assertEqual(cameras["c0"]["floor_id"], "f0")
        self.assertEqual(cameras["c1"]["floor_id"], "f1")
        self.assertAlmostEqual(cameras["c1"]["floor_elevation"], 2.85)
        self.assertAlmostEqual(cameras["c1"]["position"][1], 4.35)

    def test_room_polygons_are_in_gltf_xz(self):
        manifest = export_glb.viewer_manifest(duplex_spec(), "scene.glb", {}, None, [])
        room = manifest["rooms"][0]
        self.assertEqual(room["polygon"][0], [0.0, 0.0])
        self.assertEqual(room["polygon"][2], [4.0, -4.0])

    def test_single_storey_manifest_is_unchanged(self):
        manifest = export_glb.viewer_manifest(legacy_spec(), "scene.glb", {}, None, [])
        room = manifest["rooms"][0]
        self.assertEqual(manifest["floors"], [])
        self.assertIsNone(room["floor_id"])
        self.assertAlmostEqual(room["center"][1], 1.5)
        self.assertAlmostEqual(manifest["cameras"][0]["position"][1], 1.5)

    def test_serve_fallback_converts_floor_elevations(self):
        serve = load_serve()
        build_manifest = {
            "spec": {
                "wall_height": 2.7,
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
            },
        }
        manifest = serve.manifest_from_build_manifest(build_manifest)
        rooms = {room["id"]: room for room in manifest["rooms"]}
        cameras = {camera["name"]: camera for camera in manifest["cameras"]}
        self.assertAlmostEqual(rooms["hall"]["center"][1], 1.5)
        self.assertAlmostEqual(rooms["attic"]["center"][1], 4.35)
        self.assertEqual(rooms["attic"]["floor_id"], "f1")
        self.assertAlmostEqual(cameras["c1"]["floor_elevation"], 2.85)
        self.assertAlmostEqual(manifest["floors"][1]["top"], 5.55)

    def test_serve_payload_passes_the_floor_table_to_the_page(self):
        serve = load_serve()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            glb = root / "scene.glb"
            glb.write_bytes(b"glTF")
            manifest = {
                "eye_height": 1.5,
                "floors": [{"id": "f1", "elevation": 2.85, "height": 2.7,
                            "slab_thickness": 0.15, "slab_bottom": 2.7, "top": 5.55,
                            "rooms": ["attic"]}],
                "rooms": [{"id": "attic", "floor_id": "f1", "center": [2.0, 4.35, 2.0],
                           "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]}],
                "cameras": [{"name": "c1", "floor_id": "f1", "floor_elevation": 2.85,
                             "position": [2.0, 4.35, 2.0]}],
            }
            (root / "scene3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            payload = serve.build_payload(root, glb)
            self.assertEqual(payload["floors"][0]["id"], "f1")
            self.assertAlmostEqual(payload["rooms"][0]["center"][1], 4.35)
            self.assertEqual(payload["cameras"][0]["floor_id"], "f1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
