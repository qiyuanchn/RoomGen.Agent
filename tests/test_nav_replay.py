#!/usr/bin/env python3
"""PointNav replay builder: synthetic episodes -> static site -> HTTP check.

The fixture follows the agreed evaluator schema and deliberately contains one
episode per failure mode (stuck at a door, through-wall, cross-void, wrong-floor
jump), a successful episode, an episode without a trajectory and an episode
whose scene geometry is missing.  The tests pin both the derived failure
evidence and the fact that the generated static site serves and carries the
same data the page reads.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_SCRIPT = os.path.join(REPO, "tools", "build_nav_replay.py")
ASSET_DIR = os.path.join(REPO, "tools", "nav_replay")


def load_builder():
    spec = importlib.util.spec_from_file_location("build_nav_replay", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_builder()


SCENES = {
    "coordinate_system": "gltf",
    "scenes": [
        {
            "scene_id": "duplex_a",
            "floors": [
                {"id": "f0", "elevation": 0.0, "height": 2.8, "name": "ground"},
                {"id": "f1", "elevation": 3.0, "height": 2.8, "name": "upper"},
            ],
            "rooms": [
                {
                    "id": "living", "type": "living_room", "floor_id": "f0",
                    "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]],
                },
                {
                    "id": "kitchen", "type": "kitchen", "floor_id": "f0",
                    "polygon": [[4, 0], [7, 0], [7, 4], [4, 4]],
                },
                {
                    "id": "loft", "type": "bedroom", "floor_id": "f1",
                    "polygon": [[0, 0], [7, 0], [7, 4], [0, 4]],
                },
            ],
            "voids": [
                {
                    "id": "stair_hole", "floor_id": "f0",
                    "polygon": [[2.6, 0.2], [3.4, 0.2], [3.4, 0.8], [2.6, 0.8]],
                },
            ],
            "portals": [
                {"id": "living_kitchen_door", "floor_id": "f0",
                 "position": [4.0, 0.0, 1.2], "radius": 0.5},
            ],
        },
    ],
}


def episode(episode_id, **overrides):
    base = {
        "scene_id": "duplex_a",
        "episode_id": episode_id,
        "seed": 7,
        "start": [0.8, 0.0, 2.0],
        "goal": [3.2, 0.0, 2.0],
        "start_floor": "f0",
        "goal_floor": "f0",
        "reference_path": {
            "points": [[0.8, 0.0, 2.0], [2.0, 0.0, 2.0], [3.2, 0.0, 2.0]],
            "distance": 2.4,
        },
        "trajectory": [
            {"position": [0.8, 0.0, 2.0], "yaw": 0.0, "action": 0, "collision": False},
            {"position": [1.4, 0.0, 2.0], "yaw": 0.0, "action": 1, "collision": False},
            {"position": [2.0, 0.0, 2.0], "yaw": 0.0, "action": 1, "collision": False},
            {"position": [2.6, 0.0, 2.0], "yaw": 0.0, "action": 1, "collision": False},
            {"position": [3.2, 0.0, 2.0], "yaw": 0.0, "action": 1, "collision": False},
        ],
        "success": True,
        "spl": 1.0,
        "collisions": 0,
        "timeout": False,
        "failure_reason": None,
    }
    base.update(overrides)
    return base


def synthetic_episodes():
    stuck_trajectory = [
        {"position": [0.8, 0.0, 1.2], "yaw": 0.0, "action": 0, "collision": False},
        {"position": [2.0, 0.0, 1.2], "yaw": 0.0, "action": 1, "collision": False},
        {"position": [3.0, 0.0, 1.2], "yaw": 0.0, "action": 1, "collision": False},
    ] + [
        {"position": [3.95, 0.0, 1.2], "yaw": 0.0, "action": 1, "collision": True}
        for _ in range(6)
    ]
    return [
        episode("ep_ok"),
        episode(
            "ep_stuck",
            start=[0.8, 0.0, 1.2],
            goal=[6.3, 0.0, 1.2],
            reference_path={
                "points": [[0.8, 0.0, 1.2], [6.3, 0.0, 1.2]],
                "distance": 5.5,
            },
            trajectory=stuck_trajectory,
            success=False,
            spl=0.05,
            collisions=6,
            timeout=False,
            failure_reason="stuck at door: no forward progress",
        ),
        episode(
            "ep_wall",
            start=[0.5, 0.0, 3.2],
            goal=[6.0, 0.0, 3.2],
            reference_path={
                "points": [[0.5, 0.0, 3.2], [6.0, 0.0, 3.2]],
                "distance": 5.5,
            },
            trajectory=[
                {"position": [0.5, 0.0, 3.2], "yaw": 0.0, "action": 0, "collision": False},
                {"position": [2.0, 0.0, 3.2], "yaw": 0.0, "action": 1, "collision": False},
                {"position": [4.5, 0.0, 3.2], "yaw": 0.0, "action": 1, "collision": False},
                {"position": [6.0, 0.0, 3.2], "yaw": 0.0, "action": 1, "collision": False},
            ],
            success=False,
            spl=0.0,
            collisions=0,
            timeout=False,
            failure_reason="went through wall between living and kitchen",
        ),
        episode(
            "ep_void",
            start=[1.0, 0.0, 0.5],
            goal=[3.5, 0.0, 0.5],
            reference_path={
                "points": [[1.0, 0.0, 0.5], [3.5, 0.0, 0.5]],
                "distance": 2.5,
            },
            trajectory=[
                {"position": [1.0, 0.0, 0.5], "yaw": 0.0, "action": 0, "collision": False},
                {"position": [2.4, 0.0, 0.5], "yaw": 0.0, "action": 1, "collision": False},
                {"position": [3.0, 0.0, 0.5], "yaw": 0.0, "action": 1, "collision": False},
                {"position": [3.5, 0.0, 0.5], "yaw": 0.0, "action": 1, "collision": False},
            ],
            success=False,
            spl=0.0,
            collisions=0,
            timeout=True,
            failure_reason=None,
        ),
        episode(
            "ep_floor",
            start=[1.0, 0.0, 1.0],
            goal=[1.0, 3.0, 1.0],
            start_floor="f0",
            goal_floor="f1",
            reference_path={
                "points": [[1.0, 0.0, 1.0], [1.0, 1.5, 1.0], [1.0, 3.0, 1.0]],
                "distance": 3.0,
            },
            trajectory=[
                {"position": [1.0, 0.0, 1.0], "yaw": 0.0, "action": 0, "collision": False},
                {"position": [1.0, 0.1, 1.0], "yaw": 0.0, "action": 1, "collision": False},
                {"position": [1.0, 3.0, 1.0], "yaw": 0.0, "action": 1, "collision": False},
            ],
            success=False,
            spl=0.0,
            collisions=0,
            timeout=False,
            failure_reason="wrong floor jump: teleported to f1",
        ),
        episode(
            "ep_missing_trajectory",
            start=[0.5, 0.0, 0.5],
            goal=[2.0, 0.0, 2.0],
            reference_path={
                "points": [[0.5, 0.0, 0.5], [2.0, 0.0, 2.0]],
                "distance": 2.1213,
            },
            trajectory=[],
            success=False,
            spl=0.0,
            collisions=0,
            timeout=True,
            failure_reason="timeout: no progress",
        ),
        episode(
            "ep_missing_scene",
            scene_id="missing_scene",
            start=[0.0, 0.0, 0.0],
            goal=[2.0, 0.0, 0.0],
            start_floor="f0",
            goal_floor="f1",
            reference_path={
                "points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                "distance": 2.0,
            },
            trajectory=[
                {"position": [0.0, 0.0, 0.0], "yaw": 0.0, "action": 0, "collision": False},
                {"position": [1.0, 0.0, 0.0], "yaw": 0.0, "action": 1, "collision": True},
                {"position": [2.0, 0.0, 0.0], "yaw": 0.0, "action": 1, "collision": True},
            ],
            success=False,
            spl=0.0,
            collisions=2,
            timeout=False,
            failure_reason="stuck at door in a scene with no geometry",
        ),
    ]


def write_fixture(directory, include_bad_line=True):
    directory = Path(directory)
    (directory / "scenes.json").write_text(
        json.dumps(SCENES, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps({
            "overall": {
                "success_rate": 0.25,
                "spl": 0.2,
                "episodes": 7,
                "collisions": 9,
                "timeouts": 2,
            },
            "by_scene": {
                "duplex_a": {"success_rate": 0.2, "spl": 0.17},
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [json.dumps(item, ensure_ascii=False) for item in synthetic_episodes()]
    if include_bad_line:
        lines.insert(3, "{not valid json")
    (directory / "episodes.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def read_data_js(path):
    text = Path(path).read_text(encoding="utf-8")
    marker = "window.NAV_REPLAY_DATA = "
    start = text.index(marker) + len(marker)
    end = text.rindex(";")
    return json.loads(text[start:end])


DOM_SMOKE_JS = r"""
const fs = require('fs');
const vm = require('vm');
const site = process.argv[2];

class ClassList {
  constructor(owner) { this.owner = owner; this.items = new Set(); }
  add(...names) { for (const name of names) this.items.add(name); this.sync(); }
  remove(...names) { for (const name of names) this.items.delete(name); this.sync(); }
  toggle(name, force) {
    const wanted = force === undefined ? !this.items.has(name) : !!force;
    if (wanted) this.items.add(name); else this.items.delete(name);
    this.sync(); return wanted;
  }
  contains(name) { return this.items.has(name); }
  sync() { this.owner._className = [...this.items].join(' '); }
}

class Element {
  constructor(tag = 'div') {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.listeners = {};
    this.classList = new ClassList(this);
    this.style = {};
    this.dataset = {};
    this.textContent = '';
    this.value = '';
    this.hidden = false;
    this.disabled = false;
    this.title = '';
    this.max = '0';
    this.offsetWidth = 0;
    this._className = '';
    this._fragment = false;
  }
  set className(value) {
    this._className = value;
    this.classList.items = new Set(String(value).split(/\s+/).filter(Boolean));
  }
  get className() { return this._className; }
  get options() { return this.children.filter((child) => child.tagName === 'OPTION'); }
  appendChild(child) {
    if (child && child._fragment) {
      for (const item of child.children.slice()) this.appendChild(item);
      child.children = [];
      return child;
    }
    this.children.push(child);
    return child;
  }
  append(...nodes) { for (const node of nodes) this.appendChild(node); }
  replaceChildren(...nodes) {
    this.children = [];
    for (const node of nodes) this.appendChild(node);
  }
  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }
  dispatch(type, event = {}) {
    for (const fn of this.listeners[type] || []) {
      fn({ target: this, preventDefault() {}, ...event });
    }
  }
  querySelectorAll(selector) {
    const wanted = selector.startsWith('.') ? selector.slice(1) : null;
    const found = [];
    const visit = (node) => {
      if (wanted && node.classList && node.classList.contains(wanted)) found.push(node);
      for (const child of node.children || []) visit(child);
    };
    visit(this);
    return found;
  }
  getBoundingClientRect() { return { width: 900, height: 520, left: 0, top: 0 }; }
  getContext() { return context2d; }
  setPointerCapture() {}
  releasePointerCapture() {}
  hasPointerCapture() { return false; }
}

const drawCalls = { count: 0 };
const context2d = new Proxy({}, {
  get(target, prop) {
    if (prop === 'measureText') return () => ({ width: 10 });
    if (prop in target) return target[prop];
    return () => { drawCalls.count += 1; };
  },
  set(target, prop, value) { target[prop] = value; return true; },
});

const html = fs.readFileSync(site + '/index.html', 'utf8');
const elements = {};
for (const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g)) {
  elements[match[2]] = new Element(match[1]);
}
const document = {
  body: new Element('body'),
  getElementById(id) { return elements[id] || null; },
  createElement(tag) { return new Element(tag); },
  createDocumentFragment() {
    const fragment = new Element('fragment');
    fragment._fragment = true;
    return fragment;
  },
};
const rafQueue = [];
const sandbox = {
  console,
  document,
  window: null,
  devicePixelRatio: 1,
  requestAnimationFrame(fn) { rafQueue.push(fn); return rafQueue.length; },
  cancelAnimationFrame() {},
  ResizeObserver: class { observe() {} },
};
sandbox.window = sandbox;
sandbox.window.addEventListener = () => {};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(site + '/replay-data.js', 'utf8'), sandbox);
vm.runInContext(fs.readFileSync(site + '/replay.js', 'utf8'), sandbox);

function check(condition, message) {
  if (!condition) throw new Error(message);
}
check(sandbox.window.NAV_REPLAY_DATA.episodes.length === 7, 'data not loaded');
check(drawCalls.count > 50, 'canvas was not drawn');
check(elements['episode-title'].textContent.includes('ep_ok'),
  'first episode not selected: ' + elements['episode-title'].textContent);
check(elements['episode-list'].children.length === 7, 'episode list length');
check(elements['floor-select'].options.length >= 2, 'floor selector not populated');

const stuck = elements['episode-list'].querySelectorAll('.episode-item')
  .find((button) => button.dataset.key === 'duplex_a::ep_stuck');
check(stuck, 'stuck episode button missing');
stuck.dispatch('click');
check(elements['episode-title'].textContent.includes('ep_stuck'), 'stuck not selected');
check(elements['result-badge'].textContent.includes('Stuck'), 'failure badge');
check(elements['event-list'].children.length > 0, 'event rows missing');

elements['category-filter'].value = 'through_wall';
elements['category-filter'].dispatch('change');
check(elements['episode-list'].children.length === 1, 'category filter');
check(elements['episode-title'].textContent.includes('ep_wall'), 'wall not selected');
elements['reset-filters'].dispatch('click');
check(elements['episode-list'].children.length === 7, 'reset filters');

elements['play-button'].dispatch('click');
check(rafQueue.length > 0, 'play did not schedule a frame');
rafQueue.shift()(1000);
check(drawCalls.count > 50, 'playback did not keep drawing');
console.log('nav replay DOM smoke OK (' + drawCalls.count + ' canvas calls)');
"""


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class NavReplayBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nav_replay_")
        cls.eval_dir = Path(cls.tmp.name) / "eval"
        cls.out_dir = Path(cls.tmp.name) / "site"
        cls.eval_dir.mkdir()
        write_fixture(cls.eval_dir)
        cls.result = subprocess.run(
            [
                sys.executable, BUILD_SCRIPT,
                "--episodes", str(cls.eval_dir / "episodes.jsonl"),
                "--scenes", str(cls.eval_dir / "scenes.json"),
                "--metrics", str(cls.eval_dir / "metrics.json"),
                "--out", str(cls.out_dir),
                "--title", "Synthetic PointNav replay",
                "--quiet",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        )
        cls.data = read_data_js(cls.out_dir / "replay-data.js")
        cls.episodes = {item["key"]: item for item in cls.data["episodes"]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_build_succeeds_and_writes_the_static_site(self):
        self.assertEqual(
            self.result.returncode, 0, self.result.stdout + self.result.stderr
        )
        for name in ("index.html", "replay.js", "replay.css",
                     "replay-data.js", "replay_manifest.json"):
            self.assertTrue((self.out_dir / name).is_file(), name)
        manifest = json.loads(
            (self.out_dir / "replay_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["episodes"], 7)
        self.assertEqual(manifest["scenes"], 2)
        self.assertEqual(manifest["successes"], 1)
        self.assertEqual(manifest["failures"], 6)
        self.assertEqual(manifest["failure_categories"]["stuck_door"], 2)
        self.assertEqual(manifest["failure_categories"]["timeout"], 1)

    def test_invalid_jsonl_line_is_reported_not_fatal(self):
        self.assertTrue(any("invalid JSON" in item for item in self.data["warnings"]))
        self.assertEqual(len(self.data["episodes"]), 7)

    def test_success_episode_metrics_and_path(self):
        item = self.episodes["duplex_a::ep_ok"]
        self.assertTrue(item["success"])
        self.assertEqual(item["failure_category"], "success")
        self.assertEqual(item["reference_path"]["distance"], 2.4)
        self.assertEqual(item["metrics"]["trajectory_distance"], 2.4)
        self.assertEqual(item["metrics"]["final_distance_to_goal"], 0.0)
        self.assertEqual(item["start_floor"], "f0")
        self.assertEqual(item["goal_floor"], "f0")
        self.assertEqual(item["trajectory"][-1]["floor_id"], "f0")
        self.assertEqual(item["yaw_unit"], "degrees")
        self.assertEqual(item["yaw_convention"], "manifest")
        event_types = {event["type"] for event in item["events"]}
        self.assertEqual(event_types, {"start", "goal"})

    def test_stuck_at_door_evidence(self):
        item = self.episodes["duplex_a::ep_stuck"]
        self.assertEqual(item["failure_category"], "stuck_door")
        event_types = [event["type"] for event in item["events"]]
        self.assertIn("stuck_door", event_types)
        self.assertIn("collision", event_types)
        stuck = next(event for event in item["events"] if event["type"] == "stuck_door")
        self.assertLess(stuck["position"][0], 4.0)
        self.assertIn("collisions", stuck["label"])
        self.assertEqual(item["metrics"]["collision_steps"], 6)

    def test_through_wall_evidence(self):
        item = self.episodes["duplex_a::ep_wall"]
        self.assertEqual(item["failure_category"], "through_wall")
        evidence = [event for event in item["events"] if event["type"] == "through_wall"]
        self.assertTrue(evidence)
        crossing = evidence[0]
        self.assertGreater(crossing["position"][0], 3.9)
        self.assertLess(crossing["position"][0], 4.7)
        self.assertIn("door", crossing["label"])

    def test_cross_void_evidence_without_a_reason_string(self):
        item = self.episodes["duplex_a::ep_void"]
        self.assertIsNone(item["failure_reason"])
        self.assertEqual(item["failure_category"], "cross_void")
        evidence = [event for event in item["events"] if event["type"] == "cross_void"]
        self.assertTrue(evidence)
        point = evidence[0]["position"]
        self.assertGreaterEqual(point[0], 2.6)
        self.assertLessEqual(point[0], 3.4)
        self.assertGreaterEqual(point[2], 0.2)
        self.assertLessEqual(point[2], 0.8)
        self.assertTrue(item["geometry_reliable"])
        self.assertGreaterEqual(item["geometry_coverage"], 0.9)

    def test_wrong_floor_jump_evidence(self):
        item = self.episodes["duplex_a::ep_floor"]
        self.assertEqual(item["failure_category"], "floor_jump")
        evidence = [event for event in item["events"] if event["type"] == "floor_jump"]
        self.assertTrue(evidence)
        self.assertIn("f0 -> f1", evidence[0]["label"])
        self.assertAlmostEqual(evidence[0]["position"][1], 1.55, places=2)
        self.assertEqual(item["trajectory"][-1]["floor_id"], "f1")

    def test_missing_trajectory_degrades_to_reference_path(self):
        item = self.episodes["duplex_a::ep_missing_trajectory"]
        self.assertFalse(item["has_trajectory"])
        self.assertEqual(item["trajectory"], [])
        self.assertEqual(item["failure_category"], "timeout")
        self.assertEqual(len(item["reference_path"]["points"]), 2)
        self.assertTrue(any("no trajectory" in warning for warning in item["warnings"]))
        self.assertEqual(item["metrics"]["steps"], 0)

    def test_missing_scene_geometry_degrades_to_blank_grid(self):
        item = self.episodes["missing_scene::ep_missing_scene"]
        self.assertEqual(item["failure_category"], "stuck_door")
        scene = self.data["scenes"]["missing_scene"]
        self.assertEqual(scene["rooms"], [])
        # Only the floor ids referenced by the episode remain, as stubs with no
        # elevation: the floor selector still works without inventing geometry.
        self.assertEqual(
            [floor["id"] for floor in scene["floors"]], ["f0", "f1"]
        )
        self.assertTrue(all(floor["elevation"] is None for floor in scene["floors"]))
        self.assertTrue(any(
            "no scene geometry for missing_scene" in warning
            for warning in self.data["warnings"]
        ))

    def test_computed_metrics_match_the_episodes(self):
        computed = self.data["metrics"]["computed"]
        overall = computed["overall"]
        self.assertEqual(overall["episodes"], 7)
        self.assertEqual(overall["successes"], 1)
        self.assertEqual(overall["failures"], 6)
        self.assertAlmostEqual(overall["success_rate"], 1 / 7, places=4)
        self.assertAlmostEqual(overall["spl_mean"], 0.15, places=4)
        self.assertEqual(overall["collisions"], 8)
        self.assertEqual(overall["timeouts"], 2)
        self.assertEqual(
            overall["failure_categories"],
            {
                "cross_void": 1,
                "floor_jump": 1,
                "stuck_door": 2,
                "success": 1,
                "through_wall": 1,
                "timeout": 1,
            },
        )
        self.assertEqual(computed["by_scene"]["duplex_a"]["episodes"], 6)
        self.assertEqual(computed["by_scene"]["missing_scene"]["episodes"], 1)

    def test_reported_metrics_are_kept_separate(self):
        provided = self.data["metrics"]["provided"]
        self.assertIsNotNone(provided)
        self.assertEqual(provided["overall"]["success_rate"], 0.25)
        self.assertEqual(provided["overall"]["spl"], 0.2)
        self.assertEqual(provided["overall"]["collisions"], 9)
        self.assertTrue(provided["source"].endswith("metrics.json"))
        self.assertEqual(
            self.data["metrics"]["computed"]["overall"]["success_rate"], 0.1429
        )

    def test_floor_ids_are_attached_to_paths(self):
        item = self.episodes["duplex_a::ep_floor"]
        self.assertEqual(item["reference_path"]["floor_ids"], ["f0", "f0", "f1"])
        self.assertEqual(
            [step["floor_id"] for step in item["trajectory"]],
            ["f0", "f0", "f1"],
        )

    def test_page_wiring_and_no_external_cdn(self):
        html = (self.out_dir / "index.html").read_text(encoding="utf-8")
        script = (self.out_dir / "replay.js").read_text(encoding="utf-8")
        stylesheet = (self.out_dir / "replay.css").read_text(encoding="utf-8")
        self.assertIn('src="replay-data.js"', html)
        self.assertIn('src="replay.js"', html)
        self.assertIn('href="replay.css"', html)
        for text in (html, script, stylesheet):
            self.assertNotRegex(text, r'(?:src|href)\s*=\s*["\']https?://')
            self.assertNotIn("cdn.", text.lower())
        ids = set(re.findall(r'id="([^"]+)"', html))
        for element_id in re.findall(r"getElementById\('([^']+)'\)", script):
            self.assertIn(element_id, ids, element_id)

    def test_generated_data_matches_manifest(self):
        manifest = json.loads(
            (self.out_dir / "replay_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["episodes"], len(self.data["episodes"]))
        self.assertEqual(
            manifest["failure_categories"],
            self.data["metrics"]["computed"]["overall"]["failure_categories"],
        )
        self.assertEqual(
            {item["key"] for item in self.data["episodes"]},
            set(self.episodes),
        )

    def test_curl_serves_the_site(self):
        curl = shutil.which("curl")
        if curl is None:
            self.skipTest("curl is not installed")
        handler = partial(QuietHandler, directory=str(self.out_dir))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            base = f"http://127.0.0.1:{port}"
            for name in ("index.html", "replay.js", "replay.css",
                         "replay-data.js", "replay_manifest.json"):
                result = subprocess.run(
                    [
                        curl, "-sS", "--noproxy", "*", "--max-time", "10",
                        "-o", os.devnull, "-w", "%{http_code}",
                        f"{base}/{name}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "200", name)
            payload = subprocess.run(
                [curl, "-sS", "--noproxy", "*", "--max-time", "10",
                 f"{base}/replay-data.js"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(payload.returncode, 0, payload.stderr)
            self.assertIn("window.NAV_REPLAY_DATA", payload.stdout)
            self.assertIn("duplex_a::ep_wall", payload.stdout)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_site_can_be_built_from_an_evaluation_directory(self):
        out_dir = Path(self.tmp.name) / "discovered_site"
        result = subprocess.run(
            [
                sys.executable, BUILD_SCRIPT,
                str(self.eval_dir),
                "--out", str(out_dir),
                "--quiet",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        data = read_data_js(out_dir / "replay-data.js")
        self.assertEqual(len(data["episodes"]), 7)
        self.assertIsNotNone(data["source"]["scenes"])
        self.assertTrue(data["source"]["scenes"].endswith("scenes.json"))
        self.assertTrue(data["source"]["metrics"].endswith("metrics.json"))

    def test_scenes_json_can_point_at_scene_manifests(self):
        with tempfile.TemporaryDirectory(prefix="nav_replay_pointer_") as tmp:
            directory = Path(tmp)
            scene_dir = directory / "scene_ptr" / "build"
            scene_dir.mkdir(parents=True)
            (scene_dir / "scene3d_manifest.json").write_text(
                json.dumps({
                    "kind": "roomgen_scene3d",
                    "floors": [{"id": "f0", "elevation": 0.0}],
                    "rooms": [{
                        "id": "room", "floor_id": "f0",
                        "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]],
                    }],
                }),
                encoding="utf-8",
            )
            (directory / "scenes.json").write_text(
                json.dumps({"scenes": [{
                    "scene_id": "ptr_scene",
                    "scene_dir": str(directory / "scene_ptr"),
                }]}),
                encoding="utf-8",
            )
            item = episode(
                "ptr_ep",
                scene_id="ptr_scene",
                start=[1.0, 0.0, 2.0],
                goal=[3.0, 0.0, 2.0],
                reference_path={
                    "points": [[1.0, 0.0, 2.0], [3.0, 0.0, 2.0]],
                    "distance": 2.0,
                },
                trajectory=[
                    {"position": [1.0, 0.0, 2.0], "yaw": 0.0,
                     "action": 0, "collision": False},
                    {"position": [3.0, 0.0, 2.0], "yaw": 0.0,
                     "action": 1, "collision": False},
                ],
            )
            (directory / "episodes.jsonl").write_text(
                json.dumps(item) + "\n", encoding="utf-8"
            )
            out_dir = directory / "site"
            result = subprocess.run(
                [
                    sys.executable, BUILD_SCRIPT,
                    "--episodes", str(directory / "episodes.jsonl"),
                    "--scenes", str(directory / "scenes.json"),
                    "--out", str(out_dir),
                    "--quiet",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            data = read_data_js(out_dir / "replay-data.js")
            scene = data["scenes"]["ptr_scene"]
            self.assertEqual(len(scene["rooms"]), 1)
            self.assertEqual(scene["rooms"][0]["id"], "room")
            self.assertEqual(scene["rooms"][0]["polygon"],
                             [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])

    def test_run_json_supplies_scene_manifests_and_metrics(self):
        with tempfile.TemporaryDirectory(prefix="nav_replay_runjson_") as tmp:
            directory = Path(tmp)
            build_dir = directory / "scene" / "build"
            build_dir.mkdir(parents=True)
            (build_dir / "scene3d_manifest.json").write_text(
                json.dumps({
                    "floors": [{"id": "f0", "elevation": 0.0}],
                    "rooms": [{
                        "id": "room", "floor_id": "f0",
                        "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]],
                    }],
                }),
                encoding="utf-8",
            )
            (directory / "run.json").write_text(
                json.dumps({
                    "kind": "roomgen_pointnav",
                    "scenes": [{
                        "scene_id": "run_scene",
                        "build_dir": str(build_dir),
                    }],
                    "summary": {
                        "episodes": 1,
                        "success_rate": 1.0,
                        "spl": 1.0,
                        "collisions": 0,
                        "timeouts": 0,
                    },
                    "scene_summaries": {
                        "run_scene": {"success_rate": 1.0, "spl": 1.0},
                    },
                }),
                encoding="utf-8",
            )
            item = episode(
                "run_ep",
                scene_id="run_scene",
                start=[1.0, 0.0, 1.0],
                goal=[3.0, 0.0, 1.0],
                reference_path={
                    "points": [[1.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
                    "distance": 2.0,
                },
                trajectory=[
                    {"position": [1.0, 0.0, 1.0], "yaw": 0.0,
                     "action": "forward", "collision": False},
                    {"position": [3.0, 0.0, 1.0], "yaw": 0.0,
                     "action": "stop", "collision": False},
                ],
            )
            (directory / "episodes.jsonl").write_text(
                json.dumps(item) + "\n", encoding="utf-8"
            )
            out_dir = directory / "site"
            result = subprocess.run(
                [
                    sys.executable, BUILD_SCRIPT,
                    "--episodes", str(directory / "episodes.jsonl"),
                    "--out", str(out_dir),
                    "--quiet",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            data = read_data_js(out_dir / "replay-data.js")
            self.assertEqual(len(data["scenes"]["run_scene"]["rooms"]), 1)
            self.assertTrue(
                data["source"]["scenes"].endswith("run.json"),
                data["source"]["scenes"],
            )
            provided = data["metrics"]["provided"]
            self.assertTrue(provided["source"].endswith("run.json"))
            self.assertEqual(provided["overall"]["success_rate"], 1.0)
            self.assertIn("run_scene", provided["by_scene"])

    def test_node_parses_replay_javascript(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        result = subprocess.run(
            [node, "--check", os.path.join(ASSET_DIR, "replay.js")],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_replay_javascript_renders_the_generated_data(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        harness = Path(self.tmp.name) / "dom_smoke.cjs"
        harness.write_text(DOM_SMOKE_JS, encoding="utf-8")
        result = subprocess.run(
            [node, str(harness), str(self.out_dir)],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("nav replay DOM smoke OK", result.stdout)
        self.assertIn("canvas calls", result.stdout)

    def test_failure_reason_categories(self):
        self.assertEqual(BUILDER.category_from_reason("stuck"), "stuck")
        self.assertEqual(
            BUILDER.category_from_reason("stuck at door"), "stuck_door"
        )
        self.assertEqual(
            BUILDER.category_from_reason("went through wall"), "through_wall"
        )
        self.assertEqual(
            BUILDER.category_from_reason("cross_void"), "cross_void"
        )
        self.assertEqual(
            BUILDER.category_from_reason("wrong floor jump"), "floor_jump"
        )
        self.assertEqual(BUILDER.category_from_reason("timeout"), "timeout")
        self.assertIsNone(BUILDER.category_from_reason(None))

    def test_episode_aliases_and_top_level_actions(self):
        warnings = []
        item = BUILDER.normalize_episode({
            "scene_id": "alias_scene",
            "episode_id": 12,
            "seed": 3,
            "start_position": [0.0, 0.0, 0.0],
            "goal_position": [2.0, 0.0, 0.0],
            "start_floor": "f0",
            "goal_floor": "f0",
            "reference": {
                "points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                "distance": 2.0,
            },
            "observations": [
                {"position": [0.0, 0.0, 0.0], "heading_deg": 0.0,
                 "collision": False},
                {"position": [2.0, 0.0, 0.0], "heading_deg": 0.0,
                 "collision": False},
            ],
            "actions": ["forward", "stop"],
            "success": True,
            "spl": 1.0,
            "collisions": 0,
            "timeout": False,
            "failure_reason": None,
        }, 1, warnings)
        self.assertEqual(item["start"], [0.0, 0.0, 0.0])
        self.assertEqual(item["goal"], [2.0, 0.0, 0.0])
        self.assertEqual(item["reference_path"]["points"],
                         [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertEqual(item["reference_path"]["distance"], 2.0)
        self.assertEqual([step["action"] for step in item["trajectory"]],
                         ["forward", "stop"])
        self.assertEqual(item["yaw_unit"], "degrees")
        self.assertEqual(item["yaw_convention"], "manifest")

    def test_yaw_unit_and_convention_inference(self):
        manifest_degrees = [
            {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
            {"position": [0.0, 0.0, -1.0], "yaw": 90.0},
        ]
        unit, convention, error, samples = BUILDER.infer_yaw_metadata(
            manifest_degrees
        )
        self.assertEqual((unit, convention), ("degrees", "manifest"))
        self.assertEqual(samples, 1)
        self.assertLess(error, 1e-6)

        habitat_radians = [
            {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
            {"position": [-1.0, 0.0, 0.0], "yaw": math.pi / 2.0},
        ]
        unit, convention, error, samples = BUILDER.infer_yaw_metadata(
            habitat_radians
        )
        self.assertEqual((unit, convention), ("radians", "habitat"))
        self.assertEqual(samples, 1)
        self.assertLess(error, 1e-6)

        unit, convention, _, _ = BUILDER.infer_yaw_metadata(
            manifest_degrees, unit="radians", convention="habitat"
        )
        self.assertEqual((unit, convention), ("radians", "habitat"))

    def test_point_normalisation_accepts_evaluator_shapes(self):
        self.assertEqual(BUILDER.position3([1.5, 2.5]), [1.5, 0.0, 2.5])
        self.assertEqual(
            BUILDER.position3({"x": 1.0, "y": 0.0, "z": 3.0}), [1.0, 0.0, 3.0]
        )
        self.assertEqual(
            BUILDER.position3({"position": [1.0, 1.2, -2.0]}), [1.0, 1.2, -2.0]
        )
        self.assertEqual(BUILDER.position3({"x": 1.0, "z": 3.0}), [1.0, 0.0, 3.0])
        self.assertEqual(BUILDER.position3({"x": 1.0, "y": 3.0}), [1.0, 0.0, 3.0])

    def test_blender_z_up_input_is_converted_to_gltf_xz(self):
        with tempfile.TemporaryDirectory(prefix="nav_replay_blender_") as tmp:
            directory = Path(tmp)
            scene = {
                "coordinate_system": "blender",
                "scene_id": "blender_scene",
                "floors": [{"id": "f0", "elevation": 0.0}],
                "rooms": [
                    {"id": "room", "floor_id": "f0",
                     "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
                ],
                "portals": [
                    {"id": "door", "floor_id": "f0", "position": [4, 1.5, 0]},
                ],
            }
            (directory / "scenes.json").write_text(
                json.dumps({"scenes": [scene]}), encoding="utf-8"
            )
            episode_line = episode(
                "blender_ep",
                scene_id="blender_scene",
                start=[0.5, 1.0, 0.0],
                goal=[3.5, 1.0, 2.0],
                reference_path={"points": [[0.5, 1.0, 0.0], [3.5, 1.0, 2.0]],
                                "distance": 3.6},
                trajectory=[
                    {"position": [0.5, 1.0, 0.0], "yaw": 0.0,
                     "action": 0, "collision": False},
                    {"position": [3.5, 1.0, 2.0], "yaw": 0.0,
                     "action": 1, "collision": False},
                ],
            )
            (directory / "episodes.jsonl").write_text(
                json.dumps(episode_line) + "\n", encoding="utf-8"
            )
            out_dir = directory / "site"
            result = subprocess.run(
                [
                    sys.executable, BUILD_SCRIPT,
                    "--episodes", str(directory / "episodes.jsonl"),
                    "--scenes", str(directory / "scenes.json"),
                    "--out", str(out_dir),
                    "--quiet",
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            data = read_data_js(out_dir / "replay-data.js")
            item = data["episodes"][0]
            self.assertEqual(item["start"], [0.5, 0.0, -1.0])
            self.assertEqual(item["goal"], [3.5, 2.0, -1.0])
            room = data["scenes"]["blender_scene"]["rooms"][0]
            self.assertEqual(room["polygon"], [[0.0, 0.0], [4.0, 0.0], [4.0, -3.0], [0.0, -3.0]])
            portal = data["scenes"]["blender_scene"]["portals"][0]
            self.assertEqual(portal["position"], [4.0, -1.5])


if __name__ == "__main__":
    unittest.main()
