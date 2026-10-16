#!/usr/bin/env python3
"""Static server for the RoomGen.Agent browser 3D viewer (``scripts/scene3d_view.sh``).

One scene per server instance, no dependencies beyond the standard library:

    /                tools/scene3d_viewer/index.html
    /viewer.js       /viewer.css        the viewer page's own assets
    /vendor/<path>   three.js r180 + GLTFLoader / OrbitControls / RoomEnvironment
    /scene.glb       the build's GLB (``<scene_dir>/scene.glb`` or ``<scene_dir>/build/scene.glb``)
    /manifest.json   scene3d_manifest.json (cameras, rooms, bounds) + the page's load URLs

The panorama viewer (``PanoWorld/tools/panoworld_viewer_server.py``) was the
model for this handler - same ThreadingHTTPServer + path-guard + no-store
static serving - but its routes are panorama specific (``/media/<view>/<file>``
over a viewpoints tree and a generated page), so the 3D viewer gets its own
small server instead of a fork of that one.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import posixpath
import socket
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from shutil import copyfileobj
from urllib.parse import unquote, urlparse

ASSET_DIR = Path(__file__).resolve().parent
DEFAULT_EYE_HEIGHT = 1.5
GLB_CANDIDATES = ("scene.glb", os.path.join("build", "scene.glb"))


# --------------------------------------------------------------------------- #
# scene discovery
# --------------------------------------------------------------------------- #
def find_scene(scene_dir: Path) -> tuple:
    """Return ``(scene_root, glb_path)`` for a scene or build directory."""
    scene_dir = scene_dir.expanduser().resolve()
    if scene_dir.is_file():
        if scene_dir.suffix.lower() != ".glb":
            raise FileNotFoundError(f"{scene_dir} is not a .glb file")
        return scene_dir.parent, scene_dir
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene directory not found: {scene_dir}")
    for candidate in GLB_CANDIDATES:
        path = scene_dir / candidate
        if path.is_file():
            return scene_dir, path
    raise FileNotFoundError(
        f"no scene.glb under {scene_dir} (tried {', '.join(GLB_CANDIDATES)}).\n"
        f"Export one first, e.g.:\n"
        f"  third_party/blender-4.5.14-linux-x64/blender --background --factory-startup \\\n"
        f"    --python blender_agent/run_build.py -- --spec <spec.json> --out {scene_dir}/build \\\n"
        f"    --skip-render --export-glb"
    )


def _gltf_point(point) -> list:
    """Blender (x, y, z) metres -> glTF (x, z, -y) metres."""
    return [float(point[0]), float(point[2]), -float(point[1])]


def manifest_from_build_manifest(build_manifest: dict) -> dict:
    """Fallback manifest from ``build_manifest.json`` (older builds, no GLB manifest)."""
    spec = build_manifest.get("spec") or {}
    entries = list(build_manifest.get("cameras") or [])
    if not entries:
        # a --skip-render build has no rendered camera stats; the spec dump still
        # carries the camera poses, at yaw 0 (the only yaw the specs use today)
        entries = [{"camera": c.get("name"), "room": c.get("room"), "pos": c.get("pos"),
                    "yaw": 0.0} for c in (spec.get("cameras") or [])]

    cameras = []
    for entry in entries:
        pos = entry.get("pos")
        if not pos:
            continue
        yaw = float(entry.get("yaw", 0.0) or 0.0)  # radians, as written by run_build.py
        cameras.append({
            "name": entry.get("camera") or entry.get("name"),
            "room": entry.get("room"),
            "position": _gltf_point(pos),
            "yaw_deg": math.degrees(yaw) if abs(yaw) <= 2 * math.pi else yaw,
            "forward": [math.cos(yaw), 0.0, -math.sin(yaw)],
        })

    rooms = []
    for room in spec.get("rooms", []) or []:
        polygon = room.get("polygon") or []
        if len(polygon) < 3:
            continue
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        height = float(spec.get("wall_height", 2.7))
        camera = next((cam["name"] for cam in cameras if cam.get("room") == room.get("id")), None)
        rooms.append({
            "id": room.get("id"),
            "type": room.get("type"),
            "camera": camera,
            "center": _gltf_point(((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, DEFAULT_EYE_HEIGHT)),
            "size": [max(xs) - min(xs), max(ys) - min(ys), height],
        })
    return {"cameras": cameras, "rooms": rooms, "bounds": None, "stats": None}


def build_payload(scene_root: Path, glb_path: Path) -> dict:
    """The viewer page's /manifest.json: scene metadata + what to load where."""
    manifest = None
    source = None
    glb_manifest_path = glb_path.parent / "scene3d_manifest.json"
    build_manifest_path = glb_path.parent / "build_manifest.json"

    if glb_manifest_path.is_file():
        with glb_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        source = str(glb_manifest_path)
    elif build_manifest_path.is_file():
        with build_manifest_path.open("r", encoding="utf-8") as handle:
            build_manifest = json.load(handle)
        manifest = manifest_from_build_manifest(build_manifest)
        source = f"{build_manifest_path} (cameras converted to glTF coordinates)"
        print("[scene3d_view] no scene3d_manifest.json; falling back to build_manifest.json")

    payload = {
        "title": scene_root.name or "RoomGen.Agent scene",
        "glb": "/scene.glb",
        "glb_bytes": glb_path.stat().st_size,
        "scene_dir": str(scene_root),
        "glb_path": str(glb_path),
        "manifest_source": source,
        "eye_height": DEFAULT_EYE_HEIGHT,
        "cameras": [],
        "rooms": [],
        "bounds": None,
        "stats": None,
    }
    if manifest:
        payload.update({
            "title": manifest.get("title") or payload["title"],
            "eye_height": manifest.get("eye_height", DEFAULT_EYE_HEIGHT),
            "cameras": manifest.get("cameras", []),
            "rooms": manifest.get("rooms", []),
            "bounds": manifest.get("bounds"),
            "stats": manifest.get("stats"),
            "blender_version": manifest.get("blender_version"),
        })
    return payload


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "RoomGenScene3DViewer/1.0"
    protocol_version = "HTTP/1.1"

    def do_HEAD(self) -> None:  # noqa: N802
        self.route(head_only=True)

    def do_GET(self) -> None:  # noqa: N802
        self.route(head_only=False)

    def route(self, head_only: bool) -> None:
        route = posixpath.normpath(unquote(urlparse(self.path).path))
        if route in ("", "/", "/index.html"):
            self.send_file(ASSET_DIR / "index.html", head_only, cache="no-store")
            return
        if route == "/manifest.json":
            payload = json.dumps(self.server.payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_bytes(payload, "application/json; charset=utf-8", head_only)
            return
        if route == "/scene.glb":
            self.send_file(self.server.glb_path, head_only, cache="no-store")
            return
        if route in ("/viewer.js", "/viewer.css"):
            self.send_file(ASSET_DIR / route.lstrip("/"), head_only, cache="no-store")
            return
        if route.startswith("/vendor/"):
            self.serve_vendor(route[len("/vendor/"):], head_only)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def serve_vendor(self, relative: str, head_only: bool) -> None:
        parts = [part for part in relative.split("/") if part]
        if not parts or any(part in (".", "..") for part in parts):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid vendor path")
            return
        path = (ASSET_DIR / "vendor" / Path(*parts)).resolve()
        try:
            path.relative_to((ASSET_DIR / "vendor").resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Path escapes the vendor directory")
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Vendor asset not found")
            return
        self.send_file(path, head_only, cache="no-store")

    def content_type(self, path: Path) -> str:
        if path.suffix == ".js":
            return "text/javascript"
        if path.suffix == ".glb":
            return "model/gltf-binary"
        return mimetypes.guess_type(str(path))[0] or "application/octet-stream"

    def send_file(self, path: Path, head_only: bool, cache: str = "no-store") -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"{path.name} not found")
            return
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", self.content_type(path))
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as handle:
            copyfileobj(handle, self.wfile, length=1024 * 256)

    def send_bytes(self, payload: bytes, content_type: str, head_only: bool) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        try:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
        except BrokenPipeError:
            pass


def access_hosts() -> list:
    hosts = ["127.0.0.1"]
    try:
        for addr in socket.gethostbyname_ex(socket.gethostname())[2]:
            if addr not in hosts and not addr.startswith("127."):
                hosts.append(addr)
    except OSError:
        pass
    return hosts


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the RoomGen.Agent 3D scene viewer.")
    parser.add_argument("scene_dir", help="build/scene directory that holds scene.glb")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8004)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        scene_root, glb_path = find_scene(Path(args.scene_dir))
    except FileNotFoundError as error:
        print(f"[scene3d_view] {error}", file=sys.stderr)
        return 1

    payload = build_payload(scene_root, glb_path)

    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.glb_path = glb_path
    server.payload = payload

    stats = payload.get("stats") or {}
    print(f"[scene3d_view] serving {glb_path.name} ({glb_path.stat().st_size / 1e6:.1f} MB) from {scene_root}")
    if stats:
        print(f"[scene3d_view] {stats.get('mesh_nodes', '?')} objects, "
              f"{stats.get('triangles_rendered', '?')} triangles, {stats.get('materials', '?')} materials")
    print(f"[scene3d_view] {len(payload['cameras'])} viewpoints: "
          f"{', '.join(str(cam.get('name')) for cam in payload['cameras']) or 'none'}")
    print("[scene3d_view] Open one of these URLs from your local browser:")
    for host in access_hosts():
        print(f"  http://{host}:{args.port}/")
    print("[scene3d_view] Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[scene3d_view] stopping viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
