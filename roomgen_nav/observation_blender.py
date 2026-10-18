"""Perspective RGB-D observation backend for PointNav (Blender / Cycles CPU).

Habitat-Sim on this machine can build NavMeshes but cannot render (no EGL), so
the observation stream for the PointNav loop comes from Blender instead.  The
backend loads an existing ``scene.glb`` (the same file the viewer/Habitat
NavMesh uses) once, then renders 128x128 perspective frames on demand:

    from roomgen_nav.observation_blender import (
        ObservationBackend, env_observation_provider)

    # Default yaw convention is "manifest": compass heading, 0 = +X, +90 = -Z,
    # the same yaw as scene3d_manifest.json and roomgen_nav.env.  Pass
    # yaw_convention="habitat" for Habitat-Sim's 0 = -Z agent frame.
    backend = ObservationBackend("outputs/viewer_walk_duplex", frame_dir="frames")
    env.set_observation_provider(env_observation_provider(backend))  # optional
    obs = backend.reset(position=(2.4, 1.5, -3.15), yaw=0.0)   # saved as reset_*
    ...
    obs = backend.final(position=(9.24, 4.4, -0.9), yaw=0.0)   # saved as final_*
    obs.rgb       # (H, W, 3) uint8
    obs.depth     # (H, W) float32, z-depth in metres, 0 = no hit
    obs.distance  # (H, W) float32, Euclidean ray length in metres, 0 = no hit

The module-level ``get_observation(scene_dir, position, yaw, pitch=0,
resolution=(128, 128))`` is the one-shot entry point; it keeps one Blender
worker per scene alive between calls (``close_backends()`` shuts them down).

Coordinate contract
-------------------
* Poses are given in the **glTF / Habitat frame**: metres, +Y up.  This is the
  frame of ``scene.glb``, ``scene3d_manifest.json`` and Habitat-Sim.
* ``position`` is the camera centre in that frame.
* ``yaw`` / ``pitch`` are **degrees**.  ``pitch > 0`` looks up.
* ``yaw_convention="manifest"`` (default) matches ``scene3d_manifest.json``,
  ``roomgen_nav.env`` and ``blender_agent.export_glb``: a compass heading in
  the XZ plane, yaw 0 looks along ``+X`` and +90 along ``-Z``.
  ``yaw_convention="habitat"`` follows Habitat-Sim's agent frame: yaw 0 looks
  along ``-Z`` and +90 along ``-X``.
* Internally Blender is Z-up metres with ``p_blender = (x, -z, y)``, the inverse
  of the exporter's ``p_gltf = (x, z, -y)``.

Camera model
------------
Pinhole, no distortion, OpenGL/Habitat camera axes (+X right, +Y up, -Z
forward).  ``hfov_deg`` defaults to 90 degrees; square pixels give
``fx = fy = 0.5 * width / tan(hfov/2)`` (64 px for a 128 px wide 90-degree
frame).  The principal point is ``(width/2, height/2)`` in continuous pixel
coordinates, and pixel ``(u, v)`` samples ``(u + 0.5, v + 0.5)``.

Depth is ray-cast at pixel centres against the evaluated Blender scene, so it
is exact geometry (not a filtered render pass).  ``depth`` is the component of
the hit offset along the camera forward axis (z-depth, matching a Z pass);
``distance`` is the Euclidean ray length.  Both are float32 metres and use
``0`` for "no hit" (the Habitat convention).  ``extrinsics`` is the 4x4
camera-to-world matrix in the glTF/Habitat frame (``p_world = E @ p_camera``);
``intrinsics`` is the 3x3 pinhole K.

Rendering uses Cycles CPU at ``samples=8`` by default (allowed 1-64; 8-16 is
the intended range) and never needs EGL/X.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "render_observation.py")

DEFAULT_HFOV_DEG = 90.0
DEFAULT_SAMPLES = 8
DEFAULT_RESOLUTION = (128, 128)
DEFAULT_YAW_CONVENTION = "manifest"
YAW_CONVENTIONS = ("habitat", "manifest")

SCENE_CANDIDATES = (
    "scene.glb",
    os.path.join("build", "scene.glb"),
    os.path.join("scene", "scene.glb"),
    "scene.blend",
    os.path.join("build", "scene.blend"),
    os.path.join("scene", "scene.blend"),
)

DEPTH_CONVENTION = ("z-depth along the camera forward axis, float32 metres, "
                    "0 = no hit")
DISTANCE_CONVENTION = ("Euclidean distance from the camera centre to the hit, "
                       "float32 metres, 0 = no hit")


class ObservationError(RuntimeError):
    """The Blender worker failed, timed out or returned an invalid frame."""


# --------------------------------------------------------------------------- #
# coordinate / camera math (kept in sync with render_observation.py)
# --------------------------------------------------------------------------- #
def gltf_to_blender(point: Sequence[float]) -> np.ndarray:
    """glTF / Habitat (x, y, z) metres -> Blender (x, -z, y)."""
    x, y, z = (float(value) for value in point)
    return np.array([x, -z, y], dtype=np.float64)


def blender_to_gltf(point: Sequence[float]) -> np.ndarray:
    """Blender (x, y, z) metres -> glTF / Habitat (x, z, -y)."""
    x, y, z = (float(value) for value in point)
    return np.array([x, z, -y], dtype=np.float64)


def camera_basis(yaw_deg: float, pitch_deg: float,
                 convention: str = DEFAULT_YAW_CONVENTION):
    """Return ``(forward, right, up)`` unit vectors in glTF / Habitat coords."""
    if convention not in YAW_CONVENTIONS:
        raise ValueError(f"yaw_convention must be one of {YAW_CONVENTIONS}, "
                         f"got {convention!r}")
    yaw = np.radians(float(yaw_deg))
    pitch = np.radians(float(pitch_deg))
    if convention == "manifest":
        forward = np.array([np.cos(yaw) * np.cos(pitch),
                            np.sin(pitch),
                            -np.sin(yaw) * np.cos(pitch)])
        right = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    else:  # habitat
        forward = np.array([-np.sin(yaw) * np.cos(pitch),
                            np.sin(pitch),
                            -np.cos(yaw) * np.cos(pitch)])
        right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    forward = forward / np.linalg.norm(forward)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return forward, right, up


def camera_to_world(position: Sequence[float], yaw_deg: float, pitch_deg: float = 0.0,
                    convention: str = DEFAULT_YAW_CONVENTION) -> np.ndarray:
    """4x4 OpenGL-style camera-to-world matrix in glTF / Habitat coordinates."""
    forward, right, up = camera_basis(yaw_deg, pitch_deg, convention)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = [float(value) for value in position]
    return matrix


def pinhole_intrinsics(width: int, height: int,
                       hfov_deg: float = DEFAULT_HFOV_DEG) -> np.ndarray:
    """3x3 pinhole K; pixel centres at (u + 0.5, v + 0.5)."""
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError(f"resolution must be positive, got {width}x{height}")
    if not 0.0 < float(hfov_deg) < 180.0:
        raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg}")
    fx = 0.5 * float(width) / np.tan(np.radians(float(hfov_deg)) / 2.0)
    fy = fx
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# tool / asset discovery
# --------------------------------------------------------------------------- #
def find_blender(explicit: Optional[str] = None) -> str:
    """Locate the headless Blender binary (explicit > env > repo > PATH)."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("ROOMGEN_BLENDER")
    if env:
        candidates.append(env)
    candidates.extend((
        os.path.join(_REPO_ROOT, "third_party", "blender-4.5.14-linux-x64", "blender"),
        shutil.which("blender"),
    ))
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(str(candidate)))
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "Blender not found. Set ROOMGEN_BLENDER or install "
        "third_party/blender-4.5.14-linux-x64/blender")


def find_scene_asset(scene_dir) -> str:
    """Resolve a scene directory / GLB / .blend path to an importable asset.

    Accepted layouts, in order: the path is a ``.glb``/``.gltf``/``.blend``
    file; or a directory containing ``scene.glb``, ``build/scene.glb``,
    ``scene/scene.glb`` (or the ``.blend`` equivalents); or a directory with
    exactly one top-level ``*.glb`` / ``*.blend``.
    """
    path = os.path.abspath(os.path.expanduser(str(scene_dir)))
    if os.path.isfile(path):
        suffix = os.path.splitext(path)[1].lower()
        if suffix in (".glb", ".gltf", ".blend"):
            return path
        raise ValueError(f"unsupported scene asset {path!r} (expected .glb/.gltf/.blend)")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"scene path does not exist: {path}")
    for relative in SCENE_CANDIDATES:
        candidate = os.path.join(path, relative)
        if os.path.isfile(candidate):
            return candidate
    for suffix in (".glb", ".blend"):
        matches = sorted(name for name in os.listdir(path)
                         if name.lower().endswith(suffix)
                         and os.path.isfile(os.path.join(path, name)))
        if len(matches) == 1:
            return os.path.join(path, matches[0])
    tried = ", ".join(SCENE_CANDIDATES)
    raise FileNotFoundError(
        f"no scene asset under {path} (tried {tried}, or one top-level .glb/.blend)")


def _load_scene_manifest(asset_path: str) -> Optional[dict]:
    manifest_path = os.path.join(os.path.dirname(asset_path), "scene3d_manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# observation container
# --------------------------------------------------------------------------- #
@dataclass
class Observation:
    """One perspective RGB-D frame and the pose/camera that produced it."""

    rgb: np.ndarray                 # (H, W, 3) uint8
    depth: np.ndarray               # (H, W) float32 z-depth metres, 0 = no hit
    distance: np.ndarray            # (H, W) float32 Euclidean metres, 0 = no hit
    position: Tuple[float, float, float]
    yaw: float
    pitch: float
    yaw_convention: str
    resolution: Tuple[int, int]
    intrinsics: np.ndarray          # 3x3 float64
    extrinsics: np.ndarray          # 4x4 float64 camera-to-world (glTF/Habitat)
    metadata: dict = field(default_factory=dict)
    saved_paths: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        self.rgb = np.asarray(self.rgb, dtype=np.uint8)
        self.depth = np.asarray(self.depth, dtype=np.float32)
        self.distance = np.asarray(self.distance, dtype=np.float32)
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        self.extrinsics = np.asarray(self.extrinsics, dtype=np.float64)
        height, width = self.resolution[1], self.resolution[0]
        if self.rgb.shape != (height, width, 3):
            raise ValueError(f"rgb shape {self.rgb.shape} != {(height, width, 3)}")
        if self.depth.shape != (height, width):
            raise ValueError(f"depth shape {self.depth.shape} != {(height, width)}")
        if self.distance.shape != (height, width):
            raise ValueError(f"distance shape {self.distance.shape} != {(height, width)}")
        if self.intrinsics.shape != (3, 3) or self.extrinsics.shape != (4, 4):
            raise ValueError("intrinsics/extrinsics have the wrong shape")

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask of pixels that hit geometry."""
        return self.depth > 0.0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.resolution

    def as_sensor_dict(self) -> dict:
        """Dict form for ``PointNavEnv.set_observation_provider``.

        ``roomgen_nav.env`` merges a provider's dict into ``obs['sensor']`` and
        copies ``rgb`` / ``depth`` up to the observation, so these keys are the
        ones that matter; the pose and camera matrices ride along for replay.
        """
        return {
            "provider": "blender_cycles",
            "rgb": self.rgb,
            "depth": self.depth,
            "distance": self.distance,
            "position": self.position,
            "yaw_deg": self.yaw,
            "pitch_deg": self.pitch,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
        }

    def save(self, prefix: str) -> Dict[str, str]:
        """Write ``<prefix>_rgb.png``, ``_depth.npy``, ``_distance.npy``, ``_meta.json``."""
        from PIL import Image

        prefix = os.path.abspath(os.path.expanduser(str(prefix)))
        parent = os.path.dirname(prefix)
        if parent:
            os.makedirs(parent, exist_ok=True)
        paths = {
            "rgb": prefix + "_rgb.png",
            "depth": prefix + "_depth.npy",
            "distance": prefix + "_distance.npy",
            "meta": prefix + "_meta.json",
        }
        Image.fromarray(self.rgb, mode="RGB").save(paths["rgb"])
        np.save(paths["depth"], self.depth.astype(np.float32, copy=False))
        np.save(paths["distance"], self.distance.astype(np.float32, copy=False))
        payload = {
            "kind": "roomgen_nav_observation",
            "position_gltf": [float(value) for value in self.position],
            "yaw_deg": float(self.yaw),
            "pitch_deg": float(self.pitch),
            "yaw_convention": self.yaw_convention,
            "resolution": [int(self.resolution[0]), int(self.resolution[1])],
            "intrinsics": self.intrinsics.tolist(),
            "extrinsics": self.extrinsics.tolist(),
            "camera_axes": "OpenGL/Habitat: +X right, +Y up, -Z forward; "
                           "p_world = extrinsics @ p_camera",
            "depth_convention": DEPTH_CONVENTION,
            "distance_convention": DISTANCE_CONVENTION,
            "pixel_centres": "pixel (u, v) samples (u + 0.5, v + 0.5)",
            "metadata": self.metadata,
        }
        with open(paths["meta"], "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        return paths


# --------------------------------------------------------------------------- #
# persistent Blender worker
# --------------------------------------------------------------------------- #
def _atomic_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
    os.replace(tmp, path)


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


class ObservationBackend:
    """A persistent headless-Blender renderer for one scene asset.

    Construction imports the GLB/.blend once and blocks until the worker is
    ready.  Every :meth:`get_observation` then only renders one frame, which is
    what makes a 50-task PointNav episode practical.  Always call :meth:`close`
    (or use the object as a context manager).
    """

    def __init__(self, scene_dir, *, blender: Optional[str] = None,
                 samples: int = DEFAULT_SAMPLES,
                 hfov_deg: float = DEFAULT_HFOV_DEG,
                 yaw_convention: str = DEFAULT_YAW_CONVENTION,
                 frame_dir: Optional[str] = None,
                 timeout: Optional[float] = None,
                 ready_timeout: Optional[float] = None,
                 keep_work_dir: bool = False):
        if yaw_convention not in YAW_CONVENTIONS:
            raise ValueError(f"yaw_convention must be one of {YAW_CONVENTIONS}")
        if not 1 <= int(samples) <= 64:
            raise ValueError(f"samples must be in 1..64, got {samples}")
        if not 0.0 < float(hfov_deg) < 180.0:
            raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg}")
        self.scene_dir = os.path.abspath(os.path.expanduser(str(scene_dir)))
        self.scene_asset = find_scene_asset(self.scene_dir)
        self.blender = find_blender(blender)
        self.samples = int(samples)
        self.hfov_deg = float(hfov_deg)
        self.yaw_convention = yaw_convention
        self.frame_dir = (os.path.abspath(os.path.expanduser(str(frame_dir)))
                          if frame_dir else None)
        self.timeout = float(timeout if timeout is not None
                             else os.environ.get("ROOMGEN_OBS_TIMEOUT", "180"))
        self.ready_timeout = float(ready_timeout if ready_timeout is not None
                                   else os.environ.get("ROOMGEN_OBS_READY_TIMEOUT", "300"))
        self.keep_work_dir = bool(keep_work_dir)
        self.scene_manifest = _load_scene_manifest(self.scene_asset)
        self.ready: Optional[dict] = None

        self._work_dir = tempfile.mkdtemp(prefix="roomgen_obs_")
        self._request_id = 0
        self._lock = threading.Lock()
        self._log_handle = None
        self._process: Optional[subprocess.Popen] = None
        self._start()

    # -- lifecycle ---------------------------------------------------------
    def _start(self) -> None:
        command = [
            self.blender,
            "--background",
            "--factory-startup",
            "--python-exit-code", "1",
            "--python", _RENDER_SCRIPT,
            "--",
            "--worker", self._work_dir,
            "--scene", self.scene_asset,
            "--samples", str(self.samples),
            "--hfov-deg", str(self.hfov_deg),
            "--yaw-convention", self.yaw_convention,
            "--parent-pid", str(os.getpid()),
        ]
        log_path = os.path.join(self._work_dir, "blender.log")
        self._log_handle = open(log_path, "w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL,
                stdout=self._log_handle, stderr=subprocess.STDOUT,
                cwd=_REPO_ROOT, start_new_session=True,
            )
        except OSError as error:
            self._log_handle.close()
            self._log_handle = None
            shutil.rmtree(self._work_dir, ignore_errors=True)
            raise ObservationError(f"could not start Blender: {error}") from error

        ready_path = os.path.join(self._work_dir, "ready.json")
        deadline = time.time() + self.ready_timeout
        while True:
            payload = _read_json(ready_path)
            if payload is not None:
                if not payload.get("ok"):
                    error = payload.get("error", "unknown Blender startup error")
                    tail = self._log_tail()
                    self._stop_process()
                    raise ObservationError(
                        f"Blender failed to load {self.scene_asset}:\n{error}\n{tail}")
                self.ready = payload
                break
            if self._process.poll() is not None:
                code = self._process.returncode
                tail = self._log_tail()
                self._stop_process()
                raise ObservationError(
                    f"Blender worker exited with code {code} before ready:\n{tail}")
            if time.time() > deadline:
                tail = self._log_tail()
                self._stop_process()
                raise ObservationError(
                    f"Blender worker did not become ready within "
                    f"{self.ready_timeout:.0f}s:\n{tail}")
            time.sleep(0.02)

    @property
    def alive(self) -> bool:
        """True while the Blender worker process is running."""
        return self._process is not None and self._process.poll() is None

    def _log_tail(self, lines: int = 40) -> str:
        path = os.path.join(self._work_dir, "blender.log")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.readlines()
        except OSError:
            return ""
        return "".join(content[-lines:])

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.poll() is None:
                try:
                    _atomic_json(os.path.join(self._work_dir, "stop"), {"stop": True})
                except OSError:
                    pass
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None
        if not self.keep_work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def close(self) -> None:
        self._stop_process()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self._stop_process()
        except Exception:
            pass

    # -- frames ------------------------------------------------------------
    def get_observation(self, position, yaw, pitch: float = 0.0,
                        resolution: Tuple[int, int] = DEFAULT_RESOLUTION) -> Observation:
        """Render one perspective RGB-D frame at a glTF/Habitat pose.

        ``position`` is the camera centre in metres (+Y up); ``yaw``/``pitch``
        are degrees and follow ``yaw_convention`` (default manifest: yaw 0
        looks along +X).  Returns an :class:`Observation` with uint8 RGB,
        float32 z-depth and Euclidean distance arrays.
        """
        width, height = int(resolution[0]), int(resolution[1])
        if width < 1 or height < 1:
            raise ValueError(f"resolution must be positive, got {resolution!r}")
        position = tuple(float(value) for value in position)
        if len(position) != 3:
            raise ValueError(f"position must have three components, got {position!r}")
        yaw = float(yaw)
        pitch = float(pitch)
        if not (np.all(np.isfinite(position)) and np.isfinite(yaw) and np.isfinite(pitch)):
            raise ValueError("position/yaw/pitch must be finite")
        if abs(pitch) >= 90.0:
            raise ValueError("pitch must be within (-90, 90) degrees")

        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise ObservationError("Blender worker is not running")
            self._request_id += 1
            request_id = self._request_id
            frame_dir = os.path.join(self._work_dir, "frames", f"{request_id:06d}")
            os.makedirs(frame_dir, exist_ok=True)
            request = {
                "id": request_id,
                "position": list(position),
                "yaw": float(yaw),
                "pitch": pitch,
                "resolution": [width, height],
                "hfov_deg": self.hfov_deg,
                "samples": self.samples,
                "yaw_convention": self.yaw_convention,
                "rgb": os.path.join(frame_dir, "rgb.png"),
                "depth": os.path.join(frame_dir, "depth.npy"),
                "distance": os.path.join(frame_dir, "distance.npy"),
                "meta": os.path.join(frame_dir, "meta.json"),
            }
            response_path = os.path.join(self._work_dir, "response.json")
            try:
                os.remove(response_path)
            except OSError:
                pass
            _atomic_json(os.path.join(self._work_dir, "request.json"), request)
            response = self._wait_response(response_path, request_id)
            paths = {
                "rgb": request["rgb"],
                "depth": request["depth"],
                "distance": request["distance"],
                "meta": request["meta"],
            }
            return self._load_observation(response, paths, position, yaw, pitch,
                                          (width, height))

    def _wait_response(self, response_path: str, request_id: int) -> dict:
        deadline = time.time() + self.timeout
        while True:
            response = _read_json(response_path)
            if response is not None:
                if response.get("request_id") not in (None, request_id):
                    time.sleep(0.01)
                    continue
                if not response.get("ok"):
                    raise ObservationError(
                        f"Blender failed to render frame {request_id}:\n"
                        f"{response.get('error', 'unknown error')}")
                return response
            if self._process is None or self._process.poll() is not None:
                code = self._process.returncode if self._process is not None else "?"
                raise ObservationError(
                    f"Blender worker died while rendering frame {request_id} "
                    f"(exit {code}):\n{self._log_tail()}")
            if time.time() > deadline:
                raise ObservationError(
                    f"Blender frame {request_id} timed out after {self.timeout:.0f}s:\n"
                    f"{self._log_tail()}")
            time.sleep(0.02)

    def _load_observation(self, response: dict, paths: dict, position, yaw, pitch,
                          resolution) -> Observation:
        from PIL import Image

        metadata = response.get("metadata") or {}
        try:
            with Image.open(paths["rgb"]) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            depth = np.load(paths["depth"], allow_pickle=False).astype(np.float32, copy=False)
            distance = np.load(paths["distance"], allow_pickle=False).astype(
                np.float32, copy=False)
        except (OSError, ValueError) as error:
            raise ObservationError(f"could not read rendered frame: {error}") from error

        intrinsics = np.asarray(metadata.get("intrinsics"), dtype=np.float64)
        extrinsics = np.asarray(metadata.get("extrinsics"), dtype=np.float64)
        expected_k = pinhole_intrinsics(resolution[0], resolution[1], self.hfov_deg)
        expected_e = camera_to_world(position, yaw, pitch, self.yaw_convention)
        if intrinsics.shape != (3, 3) or not np.allclose(intrinsics, expected_k, atol=1e-6):
            raise ObservationError(
                f"camera intrinsics disagree with the pinhole contract:\n"
                f"worker={intrinsics}\nhost={expected_k}")
        if extrinsics.shape != (4, 4) or not np.allclose(extrinsics, expected_e, atol=1e-5):
            raise ObservationError(
                f"camera extrinsics disagree with the requested pose:\n"
                f"worker={extrinsics}\nhost={expected_e}")

        return Observation(
            rgb=rgb, depth=depth, distance=distance,
            position=position, yaw=float(yaw), pitch=float(pitch),
            yaw_convention=self.yaw_convention,
            resolution=resolution,
            intrinsics=intrinsics, extrinsics=extrinsics,
            metadata=metadata,
        )

    # -- reset / final capture --------------------------------------------
    def save_frame(self, observation: Observation, tag: str) -> Dict[str, str]:
        """Save an observation as ``<frame_dir>/<tag>_rgb.png`` and friends."""
        if self.frame_dir is None:
            raise ValueError("ObservationBackend was created without frame_dir")
        return observation.save(os.path.join(self.frame_dir, tag))

    def _capture(self, tag: str, position, yaw, pitch: float,
                 resolution: Tuple[int, int]) -> Observation:
        observation = self.get_observation(position, yaw, pitch, resolution)
        if self.frame_dir is not None:
            observation.saved_paths = self.save_frame(observation, tag)
        return observation

    def reset(self, position, yaw, pitch: float = 0.0,
              resolution: Tuple[int, int] = DEFAULT_RESOLUTION) -> Observation:
        """Render the episode's reset frame; saved as ``reset_*`` if frame_dir is set."""
        return self._capture("reset", position, yaw, pitch, resolution)

    def final(self, position, yaw, pitch: float = 0.0,
              resolution: Tuple[int, int] = DEFAULT_RESOLUTION) -> Observation:
        """Render the episode's final frame; saved as ``final_*`` if frame_dir is set."""
        return self._capture("final", position, yaw, pitch, resolution)


# --------------------------------------------------------------------------- #
# module-level one-shot API with a worker cache
# --------------------------------------------------------------------------- #
def env_observation_provider(backend: "ObservationBackend", *,
                             pitch: float = 0.0,
                             resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
                             eye_height: float = 1.5):
    """Adapter for ``roomgen_nav.env.PointNavEnv.set_observation_provider``.

    ``PointNavEnv.position`` is the body on the walking surface (the NavMesh /
    floor y), while observations are taken from the camera.  The provider adds
    ``eye_height`` metres to y, matching ``scene3d_manifest.json``'s
    ``eye_height`` of 1.5; on an upper storey the floor elevation is already in
    ``env.position[1]``, so the result lands on 4.4 m, exactly where the
    manifest cameras sit.

    ``PointNavEnv.yaw_deg`` is the same compass heading as the backend's
    default ``"manifest"`` convention (0 = +X, 90 = -Z), so the provider passes
    it through unchanged.  Each call renders one frame; the returned dict has
    ``rgb`` / ``depth`` / ``distance`` plus the pose and camera matrices.
    """

    def provider(env):
        position = (
            float(env.position[0]),
            float(env.position[1]) + float(eye_height),
            float(env.position[2]),
        )
        observation = backend.get_observation(
            position, float(env.yaw_deg), pitch, resolution)
        return observation.as_sensor_dict()

    return provider


_BACKENDS: Dict[tuple, ObservationBackend] = {}
_BACKENDS_LOCK = threading.Lock()
_MAX_CACHED_BACKENDS = max(1, int(os.environ.get("ROOMGEN_OBS_CACHE", "2")))


def _backend_key(scene_dir, blender, samples, hfov_deg, yaw_convention) -> tuple:
    asset = find_scene_asset(scene_dir)
    binary = find_blender(blender)
    return (asset, binary, int(samples), float(hfov_deg), str(yaw_convention))


def get_observation(scene_dir, position, yaw, pitch: float = 0.0,
                    resolution: Tuple[int, int] = DEFAULT_RESOLUTION, *,
                    yaw_convention: str = DEFAULT_YAW_CONVENTION,
                    hfov_deg: float = DEFAULT_HFOV_DEG,
                    samples: int = DEFAULT_SAMPLES,
                    blender: Optional[str] = None,
                    timeout: Optional[float] = None) -> Observation:
    """One-shot perspective RGB-D frame from ``scene_dir``.

    This is the requested ``observation_backend.get_observation`` entry point.
    Workers are cached per (scene asset, Blender binary, samples, hfov, yaw
    convention), so a PointNav loop that calls this repeatedly pays the GLB
    import once per scene.  The cache is LRU-bounded (default two scenes, set
    ``ROOMGEN_OBS_CACHE`` to change it) so a 20-scene sweep cannot keep twenty
    Blender processes inside the 8 GiB cgroup.  Call :func:`close_backends`
    when the episode or sweep ends.
    """
    key = _backend_key(scene_dir, blender, samples, hfov_deg, yaw_convention)
    with _BACKENDS_LOCK:
        backend = _BACKENDS.pop(key, None)
        if backend is not None and not backend.alive:
            backend.close()
            backend = None
        if backend is None:
            while len(_BACKENDS) >= _MAX_CACHED_BACKENDS:
                _old_key, old_backend = next(iter(_BACKENDS.items()))
                del _BACKENDS[_old_key]
                old_backend.close()
            backend = ObservationBackend(
                scene_dir, blender=blender, samples=samples, hfov_deg=hfov_deg,
                yaw_convention=yaw_convention, timeout=timeout)
        _BACKENDS[key] = backend
    return backend.get_observation(position, yaw, pitch, resolution)


def close_backends() -> None:
    """Shut down every cached Blender worker."""
    with _BACKENDS_LOCK:
        backends = list(_BACKENDS.values())
        _BACKENDS.clear()
    for backend in backends:
        backend.close()


atexit.register(close_backends)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render one perspective RGB-D frame with headless Blender")
    parser.add_argument("scene_dir", help="output dir, build dir, .glb or .blend")
    parser.add_argument("--position", type=float, nargs=3, required=True,
                        metavar=("X", "Y", "Z"), help="camera centre, glTF/Habitat metres")
    parser.add_argument("--yaw", type=float, default=0.0, help="degrees about +Y")
    parser.add_argument("--pitch", type=float, default=0.0, help="degrees, positive up")
    parser.add_argument("--resolution", type=int, nargs=2, default=list(DEFAULT_RESOLUTION))
    parser.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG,
                        help="horizontal field of view in degrees")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--yaw-convention", default=DEFAULT_YAW_CONVENTION,
                        choices=YAW_CONVENTIONS)
    parser.add_argument("--out", required=True, help="output prefix (no extension)")
    parser.add_argument("--blender", default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args(argv)

    with ObservationBackend(
            args.scene_dir, blender=args.blender, samples=args.samples,
            hfov_deg=args.hfov, yaw_convention=args.yaw_convention,
            keep_work_dir=args.keep_work_dir) as backend:
        observation = backend.get_observation(
            tuple(args.position), args.yaw, args.pitch, tuple(args.resolution))
        paths = observation.save(args.out)
    print(f"scene: {backend.scene_asset}")
    print(f"rgb: {paths['rgb']} ({observation.rgb.shape} {observation.rgb.dtype})")
    print(f"depth: {paths['depth']} ({observation.depth.shape} {observation.depth.dtype}) "
          f"valid={observation.valid.mean():.1%} "
          f"range=[{observation.depth[observation.valid].min():.3f}, "
          f"{observation.depth.max():.3f}] m")
    print(f"intrinsics fx={observation.intrinsics[0, 0]:.3f} "
          f"cx={observation.intrinsics[0, 2]:.1f} cy={observation.intrinsics[1, 2]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
