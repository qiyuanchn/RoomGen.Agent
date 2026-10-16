#!/usr/bin/env python
"""Host-side exporter: Blender build directory -> PanoWorld ``scene/`` layout.

    scene/viewpoints/<name>/panoImage_2048.png
    scene/viewpoints/<name>/place_depth.png        uint16, meters * 1000
    scene/viewpoints/<name>/place_depth_scale.txt  "1000"
    scene/viewpoints/<name>/extrinsics.txt         4x4 camera-to-world, %.10f
    scene/map.json                                 {"0000": ["0001"], ...}
    scene/topdown.png                              top-down floor plan
    scene/topdown_<floor>.png                      one plan per storey (multi-floor)
    scene/connectors.json                          every door/stair/ramp connector

``map.json`` links two viewpoints when their rooms share a door **or a stair /
ramp** (see ``spec.SceneSpec.camera_adjacency``), so the graph carries the
cross-floor adjacency; ``connectors.json`` and ``export_report.json`` name the
connections behind those links.  A multi-floor scene writes one top-down plan
per storey and ``topdown.png`` becomes the stacked composite of them, so
consumers of the single-plan format keep working.

Only numpy + PIL are used (no bpy): the depth arrives as ``depth_<cam>.npy``
from the Blender side and the panorama as ``pano_<cam>.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import spec as spec_mod  # noqa: E402

TOPDOWN_PIXELS_PER_METER = 110.0
TOPDOWN_MARGIN_M = 1.2


def encode_depth(meters: np.ndarray, scale: float) -> np.ndarray:
    """meters -> uint16 depth map; 0 stays reserved for invalid pixels."""
    scaled = np.rint(np.asarray(meters, dtype=np.float64) * scale)
    encoded = np.clip(scaled, 1.0, 65535.0).astype(np.uint16)
    encoded[np.asarray(meters) <= 0.0] = 0
    return encoded


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def _draw_floor_image(spec: spec_mod.SceneSpec, floor, manifest: dict, heading: str, scale: float):
    """One storey's top-down plan as a PIL image (world XY, metres)."""
    rooms = spec.floor_rooms(floor.id) if spec.floors else list(spec.rooms)
    segments = [seg for seg in spec.wall_segments()
                if not spec.floors or seg.floor_id == floor.id]
    xs = [p[0] for room in rooms for p in room.polygon]
    ys = [p[1] for room in rooms for p in room.polygon]
    min_x, max_x = min(xs) - TOPDOWN_MARGIN_M, max(xs) + TOPDOWN_MARGIN_M
    min_y, max_y = min(ys) - TOPDOWN_MARGIN_M, max(ys) + TOPDOWN_MARGIN_M
    width = int(round((max_x - min_x) * scale))
    height = int(round((max_y - min_y) * scale))

    def to_px(point) -> tuple:
        return (
            (float(point[0]) - min_x) * scale,
            height - (float(point[1]) - min_y) * scale,
        )

    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)

    for room in rooms:
        color = tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in room.floor_color)
        draw.polygon([to_px(p) for p in room.polygon], fill=color, outline=(110, 110, 110), width=2)

    for opening in spec.slab_openings_for(floor.id) if spec.floors else []:
        draw.polygon([to_px(p) for p in opening.outline], fill=(235, 235, 235), outline=(120, 120, 200),
                     width=2)
        centre = [sum(p[0] for p in opening.outline) / len(opening.outline),
                  sum(p[1] for p in opening.outline) / len(opening.outline)]
        draw.text(to_px(centre), "slab opening", fill=(90, 90, 170), font=_font(12), anchor="mm")

    # wall spans as light grey bands (interior spans darker), openings punched out
    for segment in segments:
        p0, p1 = np.array(segment.p0), np.array(segment.p1)
        direction = np.array(segment.direction)
        normal = np.array([-direction[1], direction[0]]) * spec.wall_thickness / 2.0
        quad = [p0 + normal, p1 + normal, p1 - normal, p0 - normal]
        draw.polygon([to_px(p) for p in quad], fill=(150, 150, 150) if segment.interior else (205, 205, 205))

    doors = spec.door_connections()
    for connector in doors:
        if spec.floors and connector.from_floor != floor.id:
            continue
        center = connector.center
        if center is None:
            continue
        segment = spec_mod.wall_segment_for_point(
            spec.wall_segments(), center, connector.from_room,
            floor_id=connector.from_floor if spec.floors else None,
        )
        direction = np.array(segment.direction if segment else (1.0, 0.0))
        normal = np.array([-direction[1], direction[0]]) * (spec.wall_thickness / 2.0 + 0.05)
        centre = np.array(center, dtype=float)
        half = direction * connector.width / 2.0
        quad = [centre - half + normal, centre + half + normal, centre + half - normal, centre - half - normal]
        draw.polygon([to_px(p) for p in quad], fill=(255, 255, 255), outline=(200, 60, 60), width=2)
        label = f"{connector.kind} {connector.width:.2f}m"
        draw.text(to_px(centre + normal * 3.0), label, fill=(190, 40, 40), font=_font(13), anchor="mm")

    for index, plan in enumerate(spec.stair_plans()):
        if plan.connector.from_floor != floor.id and plan.connector.to_floor != floor.id:
            continue
        draw.polygon([to_px(p) for p in plan.footprint], outline=(30, 90, 200), width=3)
        steps = [box for box in plan.boxes if box["role"] != "landing"]
        for box in steps:
            draw.line([to_px(box["polygon"][1]), to_px(box["polygon"][2])], fill=(30, 90, 200), width=1)
        for box in plan.boxes:
            if box["role"] == "landing":
                draw.polygon([to_px(p) for p in box["polygon"]], outline=(30, 90, 200), width=2)
        up, down = plan.base_point, plan.head_point
        arrow = (to_px(up), to_px(down))
        draw.line([arrow[0][0], arrow[0][1], arrow[1][0], arrow[1][1]], fill=(30, 90, 200), width=2)
        label = (f"stair {plan.risers}x{plan.riser_m * 1000:.0f}mm"
                 + (" switchback" if plan.switchback else ""))
        mid = ((up[0] + down[0]) / 2.0, (up[1] + down[1]) / 2.0)
        draw.text(to_px(mid), label, fill=(20, 60, 160), font=_font(13), anchor="mm")
    for plan in spec.ramp_plans():
        if plan.connector.from_floor != floor.id and plan.connector.to_floor != floor.id:
            continue
        draw.polygon([to_px(p) for p in plan.footprint], outline=(30, 120, 90), width=3)
        draw.text(to_px(plan.center), f"ramp {plan.slope * 100:.0f}%", fill=(20, 90, 60),
                  font=_font(13), anchor="mm")

    for index, window in enumerate(spec.windows):
        if spec.floors and spec.room(window.room).floor_id != floor.id:
            continue
        p0, p1 = np.array(window.wall[0], dtype=float), np.array(window.wall[1], dtype=float)
        direction = (p1 - p0)
        direction = direction / max(1e-9, float(np.linalg.norm(direction)))
        normal = np.array([-direction[1], direction[0]]) * (spec.wall_thickness / 2.0 + 0.05)
        quad = [p0 + normal, p1 + normal, p1 - normal, p0 - normal]
        draw.polygon([to_px(p) for p in quad], fill=(160, 215, 255), outline=(40, 90, 180), width=2)
        draw.text(to_px((p0 + p1) / 2.0 - normal * 3.0), f"window sill {window.sill:.2f}m",
                  fill=(30, 70, 150), font=_font(13), anchor="mm")

    for camera in spec.cameras:
        if spec.floors and spec.room(camera.room).floor_id != floor.id:
            continue
        cx, cy = to_px(camera.pos)
        radius = 7
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(230, 40, 40), outline=(120, 0, 0))
        yaw = 0.0
        length = 1.4 * scale
        tip = (cx + length * np.cos(yaw), cy - length * np.sin(yaw))
        draw.line([cx, cy, tip[0], tip[1]], fill=(200, 30, 30), width=3)
        draw.text((cx, cy + radius + 10), f"{camera.name} ({camera.pos[0]:.2f}, {camera.pos[1]:.2f})",
                  fill=(150, 20, 20), font=_font(14), anchor="ma")

    draw.text((10, 10), f"{heading} - walls {spec.room_height(rooms[0].id) if spec.floors else spec.wall_height}m, "
                        f"{spec.wall_thickness}m thick",
              fill=(60, 60, 60), font=_font(14))
    return image, {"path": None, "size": [width, height], "pixels_per_meter": scale}


def draw_topdown(spec: spec_mod.SceneSpec, manifest: dict, path: str) -> dict:
    """Top-down plan: one image for a legacy spec, a stacked composite for multi-floor.

    Multi-floor specs also get ``topdown_<floor_id>.png`` per storey (written by
    :func:`draw_topdowns`); ``topdown.png`` stays the composite so every consumer
    of the single-plan format keeps working.
    """
    if not spec.floors:
        heading = "3D-FRONT replay - top view" if spec.front3d else "P0 demo house - top view"
        image, info = _draw_floor_image(spec, None, manifest, heading, TOPDOWN_PIXELS_PER_METER)
        image.save(path)
        info["path"] = os.path.basename(path)
        return info
    return draw_topdowns(spec, manifest, path)


def draw_topdowns(spec: spec_mod.SceneSpec, manifest: dict, path: str) -> dict:
    """Write ``topdown_<floor>.png`` per storey and the stacked ``topdown.png``."""
    scale = TOPDOWN_PIXELS_PER_METER
    images = []
    floors_info = []
    out_dir = os.path.dirname(os.path.abspath(path))
    for floor in spec.floors:
        heading = f"floor '{floor.id}' - walking surface {floor.elevation:.2f} m"
        image, _info = _draw_floor_image(spec, floor, manifest, heading, scale)
        floor_path = os.path.join(out_dir, f"topdown_{floor.id}.png")
        image.save(floor_path)
        images.append((floor, image))
        floors_info.append({"floor": floor.id, "path": os.path.basename(floor_path),
                            "size": list(image.size), "pixels_per_meter": scale})
    margin = 16
    label_height = 26
    width = max(image.size[0] for _floor, image in images)
    height = sum(label_height + image.size[1] for _floor, image in images) + margin * (len(images) + 1)
    composite = Image.new("RGB", (width + 2 * margin, height), (245, 245, 245))
    draw = ImageDraw.Draw(composite)
    y = margin
    for floor, image in images:
        draw.text((margin, y), f"floor '{floor.id}' - elevation {floor.elevation:.2f} m, "
                               f"height {floor.height:.2f} m, slab {floor.slab_thickness:.2f} m",
                  fill=(40, 40, 40), font=_font(16))
        y += label_height
        composite.paste(image, (margin, y))
        y += image.size[1] + margin
    composite.save(path)
    return {"path": os.path.basename(path), "size": [composite.size[0], composite.size[1]],
            "pixels_per_meter": scale, "floors": floors_info}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export a blender_agent build to PanoWorld viewpoint format")
    parser.add_argument("--build", required=True, help="build directory written by run_build.py")
    parser.add_argument("--spec", required=True, help="the same scene_spec JSON used for the build")
    parser.add_argument("--out", required=True, help="scene output directory")
    parser.add_argument("--force", action="store_true", help="overwrite a non-empty output directory")
    args = parser.parse_args(argv)

    spec = spec_mod.load_spec(args.spec)
    build_dir = os.path.abspath(args.build)
    out_dir = os.path.abspath(args.out)
    manifest_path = os.path.join(build_dir, "build_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.force:
        raise SystemExit(f"{out_dir} is not empty (pass --force to overwrite)")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    scale = float(manifest.get("depth_scale", spec_mod.DEPTH_SCALE))
    render = manifest.get("render", {})
    width = int(render.get("width", spec_mod.PANO_WIDTH))
    height = int(render.get("height", spec_mod.PANO_HEIGHT))
    if (width, height) != (spec_mod.PANO_WIDTH, spec_mod.PANO_HEIGHT):
        print(
            f"[export_scene] WARNING: build was rendered at {width}x{height}, not the PanoWorld "
            f"canonical {spec_mod.PANO_WIDTH}x{spec_mod.PANO_HEIGHT} (fine for iteration renders)"
        )
    viewpoints_dir = os.path.join(out_dir, "viewpoints")
    os.makedirs(viewpoints_dir, exist_ok=True)

    written = []
    warnings = []
    for entry in manifest["cameras"]:
        name = entry["camera"]
        view_dir = os.path.join(viewpoints_dir, name)
        os.makedirs(view_dir, exist_ok=True)

        pano_src = os.path.join(build_dir, entry["pano"])
        depth_src = os.path.join(build_dir, entry["depth_npy"])
        pano = Image.open(pano_src)
        if pano.size != (width, height):
            raise SystemExit(f"{pano_src}: expected {width}x{height} (from the manifest), got {pano.size}")
        pano.convert("RGB").save(os.path.join(view_dir, "panoImage_2048.png"))

        meters = np.load(depth_src)
        if meters.ndim != 2 or meters.shape != (height, width):
            raise SystemExit(f"{depth_src}: expected ({height}, {width}), got {meters.shape}")
        encoded = encode_depth(meters, scale)
        invalid = int((encoded == 0).sum())
        if invalid:
            warnings.append(f"{name}: {invalid} invalid depth pixels ({100.0 * invalid / encoded.size:.4f}%)")
            print(f"[export_scene] WARNING {name}: {invalid} invalid (0) depth pixels")
        over = int((np.rint(meters * scale) > 65535).sum())
        if over:
            raise SystemExit(f"{name}: {over} pixels exceed the uint16 range at scale {scale}; increase the dome distance")
        Image.fromarray(encoded).save(os.path.join(view_dir, "place_depth.png"))
        with open(os.path.join(view_dir, "place_depth_scale.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"{scale:g}\n")

        yaw = float(entry.get("yaw", 0.0))
        matrix = np.array(spec_mod.extrinsics_matrix(entry["pos"], yaw), dtype=np.float64)
        np.savetxt(os.path.join(view_dir, "extrinsics.txt"), matrix, fmt="%.10f")
        written.append(name)
        print(
            f"[export_scene] {name}: depth {float(meters[meters > 0].min()):.3f}..{float(meters.max()):.3f} m "
            f"-> {os.path.relpath(view_dir, out_dir)}"
        )

    adjacency = manifest.get("camera_adjacency") or spec.camera_adjacency()
    graph = {name: sorted(peers) for name, peers in adjacency.items() if name in written}
    with open(os.path.join(out_dir, "map.json"), "w", encoding="utf-8") as handle:
        json.dump(graph, handle, indent=4, sort_keys=True)
        handle.write("\n")

    # cross-floor links (stairs / ramps) are part of the graph; the connector
    # list spells out which viewpoint pairs they joined
    cross_floor = []
    by_room = {}
    for camera in spec.cameras:
        by_room.setdefault(camera.room, []).append(camera.name)
    for connector in spec.connections():
        if connector.kind in ("stair", "ramp") or connector.from_floor != connector.to_floor:
            cross_floor.append({
                "kind": connector.kind, "from_room": connector.from_room, "to_room": connector.to_room,
                "from_floor": connector.from_floor, "to_floor": connector.to_floor,
                "viewpoints": sorted(set(by_room.get(connector.from_room, []))
                                     | set(by_room.get(connector.to_room, []))),
            })
    with open(os.path.join(out_dir, "connectors.json"), "w", encoding="utf-8") as handle:
        json.dump({"connectors": [
            {"kind": c.kind, "from_room": c.from_room, "to_room": c.to_room,
             "from_floor": c.from_floor, "to_floor": c.to_floor, "width": c.width,
             "height": c.height, "center": c.center, "run": c.run, "rise": c.rise,
             "steps": c.steps, "landing_m": c.landing_m} for c in spec.connections()
        ]}, handle, indent=2)
        handle.write("\n")

    topdown = draw_topdown(spec, manifest, os.path.join(out_dir, "topdown.png"))

    summary = {
        "kind": "blender_agent_scene_export",
        "build": build_dir,
        "spec": os.path.abspath(args.spec),
        "out": out_dir,
        "depth_scale": scale,
        "viewpoints": written,
        "map": graph,
        "topdown": topdown,
        "floors": [{"id": floor.id, "elevation": floor.elevation, "height": floor.height,
                    "slab_thickness": floor.slab_thickness} for floor in spec.floors],
        "cross_floor_links": cross_floor,
        "warnings": warnings,
    }
    with open(os.path.join(out_dir, "export_report.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[export_scene] wrote {len(written)} viewpoints, map.json, topdown.png to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
