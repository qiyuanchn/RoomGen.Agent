#!/usr/bin/env python3
"""Index furniture models from a MIDI-3D (3D-FRONT processed) extract.

The MIDI-3D mirror (huanngzh/3D-Front) stores one glb per object::

    <root>/<house>/<room>/<Category>_<jid>_<n>.glb

This tool walks the tree, reads each glb/glTF's POSITION accessor min/max for
an axis-aligned bbox, and writes an index.json in the same schema the
Poly Haven asset cache uses (see third_party/assets/polyhaven/index.json),
so a layout generator can treat both libraries uniformly::

    {name: {"api_dimensions_mm": [w, d, h], "categories": [...],
            "files": ["relative/path.glb"], "source": "midi3d"}}

Dimensions are converted to millimetres to match the Poly Haven convention.
glb bbox is read from the binary glTF JSON chunk; .gltf is read directly.
No Blender or third-party imports required (stdlib only).

Usage::

    # after tar -xzf 3D-FRONT-TEST-SCENE.tar.gz
    python tools/index_midi3d.py --root third_party/assets/midi3d/3D-FRONT-TEST-SCENE \
        --out third_party/assets/midi3d/index.json

    # smoke-test the bbox reader on any glb/gltf
    python tools/index_midi3d.py --probe path/to/model.glb
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

NAME_RE = re.compile(r"^(?P<cat>[A-Za-z_]+?)_(?P<jid>[0-9a-fA-F\-]{36})(?:_(?P<idx>\d+))?$")

# MIDI-3D filename categories -> layout-facing semantics (coarse, extend as needed).
CATEGORY_ALIASES = {
    "Sofa": "sofa",
    "Bed": "bed",
    "Table": "table",
    "Chair": "chair",
    "Pier_Stool": "stool",
    "Cabinet_Shelf_Desk": "cabinet",
    "Lighting": "lamp",
    "Kids_Furniture": "misc",
    "Others": "misc",
}


def read_gltf_json(path: Path) -> dict:
    """Return the glTF JSON of a .gltf or .glb file."""
    data = path.read_bytes()
    if data[:4] == b"glTF":
        # glb: 12-byte header, then chunks; first chunk is JSON (type 0x4E4F534A)
        (version, _total) = struct.unpack_from("<II", data, 4)
        if version != 2:
            raise ValueError(f"unsupported glb version {version} in {path}")
        offset = 12
        while offset + 8 <= len(data):
            (chunk_len, chunk_type) = struct.unpack_from("<II", data, offset)
            offset += 8
            if chunk_type == 0x4E4F534A:
                return json.loads(data[offset : offset + chunk_len].decode("utf-8"))
            offset += chunk_len
        raise ValueError(f"no JSON chunk in {path}")
    return json.loads(data.decode("utf-8"))


def bbox_mm(path: Path) -> list:
    """Union bbox of all POSITION accessors, as [width, depth, height] millimetres.

    glTF is Y-up, so accessor extents come out [x=width, y=height, z=depth].
    The Poly Haven api_dimensions_mm convention this index mirrors is
    [width, depth, height], hence the (dx, dz, dy) reorder.  3D-FUTURE /
    MIDI-3D models are in metres like our scenes; values are scaled x1000.
    """
    gltf = read_gltf_json(path)
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    found = False
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            acc_index = prim.get("attributes", {}).get("POSITION")
            if acc_index is None:
                continue
            acc = gltf["accessors"][acc_index]
            amin, amax = acc.get("min"), acc.get("max")
            if not amin or not amax:
                continue
            found = True
            for i in range(3):
                mins[i] = min(mins[i], float(amin[i]))
                maxs[i] = max(maxs[i], float(amax[i]))
    if not found:
        raise ValueError(f"no POSITION accessor with min/max in {path}")
    dx, dy, dz = (maxs[i] - mins[i] for i in range(3))
    return [round(dx * 1000.0, 2), round(dz * 1000.0, 2), round(dy * 1000.0, 2)]


def semantic_category(raw_cat: str) -> str:
    return CATEGORY_ALIASES.get(raw_cat, "misc")


# House-shell meshes shipped alongside furniture in the MIDI-3D extract.
SHELL_STEMS = ("floor", "wall", "ceil", "others")

# Per-category physical plausibility (metres, on the largest instance per jid).
# MIDI-3D GLBs are per-instance exports with baked scene scales, so sizes span
# orders of magnitude; the largest instance is the best estimate of natural size.
PLAUSIBILITY = {
    "bed":     lambda w, d, h: 1.5 <= max(w, d) <= 2.4 and min(w, d) >= 0.8 and h <= 1.6,
    "sofa":    lambda w, d, h: max(w, d) >= 1.3 and 0.45 <= h <= 1.2 and min(w, d) >= 0.45,
    "chair":   lambda w, d, h: max(w, d) <= 1.05 and 0.5 <= h <= 1.3 and min(w, d) >= 0.25,
    "table":   lambda w, d, h: 0.5 <= max(w, d) <= 2.4 and 0.25 <= h <= 1.15,
    "cabinet": lambda w, d, h: 0.55 <= h <= 2.5 and max(w, d) >= 0.3,
    "stool":   lambda w, d, h: max(w, d) <= 1.0 and h <= 1.2,
    "lamp":    lambda w, d, h: max(w, d, h) <= 2.2,
    "misc":    lambda w, d, h: max(w, d) <= 3.0,
}


def curate_index(index: dict, verbose: bool = True) -> dict:
    """Deduplicate to jid level and drop implausible / shell entries."""
    def jid_of(name: str) -> str:
        m = NAME_RE.match(name)
        return m.group("jid") if m else name

    best: dict = {}
    for name, entry in index.items():
        if name.split("_")[0].lower() in SHELL_STEMS or name.lower() in SHELL_STEMS:
            continue
        vol = 1.0
        for d in entry["api_dimensions_mm"]:
            vol *= d
        j = jid_of(name)
        if j not in best or vol > best[j][0]:
            best[j] = (vol, name)

    out = {}
    for _, name in best.values():
        entry = index[name]
        cat = entry["categories"][0]
        w, d, h = (v / 1000.0 for v in entry["api_dimensions_mm"])
        rule = PLAUSIBILITY.get(cat)
        if rule and not rule(w, d, h):
            continue
        out[name] = entry
    if verbose:
        print(f"[index_midi3d] curated {len(out)} of {len(index)} models")
    return out


def index_tree(root: Path, out_path: Path, verbose: bool = True) -> dict:
    index: dict = {}
    skipped = 0
    for glb in sorted(root.rglob("*.glb")) + sorted(root.rglob("*.gltf")):
        # Skip the .bin companion and texture sidecars; name pattern on stem.
        m = NAME_RE.match(glb.stem.replace("_1k", "").replace("_2k", ""))
        if not m:
            skipped += 1
            continue
        raw_cat = m.group("cat")
        name = glb.stem
        if name in index:
            continue
        try:
            dims = bbox_mm(glb)
        except (ValueError, KeyError, struct.error) as exc:
            skipped += 1
            if verbose:
                print(f"[index_midi3d] skip {glb.name}: {exc}", file=sys.stderr)
            continue
        index[name] = {
            "api_dimensions_mm": dims,
            "categories": [semantic_category(raw_cat), raw_cat.lower()],
            "files": [str(glb.relative_to(root))],
            "source": "midi3d",
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=1, sort_keys=True))
    if verbose:
        print(f"[index_midi3d] {len(index)} models indexed, {skipped} skipped -> {out_path}")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="extracted MIDI-3D scene tree")
    ap.add_argument("--out", type=Path, help="output index.json path")
    ap.add_argument("--curate-out", type=Path,
                    help="also write the curated index (shell/jid-dedupe/plausibility)")
    ap.add_argument("--probe", type=Path, help="read one glb/gltf bbox and exit")
    args = ap.parse_args()

    if args.probe:
        print(json.dumps({"path": str(args.probe), "api_dimensions_mm": bbox_mm(args.probe)}, indent=1))
        return 0
    if not args.root or not args.out:
        ap.error("--root and --out are required (or use --probe)")
    index = index_tree(args.root, args.out)
    if args.curate_out:
        curated = curate_index(index)
        args.curate_out.write_text(json.dumps(curated, indent=1, sort_keys=True))
        print(f"[index_midi3d] curated index -> {args.curate_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
