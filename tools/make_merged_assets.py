#!/usr/bin/env python3
"""Merge Poly Haven + MIDI-3D curated asset indices into one registry.

layout_gen.py picks furniture by category from a single ``--assets-index``
JSON.  This tool builds that JSON from both libraries, translating MIDI-3D
category names into the Poly Haven style ("furniture"/"bed" ...) and attaching
a source-aware ``ref`` (``ph:<id>`` or ``midi3d:<id>``) so the layout generator
and the Blender asset loader can resolve either library.

Inputs (defaults match the repo layout):
  third_party/assets/polyhaven/index.json        ph entries, api_dimensions_mm
  third_party/assets/midi3d/index_curated.json   midi3d entries (tools/index_midi3d.py)

Output:
  third_party/assets/merged_index.json           union, ph-compatible fields

Entry schema (superset of both sources)::

    {name: {"api_dimensions_mm": [w, d, h],     # width, depth, height, mm
            "dims_m": [w, d, h],                 # same in metres (layout_gen form)
            "categories": [...],                 # Poly Haven style
            "ref": "ph:<id>" | "midi3d:<id>",
            "source": "polyhaven" | "midi3d",
            "files": [...]}}

Usage::

    python tools/make_merged_assets.py
    python tools/make_merged_assets.py --out third_party/assets/merged_index.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# MIDI-3D semantic category -> Poly Haven style category tags.
CATEGORY_TRANSLATE = {
    "bed": ["furniture", "bed"],
    "sofa": ["furniture", "seating", "sofa"],
    "chair": ["furniture", "seating", "chair"],
    "stool": ["furniture", "seating", "stool"],
    "table": ["furniture", "table"],
    "cabinet": ["furniture", "shelves", "cabinet"],
    "lamp": ["lighting", "lamp"],
    "misc": ["decorative", "misc"],
}


def ph_entries(index: dict) -> dict:
    out = {}
    # Poly Haven cache index is {models: {...}, textures: {...}, version: N}.
    models = index.get("models", index) if isinstance(index, dict) else {}
    for name, entry in models.items():
        if not isinstance(entry, dict):
            continue
        dims = entry.get("api_dimensions_mm")
        if not dims:
            continue
        out[name] = {
            **entry,
            "dims_m": [round(d / 1000.0, 4) for d in dims],
            "ref": f"ph:{name}",
            "source": "polyhaven",
        }
    return out


def midi3d_entries(index: dict) -> dict:
    out = {}
    for name, entry in index.items():
        dims = entry.get("api_dimensions_mm")
        if not dims:
            continue
        semantic = (entry.get("categories") or ["misc"])[0]
        out[name] = {
            **entry,
            "dims_m": [round(d / 1000.0, 4) for d in dims],
            "categories": CATEGORY_TRANSLATE.get(semantic, ["decorative", "misc"]),
            "ref": f"midi3d:{name}",
            "source": "midi3d",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--polyhaven", type=Path, default=REPO / "third_party/assets/polyhaven/index.json")
    ap.add_argument("--midi3d", type=Path, default=REPO / "third_party/assets/midi3d/index_curated.json")
    ap.add_argument("--out", type=Path, default=REPO / "third_party/assets/merged_index.json")
    ap.add_argument("--midi3d-cache-index", type=Path,
                    default=REPO / "third_party/assets/midi3d/index.json",
                    help="rewrite the midi3d cache index in Poly Haven shape "
                         "(models/<id>.gltf_variants) so assets.ensure_library can load glbs")
    args = ap.parse_args()

    merged = {}
    if args.polyhaven.exists():
        merged.update(ph_entries(json.loads(args.polyhaven.read_text())))
    else:
        print(f"[merged] missing {args.polyhaven}, skipping")
    if args.midi3d.exists():
        merged.update(midi3d_entries(json.loads(args.midi3d.read_text())))
    else:
        print(f"[merged] missing {args.midi3d}, skipping")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Poly Haven shape so checks.load_assets_index / assets.load_index consume it as-is.
    args.out.write_text(json.dumps({"version": 2, "models": merged, "textures": {}},
                                   indent=1, sort_keys=True))

    if args.midi3d_cache_index:
        models = {}
        cache_root = args.midi3d_cache_index.parent
        for name, entry in merged.items():
            if entry.get("source") != "midi3d":
                continue
            files = entry.get("files") or []
            if not files:
                continue
            # The flat index stores paths relative to the extract root; the cache
            # index must be relative to the cache dir.  Probe both.
            variant = None
            for cand in (files[0], os.path.join("3D-FRONT-TEST-SCENE", files[0])):
                if (cache_root / cand).exists():
                    variant = cand
                    break
            if variant is None:
                print(f"[merged] WARNING glb not found for {name}: {files[0]}")
                continue
            models[name] = {
                "api_dimensions_mm": entry["api_dimensions_mm"],
                "categories": entry.get("categories") or [],
                "gltf_variants": {"1k": variant},
            }
        if models:
            args.midi3d_cache_index.parent.mkdir(parents=True, exist_ok=True)
            args.midi3d_cache_index.write_text(
                json.dumps({"version": 2, "models": models, "textures": {}}, indent=1, sort_keys=True))
            print(f"[merged] midi3d cache index: {len(models)} glbs -> {args.midi3d_cache_index}")
        else:
            print("[merged] no midi3d glbs, skipping the cache index")

    by_src = {}
    for v in merged.values():
        by_src[v["source"]] = by_src.get(v["source"], 0) + 1
    by_cat = {}
    for v in merged.values():
        cat = (v.get("categories") or ["misc"])[-1]
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print(f"[merged] {len(merged)} assets {by_src} -> {args.out}")
    print(f"[merged] categories: {dict(sorted(by_cat.items(), key=lambda kv: -kv[1]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
