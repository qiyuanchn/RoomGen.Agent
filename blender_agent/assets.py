"""Poly Haven asset pipeline: host-side downloader + Blender-side loader.

Cache layout (gitignored, under ``REPO/third_party/assets/polyhaven``)::

    index.json                       what is cached, with metadata
    .content_manifest.json           sha256 of every model/texture file, keyed by
                                     (size, mtime, ctime) so only changed files are
                                     re-read; the stable digest the batch resume
                                     fingerprints (see content_digest_block)
    library/<id>.<res>.blend         imported + normalised asset
    library/<id>.<res>.json          content digest the library was built from
    models/<id>/<id>_2k.gltf         + .bin + textures/*.jpg (as referenced)
    textures/<id>/<id>_diff_2k.jpg   Diffuse / nor_gl / Rough

Host side (no ``bpy``, numpy/PIL only) -- run with the host python::

    python blender_agent/assets.py list --category seating
    python blender_agent/assets.py fetch --assets ph:sofa_02 ph:oak_wood_planks
    python blender_agent/assets.py check --assets ph:sofa_02
    python blender_agent/assets.py index

Three libraries share this machinery, selected by the ref's scheme
(``split_asset_ref``) and each with its own cache directory:

``ph:``
    Poly Haven (CC0) downloads, cached in ``third_party/assets/polyhaven``.
``midi3d:``
    3D-FUTURE glbs indexed by ``tools/index_midi3d.py`` into
    ``third_party/assets/midi3d``.
``gen:``
    generated (TRELLIS / procedural) furniture glbs written by
    ``tools/gen_assets_trellis.py`` into ``third_party/assets/generated``.

A generated entry is a Poly Haven-shaped model entry whose glb is registered as
its ``1k`` variant (``record_generated_model``), so ``ensure_library`` builds a
``library/<id>.1k.blend`` from it exactly as it does for a ``ph:`` asset and
``append_asset`` / ``place_asset`` need no special case.

Blender side (no network): ``ensure_library`` imports + normalises one asset
into a reusable ``library/<id>.blend``, ``append_asset`` pulls that collection
into the current file and ``place_asset`` drops a *collection instance* of it
into the scene (shared mesh + texture data, cheap for repeats such as four
dining chairs or a stack of books).

Asset conventions (see ``asset_meta.json`` for per-asset overrides):

* units: the glTF importer brings Poly Haven models in at 1 unit = 1 m; the
  loader re-checks the measured bounding box against the API's ``dimensions``
  and against absolute plausibility (0.02 m .. 20 m) and refuses wild ones.
* registry frame: ``api_dimensions_mm`` is ``[width_x, depth_y, height_z]`` in
  the **Blender frame after the glTF Y-up conversion** (Blender X = glTF X,
  Blender Y = -glTF Z, Blender Z = glTF Y).  Most exports match that metadata
  axis for axis.  When a glTF was exported 90 deg yawed against its own
  dimensions (the X/Y footprint is the transpose of the registry), the loader
  applies a +90 deg Z rotation at library-build time so the built extents are
  the registry extents axis for axis; the scene checker and the builder then
  agree without a per-asset fudge factor.
* origin: after normalisation the object origin sits at the **bottom-centre of
  the bounding box**, so ``z = 0`` means "standing on the floor of a room whose
  floor is at z = 0" (the P0 shell's floors are exactly there).
* facing: the **front of the asset faces -Y** in local space, so a placement
  with ``rot_z_deg = 0`` looks towards -Y; ``rot_z_deg = 180`` faces +Y, and so
  on (counter-clockwise around +Z, matching Blender's rotation_euler.z).
  Poly Haven models arrive with the front along +Y for some assets and -Y for
  others; the per-asset correction ships in ``asset_meta.json`` (``yaw_deg``)
  and is applied as a world-space Z rotation after the registry-frame
  alignment, once at library-build time, so the scene spec never needs a fudge
  factor.  glTF imports use quaternion rotation mode, so the correction is
  applied to ``matrix_world`` rather than to the invisible Euler channels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

API_ROOT = "https://api.polyhaven.com"
USER_AGENT = "PanoWorld-blender-agent/0.2 (Poly Haven asset fetcher; CC0 assets)"
DEFAULT_CACHE = os.path.join(REPO_ROOT, "third_party", "assets", "polyhaven")
DEFAULT_GENERATED_CACHE = os.path.join(REPO_ROOT, "third_party", "assets", "generated")
DEFAULT_META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset_meta.json")

TEXTURE_MAPS = ("Diffuse", "nor_gl", "Rough", "arm")
DEFAULT_RESOLUTION = "2k"

# every library a scene spec may name in a furniture ref
ASSET_SCHEMES = ("ph", "midi3d", "gen")

#: Version of the import/normalisation convention (origin, facing, registry
#: axis alignment).  Cached ``library/<id>.blend`` files are reused only when
#: their sidecar records this version, so changing the convention rebuilds the
#: libraries instead of silently serving geometry built by the old rules.
NORMALISATION_VERSION = 2

#: Per-cache metadata recording the content digest of every indexed asset.
#: The digest covers the bytes of the model and its key texture files, not
#: only their paths, so replacing a file in place invalidates the libraries
#: and resume fingerprints derived from it.  File hashes are cached by
#: (size, mtime, ctime): an unchanged tree costs stats only, while a touched
#: or replaced file is re-read.
CONTENT_MANIFEST_NAME = ".content_manifest.json"
CONTENT_MANIFEST_KIND = "asset_content_manifest"
CONTENT_MANIFEST_SCHEMA = 1
LIBRARY_METADATA_KIND = "asset_library_content"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def split_asset_ref(ref: str) -> str:
    """``ph:sofa_02`` -> ``sofa_02`` (the prefix names the library: see ``ASSET_SCHEMES``)."""
    if ":" in ref:
        scheme, _, rest = ref.partition(":")
        if scheme not in ASSET_SCHEMES:
            raise ValueError(
                f"unsupported asset scheme in '{ref}' "
                f"(known: {', '.join(f'{name}:' for name in ASSET_SCHEMES)})"
            )
        return rest
    return ref


def sibling_cache(cache_dir: str, name: str) -> str:
    """``<assets>/polyhaven`` -> ``<assets>/<name>`` (the library caches sit side by side).

    A ``--cache`` that does not end in ``polyhaven`` (the tests' temporary caches)
    keeps the library inside itself instead, ``<cache_dir>/<name>``.
    """
    cache_dir = os.path.abspath(cache_dir)
    if os.path.basename(cache_dir) == "polyhaven":
        return os.path.join(os.path.dirname(cache_dir), name)
    return os.path.join(cache_dir, name)



def load_asset_meta(path: str = DEFAULT_META) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_index(cache_dir: str = DEFAULT_CACHE) -> dict:
    path = os.path.join(cache_dir, "index.json")
    if not os.path.isfile(path):
        return {"version": 2, "models": {}, "textures": {}}
    with open(path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    index.setdefault("models", {})
    index.setdefault("textures", {})
    return index


def save_index(index: dict, cache_dir: str = DEFAULT_CACHE) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    index["version"] = 2
    path = os.path.join(cache_dir, "index.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def asset_entry(index: dict, ref: str) -> tuple:
    """Look up ``ph:<id>`` in the index -> (kind, entry); raises when missing."""
    asset_id = split_asset_ref(ref)
    for kind in ("models", "textures"):
        if asset_id in index.get(kind, {}):
            return kind[:-1], index[kind][asset_id]
    raise KeyError(
        f"asset '{asset_id}' is not in the cache index; run "
        f"'python blender_agent/assets.py fetch --assets ph:{asset_id}' first"
    )


# --------------------------------------------------------------------------- #
# content digests (cache/library provenance)
# --------------------------------------------------------------------------- #
def _canonical_json(node) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_cache_relative(rel_path) -> str:
    """A cache-relative path with ``/`` separators; ``None`` for escapes."""
    rel = os.path.normpath(str(rel_path).replace("\\", "/"))
    if os.path.isabs(rel) or rel == ".." or rel.startswith("../"):
        return None
    return rel.replace(os.sep, "/")


def _model_content_paths(cache_dir: str, asset_id: str, entry: dict) -> list:
    """Every cached file that contributes to a model, cache-relative and sorted.

    ``gltf``/``gltf_variants`` are cache-relative in every index.  Poly Haven's
    ``files`` list is relative to ``models/<id>/`` while the generated and
    MIDI-3D indexes store cache-relative paths; resolve on disk first and fall
    back to the source convention when a file is absent.
    """
    paths = set()
    variants = entry.get("gltf_variants") or {}
    if not variants and entry.get("gltf"):
        variants = {str(entry.get("resolution") or "default"): entry["gltf"]}
    for rel in list(variants.values()) + ([entry["gltf"]] if entry.get("gltf") else []):
        rel = _safe_cache_relative(rel)
        if rel:
            paths.add(rel)
    source = str(entry.get("source") or "")
    for rel in entry.get("files") or []:
        rel = _safe_cache_relative(rel)
        if rel is None:
            continue
        direct = rel
        nested = _safe_cache_relative(os.path.join("models", asset_id, rel))
        if os.path.isfile(os.path.join(cache_dir, direct)):
            paths.add(direct)
        elif nested is not None and os.path.isfile(os.path.join(cache_dir, nested)):
            paths.add(nested)
        elif source in ("generated", "midi3d"):
            paths.add(direct)
        elif nested is not None:
            paths.add(nested)
        else:
            paths.add(direct)
    return sorted(paths)


def _texture_content_paths(entry: dict) -> list:
    paths = set()
    for rel in (entry.get("maps") or {}).values():
        rel = _safe_cache_relative(rel)
        if rel:
            paths.add(rel)
    for rel in entry.get("files") or []:
        rel = _safe_cache_relative(rel)
        if rel:
            paths.add(rel)
    return sorted(paths)


def _content_file_record(cache_dir: str, rel_path: str, previous: dict = None) -> dict:
    """Hash one file unless its ``(size, mtime, ctime)`` matches the cache.

    ``ctime`` catches an in-place replacement that restores the mtime, while a
    pure ``touch`` re-hashes the same bytes and therefore keeps the digest.
    """
    path = os.path.join(cache_dir, rel_path)
    if not os.path.isfile(path):
        return {"present": False}
    stat = os.stat(path)
    if (
        previous
        and previous.get("present")
        and previous.get("sha256")
        and previous.get("size") == stat.st_size
        and previous.get("mtime_ns") == stat.st_mtime_ns
        and previous.get("ctime_ns") == stat.st_ctime_ns
    ):
        return previous
    return {
        "present": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "sha256": _file_sha256(path),
    }


def _content_entry_digest(kind: str, asset_id: str, records: dict) -> str:
    payload = {
        "schema_version": CONTENT_MANIFEST_SCHEMA,
        "kind": kind,
        "id": asset_id,
        "files": [
            {"path": rel, "present": record["present"], "sha256": record.get("sha256")}
            for rel, record in sorted(records.items())
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _load_content_manifest(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        data = None
    if (
        not isinstance(data, dict)
        or data.get("kind") != CONTENT_MANIFEST_KIND
        or data.get("schema_version") != CONTENT_MANIFEST_SCHEMA
    ):
        return {"kind": CONTENT_MANIFEST_KIND,
                "schema_version": CONTENT_MANIFEST_SCHEMA,
                "entries": {}, "files": {}}
    if not isinstance(data.get("entries"), dict):
        data["entries"] = {}
    if not isinstance(data.get("files"), dict):
        data["files"] = {}
    return data


def _write_json_atomic(path: str, data: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def content_digest_block(cache_dir: str = DEFAULT_CACHE, index: dict = None,
                         asset_ids=None, write: bool = True) -> dict:
    """Digest the actual bytes behind every asset of one cache index.

    Returns ``content_sha256`` (aggregate), ``content`` (the stable, compact
    block the batch resume gate records) and ``entries`` (per-asset digest and
    file hashes).  Unchanged files are reused from ``.content_manifest.json``
    keyed by stat metadata; only new or touched files are read again.
    """
    cache_dir = os.path.abspath(cache_dir)
    index = load_index(cache_dir) if index is None else index
    manifest_path = os.path.join(cache_dir, CONTENT_MANIFEST_NAME)
    previous = _load_content_manifest(manifest_path)
    previous_files = previous.get("files") or {}
    wanted = None if asset_ids is None else {str(value) for value in asset_ids}
    entries = {}
    file_records = {}
    missing = 0
    for index_kind, kind in (("models", "model"), ("textures", "texture")):
        for asset_id, entry in sorted((index.get(index_kind) or {}).items()):
            if wanted is not None and asset_id not in wanted:
                continue
            if not isinstance(entry, dict):
                continue
            rel_paths = (_model_content_paths(cache_dir, asset_id, entry)
                         if kind == "model" else _texture_content_paths(entry))
            records = {}
            for rel in rel_paths:
                record = _content_file_record(cache_dir, rel, previous_files.get(rel))
                records[rel] = record
                file_records[rel] = record
                if not record["present"]:
                    missing += 1
            key = "%s/%s" % (kind, asset_id)
            entries[key] = {
                "kind": kind,
                "id": asset_id,
                "digest": _content_entry_digest(kind, asset_id, records),
                "files": [
                    {"path": rel, "present": record["present"],
                     "sha256": record.get("sha256")}
                    for rel, record in sorted(records.items())
                ],
            }
    # a symlinked cache belongs to another checkout: read it, do not write there
    if write and not os.path.islink(cache_dir):
        manifest = {
            "kind": CONTENT_MANIFEST_KIND,
            "schema_version": CONTENT_MANIFEST_SCHEMA,
            "entries": {key: {"kind": value["kind"], "id": value["id"],
                              "digest": value["digest"], "files": value["files"]}
                        for key, value in sorted(entries.items())},
            "files": {rel: file_records[rel] for rel in sorted(file_records)},
        }
        try:
            if manifest != previous:
                _write_json_atomic(manifest_path, manifest)
        except OSError:
            pass
    content = {
        "models": {value["id"]: value["digest"] for value in entries.values()
                   if value["kind"] == "model"},
        "textures": {value["id"]: value["digest"] for value in entries.values()
                     if value["kind"] == "texture"},
        "files": len(file_records),
        "missing": missing,
    }
    aggregate = hashlib.sha256(_canonical_json({
        "schema_version": CONTENT_MANIFEST_SCHEMA,
        "entries": {key: value["digest"] for key, value in sorted(entries.items())},
    }).encode("utf-8")).hexdigest()
    return {"content_sha256": aggregate, "content": content, "entries": entries}


def content_digest_for_index(index_path: str, write: bool = True) -> dict:
    """Content digest block for a cache index or a merged assets registry."""
    index_path = os.path.abspath(index_path)
    cache_dir = os.path.dirname(index_path)
    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
    except Exception as error:
        raise RuntimeError(f"cannot read asset index {index_path}: {error}")
    if not isinstance(index, dict):
        raise RuntimeError(f"asset index {index_path} is not an object")
    models = index.get("models") or {}
    merged = any(isinstance(entry, dict) and entry.get("ref") for entry in models.values())
    if not merged:
        return content_digest_block(cache_dir, index=index, write=write)

    groups = {}
    for asset_id, entry in models.items():
        ref = str((entry or {}).get("ref") or "")
        scheme = ref.partition(":")[0] if ":" in ref else None
        if scheme not in ASSET_SCHEMES:
            scheme = {"polyhaven": "ph", "midi3d": "midi3d",
                      "generated": "gen"}.get((entry or {}).get("source"))
        groups.setdefault(scheme or "ph", set()).add(asset_id)
    # make_merged_assets.py writes textures: {}, but the spec still names
    # ``ph:<texture>`` for floors and walls: fingerprint the whole Poly Haven
    # texture set alongside the merged models.
    if isinstance(index.get("textures"), dict):
        groups.setdefault("ph", set()).update(index["textures"])
    cache_names = {"ph": "polyhaven", "midi3d": "midi3d", "gen": "generated"}
    blocks = {}
    for scheme in sorted(groups):
        cache = os.path.join(cache_dir, cache_names.get(scheme, scheme))
        cache_index = os.path.join(cache, "index.json")
        if scheme == "midi3d" and os.path.islink(cache):
            # the MIDI-3D mirror is a 28 GiB read-only checkout outside the
            # repo; fingerprint its registry rather than every glb
            blocks[scheme] = {
                "content_sha256": _file_sha256(cache_index) if os.path.isfile(cache_index) else None,
                "content": {"index_only": True},
            }
            continue
        if not os.path.isfile(cache_index):
            blocks[scheme] = {"content_sha256": None, "content": {"missing_index": True}}
            continue
        ids = set(groups[scheme])
        if scheme == "ph":
            try:
                with open(cache_index, "r", encoding="utf-8") as handle:
                    cache_index_data = json.load(handle)
            except Exception:
                cache_index_data = {}
            ids.update((cache_index_data.get("textures") or {}).keys())
        block = content_digest_block(cache, asset_ids=ids, write=write)
        blocks[scheme] = {"content_sha256": block["content_sha256"],
                          "content": block["content"]}
    aggregate = hashlib.sha256(_canonical_json({
        "schema_version": CONTENT_MANIFEST_SCHEMA,
        "caches": {scheme: block["content_sha256"] for scheme, block in sorted(blocks.items())},
    }).encode("utf-8")).hexdigest()
    return {"content_sha256": aggregate, "content": {"caches": blocks}, "entries": {}}


# --------------------------------------------------------------------------- #
# host side: downloader
# --------------------------------------------------------------------------- #
def _request(url: str):
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def api_get(path: str, timeout: float = 60.0) -> dict:
    url = path if path.startswith("http") else f"{API_ROOT}{path}"
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        return json.load(response)


def download_file(url: str, dest: str, md5: str = None, force: bool = False, quiet: bool = False) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest) and not force:
        if md5 is None or _md5(dest) == md5:
            return dest
    tmp = dest + ".part"
    with urllib.request.urlopen(_request(url), timeout=300) as response, open(tmp, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    if md5 is not None and _md5(tmp) != md5:
        os.remove(tmp)
        raise RuntimeError(f"checksum mismatch for {url}")
    os.replace(tmp, dest)
    if not quiet:
        print(f"    downloaded {os.path.relpath(dest, REPO_ROOT)} ({os.path.getsize(dest) / 1024:.0f} KiB)")
    return dest


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_kind(asset_id: str) -> str:
    """'model' | 'texture' from the API (type 2 = model, 1 = texture, 0 = HDRI)."""
    info = api_get(f"/info/{asset_id}")
    kind = {"2": "model", "1": "texture"}.get(str(info.get("type")))
    if kind is None:
        raise RuntimeError(f"asset '{asset_id}' has unsupported type {info.get('type')!r}")
    return kind


def fetch_model(asset_id: str, cache_dir: str = DEFAULT_CACHE, resolution: str = DEFAULT_RESOLUTION,
                force: bool = False) -> dict:
    """Download a model's glTF (+ .bin + textures) and return its index entry.

    Called once per resolution; extra resolutions are recorded in
    ``gltf_variants`` so a build can render from 1k textures (4x less GPU/CPU
    texture memory) while the cache still holds the full 2k set.
    """
    info = api_get(f"/info/{asset_id}")
    files = api_get(f"/files/{asset_id}")
    node = files.get("gltf", {}).get(resolution, {}).get("gltf")
    if node is None:
        raise RuntimeError(f"asset '{asset_id}' has no gltf at resolution {resolution}")
    dest_dir = os.path.join(cache_dir, "models", asset_id)
    gltf_name = f"{asset_id}_{resolution}.gltf"
    download_file(node["url"], os.path.join(dest_dir, gltf_name), md5=node.get("md5"), force=force)
    downloaded = [gltf_name]
    for rel_path, file_info in sorted(node.get("include", {}).items()):
        download_file(file_info["url"], os.path.join(dest_dir, rel_path),
                      md5=file_info.get("md5"), force=force)
        downloaded.append(rel_path)
    index = load_index(cache_dir)
    previous = index["models"].get(asset_id, {})
    variants = dict(previous.get("gltf_variants", {}))
    variants[resolution] = os.path.join("models", asset_id, gltf_name)
    entry = {
        "id": asset_id,
        "kind": "model",
        "name": info.get("name", asset_id),
        "categories": info.get("categories", []),
        "tags": info.get("tags", []),
        "attributes": info.get("attributes", {}),
        "authors": info.get("authors", {}),
        "license": "CC0",
        "polycount": info.get("polycount"),
        "api_dimensions_mm": info.get("dimensions"),
        "source": f"{API_ROOT}/files/{asset_id}",
        "resolution": resolution,
        "gltf": variants.get(previous.get("resolution", resolution), variants[resolution]),
        "gltf_variants": variants,
        "files": sorted(set(previous.get("files", [])) | set(downloaded)),
    }
    print(
        f"  model {asset_id} ({resolution}): {len(downloaded)} files, {entry['polycount']} tris, "
        f"dims {[round(float(v) / 1000, 3) for v in (info.get('dimensions') or [])]} m "
        f"| variants {sorted(variants)}"
    )
    return entry


def fetch_texture(asset_id: str, cache_dir: str = DEFAULT_CACHE, resolution: str = DEFAULT_RESOLUTION,
                  force: bool = False, maps: tuple = TEXTURE_MAPS) -> dict:
    """Download a texture's PBR maps and return its index entry (tile size in m)."""
    info = api_get(f"/info/{asset_id}")
    files = api_get(f"/files/{asset_id}")
    dest_dir = os.path.join(cache_dir, "textures", asset_id)
    maps_out = {}
    for map_name in maps:
        node = files.get(map_name, {}).get(resolution, {}).get("jpg")
        if node is None:
            continue
        name = os.path.basename(node["url"])
        download_file(node["url"], os.path.join(dest_dir, name), md5=node.get("md5"), force=force)
        maps_out[map_name] = os.path.join("textures", asset_id, name)
    if "Diffuse" not in maps_out:
        raise RuntimeError(f"texture '{asset_id}' has no Diffuse map at {resolution}")
    dimensions = info.get("dimensions") or [2000, 2000]  # millimetres (Poly Haven API)
    entry = {
        "id": asset_id,
        "kind": "texture",
        "name": info.get("name", asset_id),
        "categories": info.get("categories", []),
        "tags": info.get("tags", []),
        "authors": info.get("authors", {}),
        "license": "CC0",
        "api_dimensions_mm": dimensions,
        "tile_size_m": [round(float(dimensions[0]) / 1000.0, 6), round(float(dimensions[1]) / 1000.0, 6)],
        "source": f"{API_ROOT}/files/{asset_id}",
        "resolution": resolution,
        "maps": maps_out,
    }
    print(f"  texture {asset_id}: maps {sorted(maps_out)}, tile {entry['tile_size_m']} m")
    return entry


def fetch_assets(refs, cache_dir: str = DEFAULT_CACHE, resolution: str = DEFAULT_RESOLUTION,
                 force: bool = False) -> dict:
    """Download every ``ph:<id>`` (auto-detecting model vs texture) into the cache.

    Already-cached assets are skipped unless the requested ``resolution`` variant
    is still missing.
    """
    index = load_index(cache_dir)
    for ref in refs:
        asset_id = split_asset_ref(ref)
        if not force:
            model = index["models"].get(asset_id)
            if model is not None and resolution in (model.get("gltf_variants") or {}):
                print(f"  {asset_id}: cached ({resolution})")
                continue
            texture = index["textures"].get(asset_id)
            if texture is not None and texture.get("resolution") == resolution:
                print(f"  {asset_id}: cached")
                continue
        kind = asset_kind(asset_id)
        if kind == "model":
            entry = fetch_model(asset_id, cache_dir, resolution, force)
        else:
            entry = fetch_texture(asset_id, cache_dir, resolution, force)
        index["models" if kind == "model" else "textures"][asset_id] = entry
        save_index(index, cache_dir)
    return index


# --------------------------------------------------------------------------- #
# host side: the generated (``gen:``) library
# --------------------------------------------------------------------------- #
def record_generated_model(cache_dir: str, asset_id: str, glb_relpath: str, dimensions_mm,
                           categories, name: str = None, extra: dict = None) -> dict:
    """Write one generated glb into ``<cache_dir>/index.json`` (Poly Haven shape).

    ``glb_relpath`` is relative to ``cache_dir``; the glb is registered as the
    asset's ``1k`` variant, which is what makes the ordinary ``ensure_library``
    path build ``library/<asset_id>.1k.blend`` from it.  ``dimensions_mm`` are
    the measured bounding-box dims in millimetres ([width, depth, height], the
    Poly Haven convention) and ``extra`` carries the provenance the generator
    wants to keep (backend, prompt, seed, ...).
    """
    index = load_index(cache_dir)
    previous = index["models"].get(asset_id, {})
    entry = {
        "id": asset_id,
        "kind": "model",
        "name": name or asset_id,
        "categories": list(categories),
        "license": "generated",
        "api_dimensions_mm": [round(float(v), 2) for v in dimensions_mm],
        "source": "generated",
        "gltf": glb_relpath,
        "gltf_variants": {**previous.get("gltf_variants", {}), "1k": glb_relpath},
        "files": sorted(set(previous.get("files", [])) | {glb_relpath}),
    }
    entry.update(extra or {})
    index["models"][asset_id] = entry
    save_index(index, cache_dir)
    return entry



def cmd_list(args) -> int:
    query = "/assets"
    params = []
    if args.category:
        params.append(f"t={args.category_type}")
    if args.search:
        params.append(f"q={urllib.parse.quote(args.search)}")
    data = api_get(query + ("?" + "&".join(params) if params else ""))
    rows = []
    for asset_id, info in data.items():
        categories = info.get("categories", [])
        if args.category and args.category not in categories:
            continue
        if args.search and args.search.lower() not in asset_id.lower() and \
                args.search.lower() not in str(info.get("name", "")).lower():
            continue
        rows.append((asset_id, info.get("name"), info.get("polycount"), categories))
    for asset_id, name, poly, categories in sorted(rows):
        print(f"{asset_id:34s} {str(name)[:28]:28s} {str(poly):>7s}  {','.join(categories)}")
    print(f"# {len(rows)} assets")
    return 0


def cmd_check(args) -> int:
    index = load_index(args.cache)
    missing = []
    for ref in args.assets:
        asset_id = split_asset_ref(ref)
        try:
            kind, entry = asset_entry(index, ref)
        except KeyError:
            missing.append(f"{asset_id} (not in index)")
            continue
        rel_files = [entry["gltf"]] if kind == "model" else list(entry["maps"].values())
        for rel in rel_files:
            if not os.path.isfile(os.path.join(args.cache, rel)):
                missing.append(f"{asset_id}:{rel}")
    if missing:
        print("MISSING:\n  " + "\n  ".join(missing))
        return 1
    print(f"all {len(args.assets)} assets cached in {args.cache}")
    return 0


def cmd_index(args) -> int:
    index = load_index(args.cache)
    print(json.dumps({kind: sorted(index[kind]) for kind in ("models", "textures")}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Poly Haven asset downloader for blender_agent")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="query the Poly Haven asset API")
    p_list.add_argument("--category", default=None)
    p_list.add_argument("--category-type", default="models", choices=["models", "textures", "hdris"])
    p_list.add_argument("--search", default=None)
    p_list.set_defaults(func=cmd_list)

    p_fetch = sub.add_parser("fetch", help="download assets (models or textures)")
    p_fetch.add_argument("--assets", nargs="+", required=True)
    p_fetch.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=lambda a: (fetch_assets(a.assets, a.cache, a.resolution, a.force), 0)[1])

    p_check = sub.add_parser("check", help="verify assets are cached (offline)")
    p_check.add_argument("--assets", nargs="+", required=True)
    p_check.set_defaults(func=cmd_check)

    p_index = sub.add_parser("index", help="show the cache index")
    p_index.set_defaults(func=cmd_index)

    p_preview = sub.add_parser(
        "preview",
        help="Blender-only: render assets side by side to check scale / front facing",
    )
    p_preview.add_argument("--assets", nargs="+", required=True)
    p_preview.add_argument("--out", required=True)
    p_preview.add_argument("--meta", default=DEFAULT_META)
    p_preview.add_argument("--view", default="front", choices=["front", "side"])
    p_preview.add_argument("--samples", type=int, default=24)
    p_preview.add_argument("--force", action="store_true", help="rebuild asset libraries")
    p_preview.set_defaults(func=main_preview)

    args = parser.parse_args(argv)
    if args.command == "preview":
        try:
            import bpy  # noqa: F401
        except ImportError:
            print("the 'preview' command must run inside Blender:")
            print("  blender --background --factory-startup --python blender_agent/assets.py -- preview ...")
            return 2
    return args.func(args)


# --------------------------------------------------------------------------- #
# Blender side: import, normalise, library + instancing
# --------------------------------------------------------------------------- #
def _bpy():
    import bpy  # noqa: F401  (imported lazily so the host CLI works without bpy)

    return bpy


def _measured_bbox(objects):
    import mathutils

    corners = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ mathutils.Vector(corner))
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    return lo, hi


#: A footprint whose X/Y extents match the registry only after a transpose is a
#: 90 deg frame difference between the glTF and ``api_dimensions_mm`` -- a
#: source-model/library inconsistency, not a scale problem.  Cached Poly Haven
#: geometry and its registry agree to millimetres, so 5 cm is generous without
#: ever turning a genuinely different size into a rotation.
AXIS_ALIGN_TOL_M = 0.05


def registry_axis_yaw(size_m, api_dimensions_mm, tolerance: float = AXIS_ALIGN_TOL_M) -> tuple:
    """Yaw about +Z that puts a measured footprint into the registry frame.

    ``api_dimensions_mm`` is ``[width_x, depth_y, height_z]`` in the Blender
    frame after the glTF Y-up conversion.  A model exported 90 deg yawed
    against its own metadata has the X/Y footprint of the registry transposed
    (``modern_coffee_table_01`` is the cached example: the glTF measures
    0.60 x 1.20 m while the registry says 1.20 x 0.60 m).

    Returns ``(90.0, note)`` when the direct match is out of ``tolerance`` and
    a transpose brings both footprint axes and the height inside it; otherwise
    ``(0.0, None)``.  The rotation is always +90 deg, so an asset that needs
    the other quarter turn to face -Y declares ``yaw_deg: 180`` in
    ``asset_meta.json`` on top of this alignment.
    """
    try:
        if not api_dimensions_mm or len(api_dimensions_mm) != 3 or len(size_m) != 3:
            return 0.0, None
        api = [float(value) / 1000.0 for value in api_dimensions_mm]
        measured = [float(value) for value in size_m]
    except (TypeError, ValueError):
        return 0.0, None
    if any(value <= 0.0 for value in api):
        return 0.0, None
    direct = max(abs(measured[0] - api[0]), abs(measured[1] - api[1]))
    transposed = max(abs(measured[0] - api[1]), abs(measured[1] - api[0]))
    height_error = abs(measured[2] - api[2])
    if (direct <= tolerance or height_error > tolerance
            or transposed > tolerance or transposed >= direct):
        return 0.0, None
    return 90.0, (
        f"footprint {[round(v, 3) for v in measured[:2]]} m is the transpose of "
        f"the registry {[round(v, 3) for v in api[:2]]} m"
    )


def _measurement_warning(size, api_dims, front_yaw: float):
    """Axis-wise size drift, allowing for a quarter-turn facing correction."""
    try:
        api = [float(value) / 1000.0 for value in api_dims]
    except (TypeError, ValueError):
        return None
    if len(api) != 3 or any(value <= 1e-9 for value in api):
        return None
    quarter_turns = front_yaw / 90.0
    if abs(quarter_turns - round(quarter_turns)) < 1e-9 and int(round(quarter_turns)) % 2:
        # an odd quarter turn is a deliberate X/Y transpose of the front frame
        api = [api[1], api[0], api[2]]
    elif abs(quarter_turns - round(quarter_turns)) >= 1e-9:
        # a non-quarter-turn front correction cannot keep an axis-aligned AABB
        size, api = sorted(size), sorted(api)
    ratios = [m / a for m, a in zip(size, api)]
    if ratios and (max(ratios) > 1.15 or min(ratios) < 0.87):
        return (f"measured {[round(v, 3) for v in size]} m vs API "
                f"{[round(v, 3) for v in api]} m")
    return None


def normalise_asset(objects, asset_id: str, meta: dict = None, entry: dict = None) -> dict:
    """Rotate/scale/centre imported glTF objects; returns geometry facts.

    * registry-frame alignment: a glTF whose X/Y footprint is the transpose of
      ``api_dimensions_mm`` gets a +90 deg Z rotation, so the built extents are
      the registry extents axis for axis
    * per-asset front correction from ``asset_meta.json`` (``yaw_deg``),
      applied as a world-space Z rotation after the alignment (glTF imports
      are in quaternion rotation mode, so writing ``rotation_euler`` is a
      no-op and must not be used)
    * origin moved to the bottom-centre of the bounding box
    * sanity checks against the API dimensions and absolute plausibility
    """
    import math

    import mathutils

    bpy = _bpy()
    meta = (meta or {}).get(asset_id, {})
    yaw = float(meta.get("yaw_deg", 0.0))
    for obj in objects:  # shared mesh data would be transformed twice
        if obj.data.users > 1:
            obj.data = obj.data.copy()
    bpy.context.view_layer.update()
    raw_lo, raw_hi = _measured_bbox(objects)
    raw_size = [raw_hi[i] - raw_lo[i] for i in range(3)]
    axis_yaw, axis_note = registry_axis_yaw(
        raw_size, (entry or {}).get("api_dimensions_mm"))
    total_yaw = axis_yaw + yaw
    if total_yaw:
        rotation = mathutils.Matrix.Rotation(total_yaw * math.pi / 180.0, 4, "Z")
        for obj in objects:
            obj.matrix_world = rotation @ obj.matrix_world
        bpy.context.view_layer.update()
    # bake transforms so the instance geometry is already in the final pose
    for obj in objects:
        obj.data.transform(obj.matrix_world)
        obj.matrix_world = mathutils.Matrix.Identity(4)
    bpy.context.view_layer.update()

    lo, hi = _measured_bbox(objects)
    size = [hi[i] - lo[i] for i in range(3)]
    biggest = max(size)
    if not (0.02 <= biggest <= 20.0):
        raise RuntimeError(
            f"asset '{asset_id}' measures {biggest:.3f} m on its longest axis - not a plausible "
            f"furniture scale (units problem?), bbox {[round(v, 3) for v in size]}"
        )
    api_dims = (entry or {}).get("api_dimensions_mm")
    warning = _measurement_warning(size, api_dims, yaw) if api_dims else None
    offset = mathutils.Vector((-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]))
    for obj in objects:
        obj.data.transform(mathutils.Matrix.Translation(offset))
    bpy.context.view_layer.update()
    lo, hi = _measured_bbox(objects)
    return {
        "bbox_min": [round(v, 4) for v in lo],
        "bbox_max": [round(v, 4) for v in hi],
        "size_m": [round(v, 4) for v in size],
        "yaw_applied_deg": total_yaw,
        "axis_yaw_deg": axis_yaw,
        "front_yaw_deg": yaw,
        "axis_alignment": axis_note,
        "scale_warning": warning,
    }


def import_gltf_asset(gltf_path: str, collection_name: str):
    """Import a .gltf/.glb and join its meshes; returns (collection, [objects], facts)."""
    bpy = _bpy()
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=gltf_path)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"{gltf_path}: glTF import produced no mesh objects")
    for obj in imported:
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    for obj in meshes:
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        collection.objects.link(obj)
    return collection, meshes, imported


def _library_metadata_path(lib_path: str) -> str:
    return os.path.splitext(lib_path)[0] + ".json"


def _load_library_metadata(lib_path: str) -> dict:
    try:
        with open(_library_metadata_path(lib_path), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def library_is_current(lib_path: str, content_sha256: str) -> bool:
    """Whether a cached .blend was built from the cache's current content.

    A library without the sidecar (built before content fingerprinting, or
    copied without it) is never trusted: the cost of one rebuild is a missing
    texture, while the cost of a wrong reuse is a scene that does not match the
    spec.
    """
    if not (os.path.isfile(lib_path) and content_sha256):
        return False
    recorded = _load_library_metadata(lib_path)
    return (recorded.get("kind") == LIBRARY_METADATA_KIND
            and recorded.get("content_sha256") == content_sha256)


def ensure_library(cache_dir: str, asset_id: str, meta: dict = None, force: bool = False,
                   resolution: str = None) -> str:
    """Build ``library/<id>.<res>.blend`` (import + normalise + join) if not cached.

    Runs in a throw-away empty file so the library contains nothing but the
    asset; returns the library path.  ``resolution`` picks a glTF/texture variant
    (``1k`` by default at build time: it quarters the texture memory Cycles needs
    for a furnished scene).

    The library carries a ``<id>.<res>.json`` sidecar with the content digest of
    the model and its textures and the normalisation version that produced it.
    A cached ``.blend`` is reused only when both still match; replacing a file
    in place -- or changing the import/axis/facing convention -- rebuilds the
    library instead of silently reusing the old geometry.
    """
    bpy = _bpy()
    index = load_index(cache_dir)
    kind, entry = asset_entry(index, asset_id)
    if kind != "model":
        raise ValueError(f"'{asset_id}' is a texture, not a model")
    variants = entry.get("gltf_variants") or {"2k": entry["gltf"]}
    if resolution is None:
        resolution = "1k" if "1k" in variants else sorted(variants)[0]
    if resolution not in variants:
        raise ValueError(
            f"asset '{asset_id}' has no {resolution} variant (have {sorted(variants)}); "
            f"run: python blender_agent/assets.py fetch --assets ph:{asset_id} --resolution {resolution}"
        )
    content = content_digest_block(cache_dir)
    content_sha = (content["content"].get("models") or {}).get(asset_id)
    lib_dir = os.path.join(cache_dir, "library")
    lib_path = os.path.join(lib_dir, f"{asset_id}.{resolution}.blend")
    if (not force and library_is_current(lib_path, content_sha)
            and _load_library_metadata(lib_path).get("normalisation_version") == NORMALISATION_VERSION):
        return lib_path

    gltf_path = os.path.join(cache_dir, variants[resolution])
    if not os.path.isfile(gltf_path):
        hint = ("run: python tools/gen_assets_trellis.py generate"
                if entry.get("source") == "generated" else "run assets.py fetch first")
        raise FileNotFoundError(f"{gltf_path} missing; {hint}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    collection, meshes, _ = import_gltf_asset(gltf_path, f"asset_{asset_id}")
    facts = normalise_asset(meshes, asset_id, meta, entry)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = f"asset_{asset_id}"
    joined.data.name = f"mesh_{asset_id}"
    joined["ph_id"] = asset_id
    joined["ph_license"] = entry.get("license", "CC0")

    os.makedirs(lib_dir, exist_ok=True)
    # pack the glTF's textures so the library stays self-contained if the
    # download cache is pruned later
    for image in bpy.data.images:
        if image.source == "FILE" and not image.packed_file:
            try:
                image.pack()
            except RuntimeError:
                pass
    bpy.ops.wm.save_as_mainfile(filepath=lib_path, compress=True, copy=True)
    entry_record = content["entries"].get(f"model/{asset_id}") or {}
    _write_json_atomic(_library_metadata_path(lib_path), {
        "kind": LIBRARY_METADATA_KIND,
        "schema_version": CONTENT_MANIFEST_SCHEMA,
        "asset": asset_id,
        "resolution": resolution,
        "normalisation_version": NORMALISATION_VERSION,
        "content_sha256": content_sha,
        "files": entry_record.get("files") or [],
    })
    facts["library"] = lib_path
    facts["content_sha256"] = content_sha
    facts["library_metadata"] = _library_metadata_path(lib_path)
    facts["polycount"] = len(joined.data.polygons)
    facts["materials"] = [slot.material.name for slot in joined.material_slots if slot.material]
    print(
        f"[assets] built library {os.path.basename(lib_path)}: {joined.name} "
        f"{facts['polycount']} faces, size {facts['size_m']} m"
        + (f" WARNING {facts['scale_warning']}" if facts["scale_warning"] else "")
    )
    return lib_path


def append_asset(library_path: str, asset_id: str):
    """Append the asset collection from a library .blend into the current file."""
    bpy = _bpy()
    name = f"asset_{asset_id}"
    collection = bpy.data.collections.get(name)
    if collection is not None:
        return collection
    with bpy.data.libraries.load(library_path, link=False) as (data_from, data_to):
        if name not in data_from.collections:
            raise RuntimeError(f"{library_path} does not contain collection '{name}'")
        data_to.collections = [name]
    collection = bpy.data.collections[name]
    return collection


def place_asset(collection, name: str, pos, rot_z_deg: float = 0.0, lift_m: float = 0.0,
                scale: float = 1.0):
    """Drop a collection instance of ``collection`` into the scene.

    ``pos`` is the (x, y) floor position; the asset's bottom-centre lands there
    at ``z = lift_m``.  Instancing shares mesh + texture data between repeats.
    """
    bpy = _bpy()
    empty = bpy.data.objects.new(name, None)
    empty.instance_type = "COLLECTION"
    empty.instance_collection = collection
    empty.empty_display_size = 0.2
    empty.location = (float(pos[0]), float(pos[1]), float(lift_m))
    empty.rotation_euler = (0.0, 0.0, float(rot_z_deg) * math.pi / 180.0)
    empty.scale = (float(scale),) * 3
    bpy.context.scene.collection.objects.link(empty)
    return empty


def ensure_libraries(cache_dir: str, asset_ids, meta: dict = None, force: bool = False,
                     resolution: str = None) -> dict:
    """Build every requested library first: ``ensure_library`` resets the file,
    so it must not run while a scene is being composed."""
    paths = {}
    for asset_id in asset_ids:
        paths[asset_id] = ensure_library(cache_dir, asset_id, meta, force=force, resolution=resolution)
    return paths


def preview_assets(cache_dir: str, asset_ids, out_path: str, meta_path: str = DEFAULT_META,
                   view: str = "front", samples: int = 24, spacing: float = None,
                   resolution=(1600, 360), force_libraries: bool = False) -> dict:
    """Render every asset side by side to check scale and the 'front faces -Y' rule.

    ``view='front'`` looks from -Y (each asset should show its face), ``view='side'``
    from +X.  Returns the row->asset mapping so the sheet can be read back.
    """
    import math

    bpy = _bpy()
    meta = load_asset_meta(meta_path)
    index = load_index(cache_dir)
    for asset_id in asset_ids:
        kind, entry = asset_entry(index, asset_id)
        if kind != "model":
            raise ValueError(f"'{asset_id}' is not a model")

    if spacing is None:
        sizes = []
        for asset_id in asset_ids:
            dims = index["models"][asset_id].get("api_dimensions_mm") or [1000, 1000, 1000]
            sizes.append(max(float(v) for v in dims) / 1000.0)
        spacing = max(2.2, 1.35 * max(sizes))

    libraries = ensure_libraries(cache_dir, asset_ids, meta, force=force_libraries)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    world = bpy.data.worlds.new("preview_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.55, 0.58, 1.0)
    scene.world = world
    lamp_data = bpy.data.lights.new("preview_sun", type="SUN")
    lamp_data.energy = 3.5
    lamp = bpy.data.objects.new("preview_sun", lamp_data)
    lamp.rotation_euler = (math.radians(50), 0.0, math.radians(-125))
    scene.collection.objects.link(lamp)

    rows = {}
    for position, asset_id in enumerate(asset_ids):
        collection = append_asset(libraries[asset_id], asset_id)
        obj = place_asset(collection, f"preview_{asset_id}", (position * spacing, 0.0, 0.0))
        rows[asset_id] = position
        print(f"[assets] preview row {position}: {asset_id} size {collection.objects[0].dimensions[:]}")

    camera_data = bpy.data.cameras.new("preview_cam")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = spacing * (len(asset_ids) + 0.2)
    camera = bpy.data.objects.new("preview_cam", camera_data)
    scene.collection.objects.link(camera)
    midpoint = ((len(asset_ids) - 1) * spacing / 2.0, 0.0, 0.0)
    distance = 12.0
    if view == "front":
        camera.location = (midpoint[0], -distance, distance * 0.28)
        camera.rotation_euler = (math.radians(74), 0.0, 0.0)
    else:
        camera.location = (midpoint[0] + distance, 0.0, distance * 0.28)
        camera.rotation_euler = (math.radians(74), 0.0, math.radians(90))
    scene.camera = camera

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 16
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"[assets] preview written: {out_path} ({len(asset_ids)} assets, spacing {spacing:.2f} m)")
    return {"path": out_path, "rows": rows, "spacing_m": spacing, "view": view}


def main_preview(args) -> int:
    preview_assets(args.cache, [split_asset_ref(r) for r in args.assets], args.out, args.meta,
                   view=args.view, samples=args.samples, force_libraries=args.force)
    return 0


if __name__ == "__main__":
    # host CLI: plain argv; inside Blender everything after "--" belongs to us
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = sys.argv[1:]
    sys.exit(main(argv))
