#!/usr/bin/env python
"""Numeric validation of a blender_agent P0 build against the PanoWorld format.

Checks, in order:

1. every viewpoint file exists (``panoImage_2048.png``, ``place_depth.png``,
   ``place_depth_scale.txt``, ``extrinsics.txt``) plus ``map.json`` and
   ``topdown.png``
2. image / depth / extrinsics formats and value ranges
3. camera poses match the spec exactly
4. **depth numerics**: for every camera, all ERP rays are cast against
   ``build/shell.ply`` and compared with ``place_depth.png / scale``
5. the door opening is deeper than the wall next to it in the same column
6. ``map.json`` is a symmetric graph over the supplied cameras

Ground truth ray casting note: trimesh's own ``RayMeshIntersector`` needs the
optional ``rtree`` package, which is not part of this environment.  The caster
below is an exact, dependency-free replacement: a broad phase over mesh
components (``trimesh`` connected components, ordered near to far with a
running best-distance prune) plus, for the large enclosing component, an
analytic bounding-sphere estimate that limits triangle tests to the facets
around the hit.  A brute-force cast over *all* triangles on a random ray sample
verifies that acceleration in the same run (``--sample-rays``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from blender_agent import spec as spec_mod  # noqa: E402

EPS_DET = 1e-12
T_MIN = 1e-9


# --------------------------------------------------------------------------- #
# ray casting
# --------------------------------------------------------------------------- #
def _ray_aabb_tmin(origins: np.ndarray, directions: np.ndarray, bmin: np.ndarray, bmax: np.ndarray):
    """Slab test: (hit mask, t_enter) for rays against one axis-aligned box."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / directions
        t0 = (bmin[None, :] - origins) * inv
        t1 = (bmax[None, :] - origins) * inv
    t_near = np.minimum(t0, t1)
    t_far = np.maximum(t0, t1)
    t_near = np.where(np.isnan(t_near), -np.inf, t_near)
    t_far = np.where(np.isnan(t_far), np.inf, t_far)
    t_enter = np.max(t_near, axis=1)
    t_exit = np.min(t_far, axis=1)
    hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 0.0)
    return hit, np.maximum(t_enter, 0.0)


def _ray_aabb_tmin_arrays(origins: np.ndarray, directions: np.ndarray,
                          bmin: np.ndarray, bmax: np.ndarray):
    """Slab test where every ray carries its own box (BVH traversal)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / directions
        t0 = (bmin - origins) * inv
        t1 = (bmax - origins) * inv
    t_near = np.minimum(t0, t1)
    t_far = np.maximum(t0, t1)
    t_near = np.where(np.isnan(t_near), -np.inf, t_near)
    t_far = np.where(np.isnan(t_far), np.inf, t_far)
    t_enter = np.max(t_near, axis=1)
    t_exit = np.min(t_far, axis=1)
    hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 0.0)
    return hit, np.maximum(t_enter, 0.0)


def _moller_trumbore(origins: np.ndarray, directions: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Closest intersection distance with one triangle (``inf`` when missed)."""
    if triangle.ndim == 2:
        triangle = np.broadcast_to(triangle, (len(origins), 3, 3))
    v0 = triangle[:, 0]
    edge1 = triangle[:, 1] - v0
    edge2 = triangle[:, 2] - v0
    pvec = np.cross(directions, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    ok = np.abs(det) > EPS_DET
    inv_det = np.zeros_like(det)
    inv_det[ok] = 1.0 / det[ok]
    tvec = origins - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", directions, qvec) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (t > T_MIN)
    return np.where(hit, t, np.inf)


def _ray_aabb_many_boxes(origins: np.ndarray, directions: np.ndarray,
                         bmin: np.ndarray, bmax: np.ndarray):
    """Slab test of every ray against many boxes: (n_rays, n_boxes) masks."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / directions
        t0 = (bmin[None, :, :] - origins[:, None, :]) * inv[:, None, :]
        t1 = (bmax[None, :, :] - origins[:, None, :]) * inv[:, None, :]
    t_near = np.minimum(t0, t1)
    t_far = np.maximum(t0, t1)
    t_near = np.where(np.isnan(t_near), -np.inf, t_near)
    t_far = np.where(np.isnan(t_far), np.inf, t_far)
    t_enter = t_near.max(axis=2)
    t_exit = t_far.min(axis=2)
    hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 0.0)
    return hit, np.maximum(t_enter, 0.0)


def _moller_trumbore_many(origins: np.ndarray, directions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """(n_rays, n_triangles) distance matrix for one BVH leaf, ``inf`` on miss.

    Batched because a leaf holds hundreds of triangles and thousands of rays:
    looping per triangle costs tens of microseconds of numpy overhead each.
    """
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    pvec = np.cross(directions[:, None, :], edge2[None, :, :])
    det = np.einsum("nmk,mk->nm", pvec, edge1)
    ok = np.abs(det) > EPS_DET
    inv_det = np.zeros_like(det)
    np.divide(1.0, det, out=inv_det, where=ok)
    tvec = origins[:, None, :] - v0[None, :, :]
    u = np.einsum("nmk,nmk->nm", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1[None, :, :])
    v = np.einsum("nmk,nmk->nm", directions[:, None, :], qvec) * inv_det
    t = np.einsum("nmk,mk->nm", qvec, edge2) * inv_det
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (t > T_MIN)
    return np.where(hit, t, np.inf)


class MeshRayCaster:
    """Exact nearest-hit distance along a ray, accelerated by a median-split BVH.

    The furnished scenes reach ~750k triangles, so the previous
    component-AABB broad phase is replaced by a real BVH: triangles are split
    along the longest axis at the centroid median until a leaf holds at most
    ``max_leaf`` triangles, and rays are pushed through the tree breadth-first
    (vectorised per level) with a running best-distance prune.
    """

    def __init__(self, mesh, origin_hints=None, max_leaf: int = 32, group_count: int = 64,
                 far_ratio: float = 0.55):
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("mesh did not produce triangle geometry")
        self.triangles = triangles
        self.max_leaf = int(max_leaf)

        # Split the scene into "near" geometry (room + furniture) and the far
        # enclosure (the sky dome).  The dome is a thin sphere shell far outside
        # the room: putting it in the BVH makes its huge bounding box pass every
        # ray, so it is handled analytically with a centroid kNN instead, exactly
        # like the P0 caster did.
        vertices = triangles.reshape(-1, 3)
        centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        centroid_distances = np.linalg.norm(triangles.mean(axis=1) - centre[None, :], axis=1)
        threshold = float(centroid_distances.max()) * far_ratio
        far = centroid_distances > threshold
        if not far.any() or far.all():
            far = np.zeros(len(triangles), dtype=bool)
        self.far_triangles = triangles[far].astype(np.float32)
        self.near_triangles = triangles[~far]
        if self.far_triangles.size:
            from scipy.spatial import cKDTree

            far_centroids = self.far_triangles.mean(axis=1)
            far_vertices = self.far_triangles.reshape(-1, 3)
            self.sphere_center = (far_vertices.min(axis=0) + far_vertices.max(axis=0)) / 2.0
            self.sphere_radius = float(np.max(np.linalg.norm(far_vertices - self.sphere_center, axis=1)))
            self.facet_tree = cKDTree(far_centroids)
            self.facet_k = int(min(128, len(far_centroids)))

        self._build_bvh(self.near_triangles if self.near_triangles.size else triangles)

    def _build_bvh(self, triangles: np.ndarray):
        # float32: the depth comparison targets millimetres while float32 keeps
        # micrometre precision here, and it halves the memory traffic of the
        # hundreds of millions of ray/triangle candidate tests
        self.bvh_triangles = triangles.astype(np.float32)
        triangles = self.bvh_triangles
        centroids = triangles.mean(axis=1)
        bounds = np.stack([triangles.min(axis=1), triangles.max(axis=1)], axis=1)
        self.order = np.arange(len(triangles))
        self.node_bmin = []
        self.node_bmax = []
        self.node_left = []
        self.node_right = []
        self.node_start = []
        self.node_end = []
        stack = [(0, len(triangles), -1, 0)]
        while stack:
            start, end, parent, side = stack.pop()
            index = self.order[start:end]
            node = len(self.node_bmin)
            self.node_bmin.append(bounds[index, 0].min(axis=0))
            self.node_bmax.append(bounds[index, 1].max(axis=0))
            self.node_start.append(start)
            self.node_end.append(end)
            self.node_left.append(-1)
            self.node_right.append(-1)
            if parent >= 0:  # link from the parent that queued this node
                if side == 0:
                    self.node_left[parent] = node
                else:
                    self.node_right[parent] = node
            if end - start <= self.max_leaf:
                continue
            block = centroids[index]
            extent = block.max(axis=0) - block.min(axis=0)
            axis = int(np.argmax(extent))
            if extent[axis] <= 1e-9:
                continue
            order_local = np.argpartition(block[:, axis], (end - start) // 2)
            self.order[start:end] = index[order_local]
            middle = start + (end - start) // 2
            # children are numbered when they are created, so pass the parent
            # link down instead of predicting the child indices
            stack.append((middle, end, node, 1))
            stack.append((start, middle, node, 0))
        self.node_bmin = np.asarray(self.node_bmin, dtype=np.float32)
        self.node_bmax = np.asarray(self.node_bmax, dtype=np.float32)
        self.node_left = np.asarray(self.node_left, dtype=np.int64)
        self.node_right = np.asarray(self.node_right, dtype=np.int64)
        self.is_leaf = self.node_left < 0
        self.leaf_nodes = np.flatnonzero(self.is_leaf)

    def _build_groups(self, group_count: int = 64):
        """Split the tree into a shallow group level for vectorised queries.

        A ray-synchronous walk over the full tree degenerates into millions of
        tiny numpy calls; instead the tree is cut at a fixed number of top nodes
        and every query tests those groups against all rays at once, then each
        group's leaves against just the rays that reached it.  Groups and leaves
        are walked in near-to-far order so the running best distance prunes the
        rest of the scene away.
        """
        leaves = np.flatnonzero(self.is_leaf)
        stack = [0]
        depth_nodes = [np.array([0], dtype=np.int64)]
        while len(depth_nodes[-1]) < group_count:
            children = []
            for node in depth_nodes[-1]:
                if self.is_leaf[node]:
                    children.append(np.array([node], dtype=np.int64))
                else:
                    children.append(np.array([self.node_left[node], self.node_right[node]], dtype=np.int64))
            nxt = np.concatenate(children) if children else np.array([], dtype=np.int64)
            if nxt.size == 0:
                break
            depth_nodes.append(nxt)
        frontier = depth_nodes[-1]
        groups = []
        for group_node in frontier:
            members = []
            stack = [int(group_node)]
            while stack:
                node = stack.pop()
                if self.is_leaf[node]:
                    members.append(node)
                else:
                    stack.append(int(self.node_left[node]))
                    stack.append(int(self.node_right[node]))
            groups.append({"node": int(group_node), "leaves": np.asarray(members, dtype=np.int64)})
        self.groups = groups
        total_leaves = sum(len(g["leaves"]) for g in groups)
        assert total_leaves == len(leaves), (total_leaves, len(leaves))

    def _order_near_to_far(self, origin: np.ndarray) -> None:
        """(Re)sort groups and their leaves by distance from a ray origin."""
        centres = np.asarray([
            (self.node_bmin[group["node"]] + self.node_bmax[group["node"]]) / 2.0
            for group in self.groups
        ])
        order = np.argsort(np.linalg.norm(centres - origin[None, :], axis=1))
        self.groups = [self.groups[i] for i in order]
        for group in self.groups:
            leaves = group["leaves"]
            leaf_centres = (self.node_bmin[leaves] + self.node_bmax[leaves]) / 2.0
            leaf_order = np.argsort(np.linalg.norm(leaf_centres - origin[None, :], axis=1))
            group["leaves"] = leaves[leaf_order]

    def _leaf_triangles(self, node: int) -> np.ndarray:
        return self.bvh_triangles[self.order[self.node_start[node] : self.node_end[node]]]

    def _sphere_exit(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        oc = origins - self.sphere_center[None, :]
        b = np.einsum("ij,ij->i", oc, directions)
        c = np.einsum("ij,ij->i", oc, oc) - self.sphere_radius ** 2
        disc = b * b - c
        valid = disc >= 0.0
        root = np.sqrt(np.where(valid, disc, 0.0))
        t_far = -b + root
        return np.where(valid & (t_far > T_MIN), t_far, np.inf)

    def _cast_far(self, origins: np.ndarray, directions: np.ndarray, t_best: np.ndarray) -> None:
        """Refine rays whose best near hit is at or behind the enclosure sphere."""
        t_sphere = self._sphere_exit(origins, directions)
        candidates = np.flatnonzero(~np.isfinite(t_sphere) | (t_best > t_sphere - 0.1))
        if candidates.size == 0:
            return
        step = max(1, int(4_000_000 // self.facet_k))
        for start in range(0, candidates.size, step):
            index = candidates[start : start + step]
            t_exit = t_sphere[index]
            points = origins[index] + directions[index] * t_exit[:, None]
            bad = ~np.isfinite(t_exit)
            if bad.any():
                points[bad] = self.sphere_center[None, :] + directions[index][bad] * self.sphere_radius
            _, neighbours = self.facet_tree.query(points, k=self.facet_k)
            if self.facet_k == 1:
                neighbours = neighbours[:, None]
            best = t_best[index].copy()
            for slot in range(self.facet_k):
                triangle = np.asarray(self.far_triangles[neighbours[:, slot]], dtype=np.float64)
                t = _moller_trumbore(origins[index], directions[index], triangle)
                best = np.minimum(best, t)
            t_best[index] = best

    def cast(self, origins: np.ndarray, directions: np.ndarray, chunk: int = 32_768) -> np.ndarray:
        """Nearest-hit distance for every ray (BVH for the room, sphere for the dome)."""
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        result = np.full(len(origins), np.inf)
        for start in range(0, len(origins), chunk):
            o = origins[start : start + chunk].astype(np.float32)
            d = directions[start : start + chunk].astype(np.float32)
            t_best = np.full(len(o), np.inf, dtype=np.float32)
            all_rays = np.arange(len(o), dtype=np.int64)
            frontier = [(all_rays, 0)]
            leaf_tasks = []
            while frontier:
                following = []
                for rays, node in frontier:
                    if self.is_leaf[node]:
                        leaf_tasks.append((rays, int(node)))
                        continue
                    left = int(self.node_left[node])
                    right = int(self.node_right[node])
                    # expand two levels at once when possible: fewer numpy calls,
                    # same pruning (the pairs are already near-optimal)
                    if not self.is_leaf[left] and not self.is_leaf[right]:
                        children = np.array(
                            [self.node_left[left], self.node_right[left],
                             self.node_left[right], self.node_right[right]], dtype=np.int64
                        )
                    else:
                        children = np.array([left, right], dtype=np.int64)
                    hit, t_enter = _ray_aabb_many_boxes(
                        o[rays], d[rays], self.node_bmin[children], self.node_bmax[children]
                    )
                    hit &= t_enter <= t_best[rays][:, None]
                    for slot, child in enumerate(children):
                        selected = rays[hit[:, slot]]
                        if selected.size:
                            following.append((selected, int(child)))
                frontier = following
            for rays, node in leaf_tasks:
                sub_o, sub_d = o[rays], d[rays]
                leaf_triangles = self._leaf_triangles(node)
                step = max(1, int(1_500_000 // max(1, len(leaf_triangles))))
                best = t_best[rays].copy()
                for offset in range(0, len(rays), step):
                    block = slice(offset, offset + step)
                    distances = _moller_trumbore_many(sub_o[block], sub_d[block], leaf_triangles)
                    best[block] = np.minimum(best[block], distances.min(axis=1))
                t_best[rays] = best
            if getattr(self, "far_triangles", np.empty(0)).size:
                self._cast_far(o.astype(np.float64), d.astype(np.float64),
                               t_best.astype(np.float64))
            result[start : start + chunk] = t_best.astype(np.float64)
        return result

    def cast_bruteforce(self, origins: np.ndarray, directions: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """Reference implementation: every ray against every triangle."""
        result = np.full(len(origins), np.inf)
        for start in range(0, len(origins), chunk):
            o = origins[start : start + chunk]
            d = directions[start : start + chunk]
            best = np.full(len(o), np.inf)
            for triangle in self.triangles:
                best = np.minimum(best, _moller_trumbore(o, d, triangle))
            result[start : start + chunk] = best
        return result


def cast_with_trimesh(mesh, origins, directions, chunk=200_000):
    """Optional second opinion via trimesh's rtree-backed intersector."""
    try:
        import rtree  # noqa: F401
    except Exception:
        return None
    distance = np.full(len(origins), np.inf)
    for start in range(0, len(origins), chunk):
        o = origins[start : start + chunk]
        d = directions[start : start + chunk]
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=o, ray_directions=d, multiple_hits=False
        )
        if len(index_ray):
            distance[start + index_ray] = np.linalg.norm(locations - o[index_ray], axis=1)
    return distance


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def erp_rays(c2w: np.ndarray, width: int, height: int):
    """Rays for every ERP pixel, exactly as ``compute_plucmap_pano`` does."""
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    lon = (u + 0.5) / width * 2.0 * np.pi - np.pi
    lat = np.pi / 2.0 - (v + 0.5) / height * np.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)
    local = np.stack(
        [cos_lat * np.sin(lon_grid), -np.sin(lat_grid), cos_lat * np.cos(lon_grid)], axis=0
    ).reshape(3, -1)
    directions = (c2w[:3, :3] @ local).T
    origins = np.tile(c2w[:3, 3], (directions.shape[0], 1))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return origins, directions / norms


def read_depth_meters(path: str, scale: float):
    array = np.array(Image.open(path))
    if array.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16 PNG, got {array.dtype}")
    return array.astype(np.float64) / scale, array > 0


def lat_row_to_pixel(lat_rad: float, height: int) -> float:
    return (math.pi / 2.0 - lat_rad) / math.pi * height - 0.5


def resolve_render_size(build_dir: str) -> tuple:
    """(width, height) the build was rendered at, from its manifest."""
    manifest_path = os.path.join(build_dir, "build_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        render = manifest.get("render", {})
        if "width" in render and "height" in render:
            return int(render["width"]), int(render["height"])
    return spec_mod.PANO_WIDTH, spec_mod.PANO_HEIGHT


def door_spot_check(spec: spec_mod.SceneSpec, camera: spec_mod.CameraSpec, rendered: np.ndarray) -> dict:
    """Compare the depth through the door with the wall above it, same column."""
    opening = spec.openings[0] if spec.openings else None
    if opening is None:
        return {"pass": True, "skipped": "spec has no openings"}
    segment = spec_mod.wall_segment_for_point(spec.wall_segments(), opening.center, opening.between[0])
    if segment is None:
        return {"pass": True, "skipped": "door wall segment not found"}
    height, width = rendered.shape
    cam = np.array(camera.pos, dtype=np.float64)
    direction = np.array(segment.direction, dtype=np.float64)
    normal = np.array([-direction[1], direction[0]])
    door = np.array(opening.center, dtype=np.float64)

    to_door = door - cam[:2]
    length = float(np.linalg.norm(to_door))
    if length < 1e-6:
        return {"pass": True, "skipped": "camera sits on the door centre"}
    unit = to_door / length
    denominator = float(np.dot(unit, normal))
    if abs(denominator) < 1e-6:
        return {"pass": True, "skipped": "camera looks along the door wall"}
    # horizontal distance from the camera to the wall plane along this column
    horizontal = float(np.dot(door - cam[:2], normal)) / denominator

    column = int(round((math.atan2(to_door[1], to_door[0]) + math.pi) / (2 * math.pi) * width - 0.5))
    column = column % width

    def depth_at(lat_deg: float) -> float:
        row = int(round(lat_row_to_pixel(math.radians(lat_deg), height)))
        row = max(2, min(height - 3, row))
        col = max(2, min(width - 3, column))
        return float(np.median(rendered[row - 2 : row + 3, col - 2 : col + 3]))

    door_z_min, door_z_max = 0.05, opening.height - 0.05
    lat_door = math.degrees(math.atan2(((door_z_min + door_z_max) / 2.0) - cam[2], horizontal))
    wall_z_min, wall_z_max = opening.height + 0.1, spec.wall_height - 0.05
    lat_wall = math.degrees(math.atan2(((wall_z_min + wall_z_max) / 2.0) - cam[2], horizontal))
    door_depth = depth_at(lat_door)
    wall_depth = depth_at(lat_wall)
    ok = door_depth > 1.5 * wall_depth and door_depth > 0.0 and wall_depth > 0.0
    return {
        "pass": bool(ok),
        "column": int(column),
        "wall_distance_m": horizontal,
        "door_row": int(round(lat_row_to_pixel(math.radians(lat_door), height))),
        "wall_row": int(round(lat_row_to_pixel(math.radians(lat_wall), height))),
        "door_depth_m": door_depth,
        "wall_depth_m": wall_depth,
        "ratio": (door_depth / wall_depth) if wall_depth > 0 else None,
    }


# --------------------------------------------------------------------------- #
# main validation
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a blender_agent P0 export")
    parser.add_argument("--build", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--sample-rays", type=int, default=4000, help="rays used for the brute-force cross-check")
    parser.add_argument("--max-panos", type=int, default=0, help="limit the number of cameras checked (debug)")
    args = parser.parse_args(argv)

    spec = spec_mod.load_spec(args.spec)
    build_dir = os.path.abspath(args.build)
    scene_dir = os.path.abspath(args.scene)
    cameras = spec.cameras if args.max_panos <= 0 else spec.cameras[: args.max_panos]
    render_width, render_height = resolve_render_size(build_dir)

    checks = []

    def record(name: str, passed: bool, **details):
        checks.append({"name": name, "pass": bool(passed), **details})
        return passed

    record(
        "canonical_resolution",
        (render_width, render_height) == (spec_mod.PANO_WIDTH, spec_mod.PANO_HEIGHT),
        render_size=[render_width, render_height],
        expected=[spec_mod.PANO_WIDTH, spec_mod.PANO_HEIGHT],
    )

    # ---- 1/2/3: files, formats, poses ------------------------------------- #
    missing = []
    for camera in cameras:
        for name in ("panoImage_2048.png", "place_depth.png", "place_depth_scale.txt", "extrinsics.txt"):
            path = os.path.join(scene_dir, "viewpoints", camera.name, name)
            if not os.path.isfile(path):
                missing.append(os.path.relpath(path, scene_dir))
    for name in ("map.json", "topdown.png"):
        if not os.path.isfile(os.path.join(scene_dir, name)):
            missing.append(name)
    record("files_present", not missing, missing=missing)

    pano_ok, pano_details = True, {}
    depth_ok, depth_details = True, {}
    extr_ok, extr_details = True, {}
    pose_ok, pose_details = True, {}
    extrinsics = {}
    for camera in cameras:
        view_dir = os.path.join(scene_dir, "viewpoints", camera.name)
        pano = Image.open(os.path.join(view_dir, "panoImage_2048.png"))
        if pano.size != (render_width, render_height) or pano.mode not in ("RGB", "RGBA"):
            pano_ok = False
            pano_details[camera.name] = {"size": pano.size, "mode": pano.mode}

        raw_depth = np.array(Image.open(os.path.join(view_dir, "place_depth.png")))
        with open(os.path.join(view_dir, "place_depth_scale.txt"), "r", encoding="utf-8") as handle:
            scale = float(handle.read().strip())
        if raw_depth.dtype != np.uint16 or raw_depth.shape != (render_height, render_width) or scale <= 0:
            depth_ok = False
            depth_details[camera.name] = {"dtype": str(raw_depth.dtype), "shape": list(raw_depth.shape), "scale": scale}

        matrix = np.loadtxt(os.path.join(view_dir, "extrinsics.txt"))
        extrinsics[camera.name] = matrix
        rotation = matrix[:3, :3]
        det = float(np.linalg.det(rotation))
        orthogonal = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
        expected = np.array(spec_mod.extrinsics_matrix(camera.pos, 0.0))
        pose_error = float(np.max(np.abs(matrix - expected)))
        if matrix.shape != (4, 4) or abs(det - 1.0) > 1e-6 or orthogonal > 1e-6:
            extr_ok = False
            extr_details[camera.name] = {"shape": list(matrix.shape), "det": det, "orthogonality": orthogonal}
        if pose_error > 1e-6:
            pose_ok = False
            pose_details[camera.name] = {"max_abs_error": pose_error}
    record("pano_format", pano_ok, **pano_details)
    record("depth_format", depth_ok, **depth_details)
    record("extrinsics_valid", extr_ok, **extr_details)
    record("camera_pose_matches_spec", pose_ok, **pose_details)

    # ---- 4: depth numerics ------------------------------------------------ #
    mesh = None
    caster = None
    ply_path = None
    manifest_path = os.path.join(build_dir, "build_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            build_manifest = json.load(handle)
        candidate = (build_manifest.get("visible_geometry") or {}).get("ply")
        if candidate:
            ply_path = os.path.join(build_dir, candidate)
    if ply_path is None or not os.path.isfile(ply_path):
        ply_path = os.path.join(build_dir, "shell.ply")
    depth_reports = []
    if os.path.isfile(ply_path):
        import trimesh

        mesh = trimesh.load(ply_path)
        if hasattr(mesh, "geometry"):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        hint = np.array([cam.pos for cam in cameras], dtype=np.float64)
        caster = MeshRayCaster(mesh, hint)

    for camera in cameras:
        view_dir = os.path.join(scene_dir, "viewpoints", camera.name)
        raw_depth = np.array(Image.open(os.path.join(view_dir, "place_depth.png")))
        with open(os.path.join(view_dir, "place_depth_scale.txt"), "r", encoding="utf-8") as handle:
            scale = float(handle.read().strip())
        rendered, valid = read_depth_meters(os.path.join(view_dir, "place_depth.png"), scale)
        c2w = extrinsics[camera.name]
        origins, directions = erp_rays(c2w, render_width, render_height)

        started = time.time()
        truth = caster.cast(origins, directions).reshape(render_height, render_width)
        seconds = time.time() - started
        truth_valid = np.isfinite(truth)

        both = truth_valid & valid
        invalid_render = truth_valid & ~valid
        invalid_truth = valid & ~truth_valid
        valid_fraction = float(both.mean())
        error = np.abs(rendered[both] - truth[both])
        median = float(np.median(error)) if error.size else float("nan")
        p95 = float(np.percentile(error, 95)) if error.size else float("nan")
        worst = float(error.max()) if error.size else float("nan")

        spot = door_spot_check(spec, camera, rendered)
        passed = valid_fraction >= 0.995 and median < 0.005 and p95 < 0.02 and spot["pass"]
        depth_reports.append(
            {
                "camera": camera.name,
                "valid_fraction": valid_fraction,
                "pixels": int(rendered.size),
                "valid_pixels": int(both.sum()),
                "invalid_in_render": int(invalid_render.sum()),
                "no_geometry_hit": int(invalid_truth.sum()),
                "median_error_m": median,
                "p95_error_m": p95,
                "max_error_m": worst,
                "cast_seconds": round(seconds, 2),
                "door_spot_check": spot,
                "pass": bool(passed),
            }
        )
        print(
            f"[validate] camera {camera.name}: valid={valid_fraction * 100:.3f}% "
            f"median|d|={median * 1000:.3f}mm p95|d|={p95 * 1000:.3f}mm max|d|={worst * 1000:.1f}mm "
            f"invalid_render={int(invalid_render.sum())} miss={int(invalid_truth.sum())} "
            f"door/wall={spot.get('door_depth_m', float('nan')):.2f}/{spot.get('wall_depth_m', float('nan')):.2f}m "
            f"({seconds:.1f}s)"
        )

    thresholds = {"valid_fraction_min": 0.995, "median_error_max_m": 0.005, "p95_error_max_m": 0.02}
    record(
        "depth_matches_geometry",
        bool(depth_reports) and all(item["pass"] for item in depth_reports),
        thresholds=thresholds,
        cameras=depth_reports,
        ply=os.path.relpath(ply_path, REPO_ROOT) if os.path.isfile(ply_path) else None,
    )

    # ---- caster cross-check ---------------------------------------------- #
    cross = {"available": bool(caster is not None)}
    if caster is not None and args.sample_rays > 0:
        rng = np.random.default_rng(0)
        camera = cameras[0]
        origins, directions = erp_rays(extrinsics[camera.name], render_width, render_height)
        # brute force costs rays x triangles, so scale the sample to the mesh
        budget = max(50, min(args.sample_rays, int(3e8 / max(1, len(caster.triangles)))))
        sample = rng.choice(len(origins), size=min(budget, len(origins)), replace=False)
        started = time.time()
        brute = caster.cast_bruteforce(origins[sample], directions[sample])
        brute_seconds = time.time() - started
        fast = caster.cast(origins[sample], directions[sample])
        both_finite = np.isfinite(brute) & np.isfinite(fast)
        mismatch = np.isinf(brute) != np.isinf(fast)
        delta = float(np.max(np.abs(brute[both_finite] - fast[both_finite]))) if both_finite.any() else 0.0
        cross.update(
            {
                "sample_rays": int(len(sample)),
                "triangles": int(len(caster.triangles)),
                "brute_force_seconds": round(brute_seconds, 2),
                "max_abs_delta_m": delta,
                "hit_mask_mismatches": int(mismatch.sum()),
                "brute_force_pass": bool(not mismatch.any()) and delta < 1e-4,
                "delta_note": "float32 BVH vs float64 brute force",
            }
        )
        engine = cast_with_trimesh(mesh, origins[sample], directions[sample])
        if engine is not None:
            both_finite = np.isfinite(engine) & np.isfinite(fast)
            cross["trimesh_rtree_max_abs_delta_m"] = (
                float(np.max(np.abs(engine[both_finite] - fast[both_finite]))) if both_finite.any() else None
            )
            cross["trimesh_rtree_available"] = True
        else:
            cross["trimesh_rtree_available"] = False
            cross["trimesh_rtree_note"] = "rtree not installed; trimesh's RayMeshIntersector was unavailable"
        record("caster_cross_check", bool(cross.get("brute_force_pass")), **cross)
    else:
        record("caster_cross_check", False, available=False)

    # ---- 5: map ----------------------------------------------------------- #
    map_path = os.path.join(scene_dir, "map.json")
    map_ok, map_details = False, {}
    if os.path.isfile(map_path):
        with open(map_path, "r", encoding="utf-8") as handle:
            graph = json.load(handle)
        expected = spec.camera_adjacency()
        map_details = {"graph": graph, "expected": expected}
        map_ok = set(graph) == set(expected) and all(
            sorted(graph.get(node, [])) == expected[node] for node in expected
        )
        map_ok = map_ok and all(node in graph.get(peer, []) for node, peers in graph.items() for peer in peers)
    record("map_json_reachable", bool(map_ok), **map_details)

    # ---- topdown sanity --------------------------------------------------- #
    topdown = os.path.join(scene_dir, "topdown.png")
    if os.path.isfile(topdown):
        image = np.array(Image.open(topdown).convert("L"))
        record("topdown_non_empty", bool(image.size) and float(image.std()) > 1.0,
               shape=list(image.shape), std=float(image.std()))
    else:
        record("topdown_non_empty", False, shape=None)

    report = {
        "kind": "blender_agent_p0_validation",
        "spec": args.spec,
        "build": os.path.abspath(build_dir),
        "scene": os.path.abspath(scene_dir),
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
    }
    print()
    for item in checks:
        print(f"[validate] {'PASS' if item['pass'] else 'FAIL'}  {item['name']}")
    print(f"[validate] overall: {'PASS' if report['pass'] else 'FAIL'}")
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"[validate] report written to {args.report}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
