#!/usr/bin/env python
"""Numeric validation of a blender_agent P0 build against the PanoWorld format.

Checks, in order:

1. every viewpoint file exists (``panoImage_2048.png``, ``place_depth.png``,
   ``place_depth_scale.txt``, ``extrinsics.txt``) plus ``map.json`` and
   ``topdown.png``
2. image / depth / extrinsics formats and value ranges
3. camera poses match the spec exactly
4. **depth numerics**: for every camera, all ERP rays are cast against
   ``build/furnished.ply`` (the manifest's ``visible_geometry``) and compared
   with ``place_depth.png / scale``
5. **door geometry**, per opening and independent of the cameras: local rays
   from *both* sides of every door measure the hole in the wall (is it really
   open, is the wall really there beside it, is the opening as wide and as high
   as the spec says) -- see :func:`check_door_openings`
6. **door visibility coverage** (informational, never fails a scene): for every
   camera the doors of its own room are projected into the ERP frame with the
   matrix-inverse projection of :func:`world_to_erp` and compared with the
   rendered depth, so a scene report says how many doorways each viewpoint
   actually sees through; reasonable occlusion is a number here, not a verdict
7. ``map.json`` is a symmetric graph over the supplied cameras

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
            bad = ~np.isfinite(t_exit)
            points = np.empty_like(origins[index])
            good = ~bad
            points[good] = origins[index][good] + directions[index][good] * t_exit[good, None]
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


def world_to_erp(points, c2w: np.ndarray, width: int, height: int):
    """World point(s) -> ERP pixel coordinates ``(u, v)`` and camera distance.

    The inverse of :func:`erp_rays`, and the only place in the repo where a
    world direction becomes a panorama column: a pixel centre ``(u, v)`` shoots
    the ray ``R @ (cos(lat) sin(lon), -sin(lat), cos(lat) cos(lon))``, so the
    point is transformed into the camera frame with ``R.T`` (the rotation is
    orthonormal) and ``lon``/``lat`` are read off that local vector.  Nothing
    here re-states the sign convention by hand, which is what the old
    ``atan2(dy, dx)`` column did -- mirrored about the camera's +X axis under
    the ``R0`` convention of ``spec.py`` (a target to the north-east was
    sampled to the south-east, a target to the north to the south).

    Returns ``(u, v, depth)``: fractional pixel coordinates (``x = u`` is the
    column, ``y = v`` the row, both counted from 0 with the pixel *centre* at
    ``round(u)``) and the straight-line distance from the camera.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    c2w = np.asarray(c2w, dtype=np.float64)
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]
    depth = np.linalg.norm(local, axis=1)
    lon = np.arctan2(local[:, 0], local[:, 2])
    lat = np.arctan2(-local[:, 1], np.hypot(local[:, 0], local[:, 2]))
    u = (lon + np.pi) / (2.0 * np.pi) * width - 0.5
    v = (np.pi / 2.0 - lat) / np.pi * height - 0.5
    return u, v, depth


def erp_pixel_of_point(point, c2w: np.ndarray, width: int, height: int):
    """``(column, row, distance)`` of one world point, with the seam wrapped."""
    u, v, depth = world_to_erp([point], c2w, width, height)
    column = int(round(float(u[0]))) % width
    row = int(round(float(v[0])))
    return column, row, float(depth[0])


def read_depth_meters(path: str, scale: float):
    array = np.array(Image.open(path))
    if array.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16 PNG, got {array.dtype}")
    return array.astype(np.float64) / scale, array > 0


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


def check_door_openings(spec: spec_mod.SceneSpec, caster, probe_m: float = 0.35,
                        step_m: float = 0.02, width_tol_m: float = 0.06,
                        height_tol_m: float = 0.08) -> dict:
    """Geometry acceptance for **every** opening, measured with local rays.

    A door is a hole through a wall, so the acceptance is local and needs no
    camera: stand ``probe_m`` in front of the wall on *each* side of the opening
    and cast horizontal rays perpendicular to it.

    * rays that cross the declared opening must leave the wall slab on the far
      side (``hit >= probe + thickness / 2``) -- the hole is really open, i.e.
      the ray travels *through* the wall instead of into it;
    * rays that cross the wall beside the opening must stop *inside* the slab
      (the wall is really there, so the opening is not a missing wall);
    * scanning the two axes measures the opening: the contiguous open run along
      the wall is its width, the open run from the floor up is its height, and
      both are compared with the spec.

    This replaces the old spot check (``spec.openings[0]`` against every camera,
    one column each, ``door_depth > 1.5 * wall_depth``): a door sealed in another
    wall was never looked at, and the ratio itself was meaningless for a real
    2-D plan (a camera looking along a corridor measures the far wall, not the
    wall above the door).
    """
    half = float(spec.wall_thickness) / 2.0
    t_enter, t_exit = probe_m - half, probe_m + half
    segments = spec.wall_segments()
    doors: list = []

    for index, opening in enumerate(spec.openings):
        report: dict = {
            "index": index,
            "between": list(opening.between),
            "center": [float(opening.center[0]), float(opening.center[1])],
            "spec_width_m": float(opening.width),
            "spec_height_m": float(opening.height),
        }
        segment = spec_mod.wall_segment_for_point(segments, opening.center, opening.between[0])
        if segment is None:
            report.update({"pass": False, "failures": ["opening centre is not on a wall shared by both rooms"]})
            doors.append(report)
            continue

        direction = np.array(segment.direction, dtype=np.float64)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        centre = np.array([float(opening.center[0]), float(opening.center[1])], dtype=np.float64)
        width, height = float(opening.width), float(opening.height)
        failures: list = []

        # which side of the wall each room is on (the ray probe has to stand
        # inside that room, so the sign comes from the room polygon itself)
        probe_sign = {}
        for room_id in opening.between:
            probe_sign[room_id] = 1.0 if spec_mod.point_in_polygon(
                centre + normal * probe_m, spec.room(room_id).polygon) else -1.0
        if probe_sign[opening.between[0]] == probe_sign[opening.between[1]]:
            probe_sign = {opening.between[0]: 1.0, opening.between[1]: -1.0}

        def offset_ray(along: float, z: float, side: float):
            origin = np.array([centre[0] + direction[0] * along + normal[0] * side * probe_m,
                               centre[1] + direction[1] * along + normal[1] * side * probe_m, z])
            return origin, np.array([-normal[0] * side, -normal[1] * side, 0.0])

        def cast(samples):
            origins = np.array([item[0] for item in samples], dtype=np.float64)
            directions = np.array([item[1] for item in samples], dtype=np.float64)
            return caster.cast(origins, directions, chunk=8192)

        def longest_open_run(mask) -> tuple:
            best, start = None, None
            for position, is_open in enumerate(mask):
                if is_open and start is None:
                    start = position
                if start is not None and (not is_open or position == len(mask) - 1):
                    end = position if is_open else position - 1
                    if best is None or end - start > best[1] - best[0]:
                        best = (start, end)
                    start = None
            return best

        z_scan = min(max(height / 2.0, 0.45), height - 0.15, float(spec.wall_height) - 0.15)
        along_scan = np.arange(-(width / 2.0 + 0.5), width / 2.0 + 0.5 + 1e-9, step_m)
        z_values = np.arange(0.10, float(spec.wall_height) - 0.05 + 1e-9, step_m)

        measured_widths, measured_heights = [], []
        for room_id in opening.between:
            side = probe_sign[room_id]
            through = [offset_ray(along, z, side)
                       for z in np.linspace(0.10, max(0.10, height - 0.10), 5)
                       for along in np.linspace(-(width / 2.0 - 0.05), width / 2.0 - 0.05, 7)]
            open_fraction = float(np.mean(cast(through) >= t_exit - 1e-4))

            beside = [offset_ray(along, z, side)
                      for along in (-(width / 2.0 + 0.15), width / 2.0 + 0.15)
                      for z in (0.6, z_scan)]
            beside_hits = cast(beside)
            solid_fraction = float(np.mean((beside_hits >= t_enter - 1e-4) & (beside_hits <= t_exit + 1e-3)))

            run = longest_open_run(cast([offset_ray(along, z_scan, side) for along in along_scan]) >= t_exit - 1e-4)
            if run is None:
                measured_width, centre_offset = 0.0, None
            else:
                measured_width = float((run[1] - run[0] + 1) * step_m)
                low, high = along_scan[run[0]] - step_m / 2.0, along_scan[run[1]] + step_m / 2.0
                centre_offset = float((low + high) / 2.0)
                if not low <= 0.0 <= high:
                    failures.append(f"{room_id}: the open run at z={z_scan:.2f} m does not contain the door centre")
            measured_widths.append(measured_width)

            open_z = cast([offset_ray(0.0, z, side) for z in z_values]) >= t_exit - 1e-4
            if not open_z[0]:
                failures.append(f"{room_id}: the opening does not reach the floor (first sample at "
                                f"z={z_values[0]:.2f} m is wall)")
            top = 0.0
            for value, is_open in zip(z_values, open_z):
                if not is_open:
                    break
                top = float(value)
            measured_height = (top + step_m / 2.0) if top > 0.0 else 0.0
            measured_heights.append(measured_height)

            report.setdefault("sides", {})[room_id] = {
                "probe_sign": side,
                "open_fraction": round(open_fraction, 4),
                "wall_solid_fraction": round(solid_fraction, 4),
                "open_width_m": round(measured_width, 3),
                "open_height_m": round(measured_height, 3),
                "width_centre_offset_m": None if centre_offset is None else round(centre_offset, 3),
            }
            if open_fraction < 0.98:
                failures.append(f"{room_id}: only {open_fraction * 100:.1f}% of the rays through the declared "
                                f"opening leave the wall slab (the opening is not open)")
            if solid_fraction < 0.95:
                failures.append(f"{room_id}: {100 * (1 - solid_fraction):.1f}% of the rays beside the opening do "
                                f"not stop inside the wall (the wall is missing next to the door)")
            if abs(measured_width - width) > width_tol_m:
                failures.append(f"{room_id}: measured opening width {measured_width:.2f} m vs spec {width:.2f} m")
            if abs(measured_height - height) > height_tol_m:
                failures.append(f"{room_id}: measured opening height {measured_height:.2f} m vs spec {height:.2f} m")

        report["measured_width_m"] = round(sum(measured_widths) / len(measured_widths), 3)
        report["measured_height_m"] = round(sum(measured_heights) / len(measured_heights), 3)
        report["pass"] = not failures
        report["failures"] = failures
        doors.append(report)

    return {
        "pass": all(door["pass"] for door in doors),
        "openings": len(doors),
        "probe_m": probe_m,
        "wall_half_thickness_m": half,
        "doors": doors,
    }


def door_visibility_check(spec: spec_mod.SceneSpec, camera: spec_mod.CameraSpec,
                          rendered: np.ndarray, c2w: np.ndarray) -> dict:
    """Informational: which of the camera's own doors does it see through?

    Every door of the camera's room is projected into the ERP frame by
    :func:`world_to_erp` (matrix inverse -- no hand-written column arithmetic)
    and the rendered depth at that pixel is compared with the distance to the
    door plane: a sight line that really passes through the opening arrives
    *behind* the door plane, an occluded one arrives in front of it.  The result
    is a coverage number, so reasonable occlusion (furniture between the camera
    and the doorway) is reported instead of failing the scene.
    """
    height, width = rendered.shape
    doors = [opening for opening in spec.openings if camera.room in opening.between]
    if not doors:
        return {"role": "informational", "pass": True, "skipped": "camera room has no opening"}
    entries = []
    for index, opening in enumerate(doors):
        z_sample = float(min(max(opening.height / 2.0, 0.30), opening.height - 0.10))
        point = np.array([float(opening.center[0]), float(opening.center[1]), z_sample])
        column, row, distance = erp_pixel_of_point(point, c2w, width, height)
        column = max(2, min(width - 3, column))
        row = max(2, min(height - 3, row))
        measured = float(np.median(rendered[row - 2:row + 3, column - 2:column + 3]))
        entries.append({
            "between": list(opening.between),
            "opening": index,
            "column": int(column),
            "row": int(row),
            "door_distance_m": round(distance, 3),
            "rendered_depth_m": round(measured, 3),
            "ratio": round(measured / distance, 3) if distance > 0 else None,
            "visible": bool(measured > 0.0 and measured >= distance - 0.10),
        })
    seen = sum(1 for entry in entries if entry["visible"])
    best = max(entries, key=lambda entry: entry["ratio"] or 0.0)
    return {
        "role": "informational",
        "pass": True,
        "doors": entries,
        "visible": seen,
        "of": len(entries),
        "visible_fraction": round(seen / len(entries), 3),
        "ratio": best["ratio"],
        "note": "occlusion is a coverage number here; the scene gate is door_openings_geometry",
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
    shell_ply_path = os.path.join(build_dir, "shell.ply")
    manifest_path = os.path.join(build_dir, "build_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            build_manifest = json.load(handle)
        candidate = (build_manifest.get("visible_geometry") or {}).get("ply")
        if candidate:
            ply_path = os.path.join(build_dir, candidate)
        shell_candidate = (build_manifest.get("shell") or {}).get("ply")
        if shell_candidate:
            shell_ply_path = os.path.join(build_dir, shell_candidate)
    if ply_path is None or not os.path.isfile(ply_path):
        ply_path = os.path.join(build_dir, "shell.ply")
    def load_caster(path: str):
        import trimesh

        loaded = trimesh.load(path)
        if hasattr(loaded, "geometry"):
            loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
        return loaded, MeshRayCaster(loaded, np.array([cam.pos for cam in cameras], dtype=np.float64))

    depth_reports = []
    door_visibility = {}
    if os.path.isfile(ply_path):
        mesh, caster = load_caster(ply_path)

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

        # informational: does this camera see through the doorways of its room?
        visibility = door_visibility_check(spec, camera, rendered, c2w)
        door_visibility[camera.name] = visibility

        passed = valid_fraction >= 0.995 and median < 0.005 and p95 < 0.02
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
                "door_spot_check": visibility,
                "pass": bool(passed),
            }
        )
        print(
            f"[validate] camera {camera.name}: valid={valid_fraction * 100:.3f}% "
            f"median|d|={median * 1000:.3f}mm p95|d|={p95 * 1000:.3f}mm max|d|={worst * 1000:.1f}mm "
            f"invalid_render={int(invalid_render.sum())} miss={int(invalid_truth.sum())} "
            f"doors_visible={visibility.get('visible', '?')}/{visibility.get('of', '?')} "
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

    # ---- 5: door geometry, every opening, both sides ---------------------- #
    # The doors are measured on the *shell* (walls and their cuts), not on the
    # furnished scene: the question here is the wall contract (open hole, wall
    # thickness, opening size), and furniture is ``checks.py``'s and the
    # visibility metric's business.
    door_ply_path = shell_ply_path if os.path.isfile(shell_ply_path) else ply_path
    if not spec.openings:
        record("door_openings_geometry", True, skipped="spec has no openings")
    elif door_ply_path is None or not os.path.isfile(door_ply_path):
        record("door_openings_geometry", False, error="no PLY with the shell geometry to measure the doors")
    else:
        if door_ply_path == ply_path and caster is not None:
            door_caster = caster
        else:
            _door_mesh, door_caster = load_caster(door_ply_path)
        doors = check_door_openings(spec, door_caster)
        for door in doors["doors"]:
            print(f"[validate] door {door['index']} {door['between']}: "
                  f"{'PASS' if door['pass'] else 'FAIL'} "
                  f"width={door.get('measured_width_m', float('nan')):.2f}m "
                  f"height={door.get('measured_height_m', float('nan')):.2f}m"
                  + (f" -- {door['failures'][0]}" if door.get("failures") else ""))
        record("door_openings_geometry", doors["pass"],
               ply=os.path.relpath(door_ply_path, REPO_ROOT), **doors)
        del door_caster

    # ---- 6: door visibility coverage (informational, never gates) --------- #
    seen = sum(entry.get("visible", 0) for entry in door_visibility.values())
    pairs = sum(entry.get("of", 0) for entry in door_visibility.values())
    record("door_visibility_coverage", True, role="informational",
           cameras=door_visibility, doors_visible=seen, doors_checked=pairs,
           visible_fraction=round(seen / pairs, 3) if pairs else None,
           note="per-camera door visibility; reasonable occlusion is a number here, not a failure")

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
