"""Equirectangular panorama rendering and per-pixel Euclidean depth.

Camera orientation (the part that must match PanoWorld byte for byte)
--------------------------------------------------------------------
PanoWorld's ``compute_plucmap_pano`` uses an OpenCV-style camera frame
(``+X`` right, ``+Y`` down, ``+Z`` forward) with

    ray_local = (cos(lat) sin(lon), -sin(lat), cos(lat) cos(lon))

so the camera-to-world matrix stored in ``extrinsics.txt`` has camX = image
right, camY = image down, camZ = image forward.  At yaw 0 that matrix is
``spec.R0`` (forward = world +X, image up = world +Z).

Blender's camera frame is ``+X`` right, ``+Y`` up, ``-Z`` forward, so the
rotation to hand to Blender is ``R_pano @ diag(1, -1, -1)``: image right stays
camX, image up becomes ``-camY`` and the view direction becomes ``-camZ``.
``blender_agent/render_selftest.py`` re-verifies this mapping against the
formula by rendering emissive markers along known world axes.

Depth comes from the Cycles *Position* pass (world-space hit point of the
camera ray), composited into a 32-bit OpenEXR, read back inside the same
Blender session and converted with ``||position - camera_position||``.
"""

from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import bpy
import numpy as np
from mathutils import Matrix

from blender_agent import spec as spec_mod

POSITION_EXR_COLOR_DEPTH = "32"


def blender_camera_rotation(yaw_rad: float = 0.0) -> Matrix:
    """Blender camera-to-world rotation for a PanoWorld ERP camera."""
    r_pano = np.array(spec_mod.yaw_rotation(yaw_rad), dtype=np.float64)
    flip = np.diag([1.0, -1.0, -1.0])
    rot = r_pano @ flip
    matrix = Matrix(
        (
            (rot[0][0], rot[0][1], rot[0][2], 0.0),
            (rot[1][0], rot[1][1], rot[1][2], 0.0),
            (rot[2][0], rot[2][1], rot[2][2], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return matrix


def create_pano_camera(
    camera: spec_mod.CameraSpec,
    yaw_rad: float = 0.0,
    clip_end: float = 10000.0,
) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(f"pano_{camera.name}")
    cam_data.type = "PANO"
    cam_data.panorama_type = "EQUIRECTANGULAR"
    cam_data.clip_start = 0.01
    cam_data.clip_end = clip_end
    cam_obj = bpy.data.objects.new(f"pano_{camera.name}", cam_data)
    matrix = blender_camera_rotation(yaw_rad)
    matrix.translation = (float(camera.pos[0]), float(camera.pos[1]), float(camera.pos[2]))
    cam_obj.matrix_world = matrix
    bpy.context.scene.collection.objects.link(cam_obj)
    return cam_obj


def configure_render(
    scene: bpy.types.Scene,
    width: int = spec_mod.PANO_WIDTH,
    height: int = spec_mod.PANO_HEIGHT,
    samples: int = 48,
    use_denoising: bool = True,
    view_transform: str = "Standard",
    exposure: float = 0.0,
    threads: int = 16,
    denoising_quality: str = "BALANCED",
    denoising_input_passes: str = "RGB",
) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = 0
    scene.cycles.use_denoising = use_denoising
    if use_denoising:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        # measured peak RSS for a 2048x1024 pano on this container:
        # HIGH + albedo/normal passes 2306 MiB, BALANCED + RGB 1993 MiB,
        # denoising off 1880 MiB - the pod limit leaves ~2.3 GiB of headroom
        scene.cycles.denoising_quality = denoising_quality
        scene.cycles.denoising_input_passes = denoising_input_passes
    # Box filter exactly one pixel wide: the Position pass then averages the hit
    # point over the pixel footprint instead of bleeding across silhouettes.
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.render.filter_size = 1.0

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = view_transform
    scene.view_settings.look = "None"
    scene.view_settings.exposure = float(exposure)
    scene.render.film_transparent = False
    # The container this runs in caps memory well below the machine's core
    # count; Cycles and OIDN spawn per-thread buffers, so keep the pool small.
    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads


def setup_position_pass(
    scene: bpy.types.Scene,
    output_dir: str,
    slot_name: str,
    raw_slot_name: Optional[str] = None,
) -> bpy.types.CompositorNodeOutputFile:
    """Compositor: Render Layers -> Composite (beauty) + File Output (Position EXR).

    ``raw_slot_name`` adds a second slot holding the *linear* combined pass as a
    32-bit EXR, which ``denoise_pano_file`` can denoise later in a separate,
    low-memory Blender process.
    """
    view_layer = bpy.context.view_layer
    view_layer.use_pass_position = True
    view_layer.cycles.denoising_store_passes = False

    scene.use_nodes = True
    node_tree = scene.node_tree
    node_tree.nodes.clear()
    render_layers = node_tree.nodes.new("CompositorNodeRLayers")
    composite = node_tree.nodes.new("CompositorNodeComposite")
    node_tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    file_output = node_tree.nodes.new("CompositorNodeOutputFile")
    file_output.base_path = output_dir
    file_output.format.file_format = "OPEN_EXR"
    file_output.format.color_depth = POSITION_EXR_COLOR_DEPTH
    file_output.format.color_mode = "RGB"
    if hasattr(file_output.format, "color_management"):
        file_output.format.color_management = "OVERRIDE"
        file_output.format.view_settings.view_transform = "Raw"
        file_output.format.view_settings.look = "None"
    slot = file_output.file_slots[0]
    slot.path = slot_name
    node_tree.links.new(render_layers.outputs["Position"], file_output.inputs[0])

    if raw_slot_name is not None:
        # file_slots.new(name) already names the slot; the slot object itself is
        # exposed through the node's inputs (NodeOutputFileSlotFile has no .name)
        file_output.file_slots.new(raw_slot_name)
        file_output.file_slots[1].path = raw_slot_name
        node_tree.links.new(render_layers.outputs["Image"], file_output.inputs[1])
    return file_output


def denoise_pano_file(raw_exr: str, out_png: str, prefilter: str = "ACCURATE", quality: str = "FAST") -> dict:
    """OpenImageDenoise the linear EXR written by :func:`setup_position_pass`.

    Runs in its own Blender process on purpose: OIDN inside the 2048x1024 render
    process pushes the peak past this container's memory cap (measured 2306 MiB
    with HIGH, 1993 MiB with BALANCED+RGB against a ~1.9 GiB budget), while
    denoising one float image with OIDN/FAST costs ~1.5 GiB on its own.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    image = bpy.data.images.load(raw_exr)
    width, height = int(image.size[0]), int(image.size[1])
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.filepath = os.path.join(os.path.dirname(os.path.abspath(out_png)), "denoise_tmp_")

    scene.use_nodes = True
    node_tree = scene.node_tree
    node_tree.nodes.clear()
    source = node_tree.nodes.new("CompositorNodeImage")
    source.image = image
    denoise = node_tree.nodes.new("CompositorNodeDenoise")
    denoise.prefilter = prefilter
    denoise.quality = quality
    denoise.use_hdr = True
    output = node_tree.nodes.new("CompositorNodeOutputFile")
    output.base_path = os.path.dirname(os.path.abspath(out_png))
    output.format.file_format = "PNG"
    output.format.color_mode = "RGB"
    output.format.color_depth = "8"
    if hasattr(output.format, "color_management"):
        output.format.color_management = "OVERRIDE"
        output.format.view_settings.view_transform = "Standard"
        output.format.view_settings.look = "None"
    stem = os.path.splitext(os.path.basename(out_png))[0]
    slot = output.file_slots[0]
    slot.path = f"{stem}_tmp_"
    node_tree.links.new(source.outputs["Image"], denoise.inputs["Image"])
    node_tree.links.new(denoise.outputs["Image"], output.inputs[0])

    # the compositor runs as part of a render; the scene holds no geometry, so
    # this only evaluates the Image -> Denoise -> File Output chain
    bpy.ops.render.render(write_still=False)

    written = sorted(glob.glob(os.path.join(output.base_path, f"{stem}_tmp_*.png")))
    if not written:
        raise RuntimeError(f"denoise pass produced no output for {raw_exr}")
    os.replace(written[-1], out_png)
    for stale in written[:-1]:
        os.remove(stale)
    bpy.data.images.remove(image)
    return {"raw_exr": os.path.basename(raw_exr), "pano": os.path.basename(out_png),
            "prefilter": prefilter, "quality": quality, "use_hdr": True, "size": [width, height]}


def read_image_pixels(path: str) -> np.ndarray:
    """Load an image's float pixels as ``(H, W, 4)``, row 0 = image top."""
    image = bpy.data.images.load(path)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        if width == 0 or height == 0:
            raise RuntimeError(f"image '{path}' has zero size")
        buffer = np.empty(width * height * image.channels, dtype=np.float32)
        image.pixels.foreach_get(buffer)
        array = buffer.reshape(height, width, image.channels)
        array = array[::-1]  # Blender stores bottom-up
        if image.channels < 4:
            padded = np.zeros((height, width, 4), dtype=np.float32)
            padded[..., : image.channels] = array
            array = padded
    finally:
        bpy.data.images.remove(image)
    return np.ascontiguousarray(array)


def _find_position_exr(output_dir: str, slot_name: str) -> str:
    matches = sorted(glob.glob(os.path.join(output_dir, f"{slot_name}*.exr")))
    if not matches:
        raise FileNotFoundError(f"Position pass EXR for '{slot_name}' not found in {output_dir}")
    return matches[-1]


def render_camera(
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    camera: spec_mod.CameraSpec,
    output_dir: str,
    samples: int = 48,
    save_raw_exr: bool = False,
) -> dict:
    """Render one pano: RGB PNG plus a float32 ``(H, W)`` Euclidean depth npy."""
    os.makedirs(output_dir, exist_ok=True)
    slot_name = f"position_{camera.name}"
    raw_slot_name = f"raw_{camera.name}" if save_raw_exr else None
    setup_position_pass(scene, output_dir, slot_name, raw_slot_name=raw_slot_name)

    pano_path = os.path.join(output_dir, f"pano_{camera.name}.png")
    scene.camera = camera_obj
    scene.render.filepath = pano_path
    bpy.ops.render.render(write_still=True)

    exr_path = _find_position_exr(output_dir, slot_name)
    positions = read_image_pixels(exr_path)
    origin = np.array(camera.pos, dtype=np.float64)
    delta = positions[..., :3].astype(np.float64) - origin[None, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    invalid = ~np.isfinite(distance) | (distance < 1e-4) | (distance > spec_mod.DEPTH_MAX_METERS * 0.999)
    distance = np.where(invalid, 0.0, distance).astype(np.float32)

    depth_path = os.path.join(output_dir, f"depth_{camera.name}.npy")
    np.save(depth_path, distance)

    raw_exr = None
    if raw_slot_name is not None:
        raw_exr = _find_position_exr(output_dir, raw_slot_name)

    valid = distance > 0.0
    stats = {
        "camera": camera.name,
        "pos": [float(v) for v in camera.pos],
        "yaw": 0.0,
        "pano": os.path.basename(pano_path),
        "depth_npy": os.path.basename(depth_path),
        "depth_exr": os.path.basename(exr_path),
        "raw_exr": os.path.basename(raw_exr) if raw_exr else None,
        "width": int(distance.shape[1]),
        "height": int(distance.shape[0]),
        "samples": int(samples),
        "invalid_pixels": int((~valid).sum()),
        "valid_fraction": float(valid.mean()),
        "depth_min": float(distance[valid].min()) if valid.any() else 0.0,
        "depth_max": float(distance[valid].max()) if valid.any() else 0.0,
        "depth_mean": float(distance[valid].mean()) if valid.any() else 0.0,
    }
    return stats


def _evaluated_triangles(obj, depsgraph):
    """(verts (n,3) float32 in *local* space, tris (m,3) int32) of an evaluated mesh.

    Coordinates stay local on purpose: ``depsgraph.object_instances`` already
    carries the final world matrix for each instance (including collection
    instancing), so applying it again would displace every duplicated asset.
    """
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if not mesh.polygons:
            return None
        mesh.calc_loop_triangles()
        vertex_count = len(mesh.vertices)
        coords = np.empty(vertex_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(vertex_count, 3)
        tri_count = len(mesh.loop_triangles)
        indices = np.empty(tri_count * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", indices)
        return coords, indices.reshape(tri_count, 3)
    finally:
        evaluated.to_mesh_clear()


def write_binary_ply(path: str, chunks) -> dict:
    """Write triangle meshes as one binary_little_endian PLY (in world metres)."""
    vertex_total = sum(len(verts) for verts, _ in chunks)
    face_total = sum(len(faces) for _, faces in chunks)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated by blender_agent.panorama.export_visible_ply\n"
        f"element vertex {vertex_total}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {face_total}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    face_dtype = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        # PLY wants every vertex first, then every face: two passes, not interleaved
        for verts, _ in chunks:
            handle.write(np.ascontiguousarray(verts, dtype="<f4").tobytes())
        base = 0
        for verts, faces in chunks:
            block = np.empty(len(faces), dtype=face_dtype)
            block["n"] = 3
            shifted = faces.astype("<i4") + base
            block["a"], block["b"], block["c"] = shifted[:, 0], shifted[:, 1], shifted[:, 2]
            handle.write(block.tobytes())
            base += len(verts)
    return {"vertices": int(vertex_total), "triangles": int(face_total),
            "bytes": os.path.getsize(path)}


def export_visible_ply(path: str, objects=None, skip_names=(), include_instances: bool = True) -> dict:
    """Export every camera-visible triangle to a PLY, in Blender world coordinates.

    ``objects`` restricts the export to those objects' instances; ``None`` means
    "everything the renderer can see", including collection instances such as
    the furniture and the sky dome.  Modifiers (e.g. the shell bevel) are
    evaluated, so the PLY matches the rendered geometry.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    wanted = None if objects is None else {obj.name for obj in objects}
    skip = set(skip_names)
    chunks = []
    seen = set()
    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.type != "MESH" or obj.name in skip:
            continue
        if wanted is not None and obj.name not in wanted:
            continue
        key = (obj.name, tuple(round(v, 4) for v in instance.matrix_world.to_translation()))
        if key in seen:
            continue
        seen.add(key)
        geometry = _evaluated_triangles(obj, depsgraph)
        if geometry is None:
            continue
        verts, indices = geometry
        matrix = np.array(instance.matrix_world, dtype=np.float64)
        verts = (verts @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
        chunks.append((verts, indices))
    if not chunks:
        raise RuntimeError("no visible mesh geometry to export")
    stats = write_binary_ply(path, chunks)
    stats["instances"] = len(chunks)
    return stats


def export_shell_ply(path: str, objects: Sequence[bpy.types.Object]) -> dict:
    """Write the render-used geometry to a PLY in Blender world coordinates."""
    return export_visible_ply(path, objects=objects, include_instances=True)
