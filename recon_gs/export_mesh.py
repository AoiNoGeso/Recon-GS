"""Step 5: Export mesh from trained 3DGS via TSDF Fusion.

Pipeline
--------
1. Load trained Gaussians from .ply
2. Build surface-exclusion prompts from config flags (floor/ceiling/ground/sky)
3. For each training camera:
   a. Render RGB + depth (z in camera space)
   b. Run Grounded-SAM2 on the RGB frame to detect surface regions
   c. Zero-out masked pixels before TSDF integration
4. Integrate unmasked (depth, RGB) frames into a TSDF volume
5. Extract triangle mesh, clean, and save
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import pycolmap
from plyfile import PlyData
from gsplat import rasterization
import open3d as o3d

from recon_gs.align import load_or_compute_gravity_rotation, apply_to_c2w
from recon_gs.config import (
    MESH_VOXEL_SIZE,
    MESH_MAX_DEPTH,
    MESH_MIN_CLUSTER_TRIANGLES,
    MESH_MASK_FLOOR,
    MESH_MASK_CEILING,
    MESH_MASK_GROUND,
    MESH_MASK_SKY,
    MESH_FLOOR_PROMPTS,
    MESH_CEILING_PROMPTS,
    MESH_GROUND_PROMPTS,
    MESH_SKY_PROMPTS,
    MESH_GDINO_BOX_THRESHOLD,
    MESH_GDINO_TEXT_THRESHOLD,
    MESH_FILL_PLANES,
    MESH_PLANE_VOXEL_SIZE,
    MESH_PLANE_MIN_POINTS,
    MESH_PLANE_HEIGHT_PERCENTILE,
)

_GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_SAM2_MODEL_ID = "facebook/sam2-hiera-large"


# --------------------------------------------------------------------------- #
# PLY loading
# --------------------------------------------------------------------------- #

def _load_ply_gaussians(ply_path: Path, device: torch.device):
    """Load Gaussians from a .ply file exported by _export_ply().

    Returns (means, scales, quats, opacities, sh_coeffs) — all on `device`.
    scales are activation-applied (not log).
    """
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]

    means = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=1), dtype=torch.float32, device=device
    )

    scales = torch.tensor(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1),
        dtype=torch.float32,
        device=device,
    ).exp()

    quats = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
        dtype=torch.float32,
        device=device,
    )
    quats = quats / quats.norm(dim=-1, keepdim=True)

    opacities = torch.tensor(v["opacity"], dtype=torch.float32, device=device).sigmoid()

    sh_dc = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1),
        dtype=torch.float32,
        device=device,
    )
    sh_coeffs = sh_dc.unsqueeze(1)  # (N, 1, 3)

    return means, scales, quats, opacities, sh_coeffs


# --------------------------------------------------------------------------- #
# Grounded-SAM2 surface masking
# --------------------------------------------------------------------------- #

def _build_surface_prompts() -> list[str]:
    """Collect active surface prompts from config flags."""
    prompts: list[str] = []
    if MESH_MASK_FLOOR:
        prompts.extend(MESH_FLOOR_PROMPTS)
    if MESH_MASK_CEILING:
        prompts.extend(MESH_CEILING_PROMPTS)
    if MESH_MASK_GROUND:
        prompts.extend(MESH_GROUND_PROMPTS)
    if MESH_MASK_SKY:
        prompts.extend(MESH_SKY_PROMPTS)
    return prompts


def _load_gsam2_models(device: torch.device):
    """Load GroundingDINO and SAM2 models."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    gdino_processor = AutoProcessor.from_pretrained(_GDINO_MODEL_ID)
    gdino_model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(_GDINO_MODEL_ID)
        .to(device)
        .eval()
    )

    sam2_model = build_sam2_hf(_SAM2_MODEL_ID, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return gdino_processor, gdino_model, sam2_predictor


def _surface_mask_gsam2(
    image_np: np.ndarray,
    gdino_processor,
    gdino_model,
    sam2_predictor,
    prompt: str,
    device: torch.device,
) -> np.ndarray:
    """Return boolean mask (H, W); True = surface pixel to exclude from TSDF.

    Args:
        image_np : (H, W, 3) uint8 RGB image
        prompt   : period-separated GroundingDINO text prompt
    """
    from PIL import Image as PILImage

    image_pil = PILImage.fromarray(image_np)
    h, w = image_np.shape[:2]

    inputs = gdino_processor(
        images=image_pil,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=MESH_GDINO_BOX_THRESHOLD,
        text_threshold=MESH_GDINO_TEXT_THRESHOLD,
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"].cpu().numpy()  # (N, 4) xyxy

    mask = np.zeros((h, w), dtype=bool)
    if len(boxes) == 0:
        return mask

    sam2_predictor.set_image(image_np)
    sam_masks, _, _ = sam2_predictor.predict(
        box=boxes,
        multimask_output=False,
    )
    # sam_masks: (N, 1, H, W) or (N, H, W)
    if sam_masks.ndim == 4:
        sam_masks = sam_masks[:, 0]
    for m in sam_masks:
        mask |= m.astype(bool)

    return mask


# --------------------------------------------------------------------------- #
# Per-view rendering
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _render_view(
    means: Tensor,
    scales: Tensor,
    quats: Tensor,
    opacities: Tensor,
    sh_coeffs: Tensor,
    c2w: Tensor,
    K: Tensor,
    W: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB and depth for one camera view.

    Returns:
        rgb   : (H, W, 3) float32 [0, 1]
        depth : (H, W)    float32 [m], 0 = no hit
    """
    viewmat = torch.linalg.inv(c2w)[None]  # (1, 4, 4)

    # ---- RGB ----
    rgb_render, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=viewmat,
        Ks=K[None],
        width=W,
        height=H,
        sh_degree=0,
        packed=False,
    )
    rgb = rgb_render[0].clamp(0, 1).cpu().numpy()  # (H, W, 3)

    # ---- Depth (z in camera space) ----
    w2c = torch.linalg.inv(c2w)
    means_cam = (w2c[:3, :3] @ means.T + w2c[:3, 3:]).T  # (N, 3)
    z_vals = means_cam[:, 2].clamp(min=0.0)               # (N,)

    depth_render, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=z_vals.view(-1, 1),
        viewmats=viewmat,
        Ks=K[None],
        width=W,
        height=H,
        sh_degree=None,
        packed=False,
    )
    depth = depth_render[0, ..., 0].cpu().numpy()  # (H, W)

    return rgb, depth


# --------------------------------------------------------------------------- #
# Convex-hull plane filling
# --------------------------------------------------------------------------- #

def _fill_planes_from_mesh(
    mesh_clean: "o3d.geometry.TriangleMesh",
    world_up: np.ndarray,
) -> list["o3d.geometry.TriangleMesh"]:
    """Derive floor/ceiling plane meshes from the CLEANED TSDF mesh vertices.

    Uses the bottom/top MESH_PLANE_HEIGHT_PERCENTILE % of mesh vertices as
    floor/ceiling candidates.  The cleaned mesh is noise-free, so plane fits
    are more accurate than using raw per-frame depth points.
    """
    verts  = np.asarray(mesh_clean.vertices).copy()   # (V, 3)
    colors = (
        np.asarray(mesh_clean.vertex_colors).copy()
        if mesh_clean.has_vertex_colors()
        else np.ones((len(verts), 3), dtype=np.float64)
    )

    up = world_up.astype(np.float32)
    up /= np.linalg.norm(up)
    heights = verts @ up

    pct = MESH_PLANE_HEIGHT_PERCENTILE
    low_cut  = float(np.percentile(heights, pct))
    high_cut = float(np.percentile(heights, 100.0 - pct))
    print(f"  Mesh height range [{heights.min():.2f}, {heights.max():.2f}]  "
          f"floor≤{low_cut:.2f}  ceiling≥{high_cut:.2f}")

    meshes: list[o3d.geometry.TriangleMesh] = []
    for label, mask in [("floor", heights <= low_cut), ("ceiling", heights >= high_cut)]:
        m = _convex_hull_plane_mesh(
            verts[mask].astype(np.float32),
            colors[mask].astype(np.float32),
            world_up,
            label,
        )
        if m is not None:
            meshes.append(m)
    return meshes


def _pca_plane(points: np.ndarray, world_up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a plane to points via PCA (least-squares).

    Returns (centroid, normal, u_axis, v_axis) all as float64 unit vectors.
    normal is flipped to face the same half-space as world_up.
    """
    pts = points.astype(np.float64)
    centroid = pts.mean(axis=0)
    cov = (pts - centroid).T @ (pts - centroid) / len(pts)
    _, eigvecs = np.linalg.eigh(cov)   # eigenvalues ascending → eigvecs[:, 0] = smallest
    normal = eigvecs[:, 0]             # plane normal = min-variance direction

    # Ensure normal points toward world_up (not away)
    up = world_up.astype(np.float64)
    if normal @ up < 0:
        normal = -normal

    # Build orthonormal tangent basis in the plane
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(normal, ref);  u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    return centroid, normal, u_axis, v_axis


def _convex_hull_plane_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    world_up: np.ndarray,
    label: str,
) -> o3d.geometry.TriangleMesh | None:
    """Build a flat convex-hull mesh for a surface group (floor or ceiling).

    Steps:
      1. Voxel-downsample the point group.
      2. Fit a best-fit plane via PCA (handles tilted floors/ceilings).
      3. Project all points onto that plane.
      4. Compute 2D convex hull; triangulate as a fan from the centroid.

    Returns a TriangleMesh, or None if the group has too few points.
    """
    from scipy.spatial import ConvexHull

    if len(points) < MESH_PLANE_MIN_POINTS:
        print(f"  [{label}] too few points ({len(points):,}), skipped")
        return None

    # Voxel downsample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd_down = pcd.voxel_down_sample(MESH_PLANE_VOXEL_SIZE)
    pts = np.array(pcd_down.points, dtype=np.float32)
    cols = np.array(pcd_down.colors, dtype=np.float32)
    del pcd, pcd_down
    print(f"  [{label}] {len(pts):,} points after downsampling (was {len(points):,})")

    if len(pts) < MESH_PLANE_MIN_POINTS:
        print(f"  [{label}] too few points after downsampling, skipped")
        return None

    # PCA plane fit: handles tilted surfaces
    centroid, normal, u_axis, v_axis = _pca_plane(pts, world_up)
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(normal @ world_up.astype(np.float64)), 0, 1))))
    print(f"  [{label}] PCA plane normal=({normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f})  "
          f"tilt={tilt_deg:.1f}°")

    # Project points onto the fitted plane (along normal direction)
    pts_f64 = pts.astype(np.float64)
    dists = (pts_f64 - centroid) @ normal           # signed distance to plane
    projected = pts_f64 - np.outer(dists, normal)   # (N, 3) on the plane

    # 2D coordinates in the plane's local frame
    pts_2d = np.stack([projected @ u_axis, projected @ v_axis], axis=1)

    try:
        hull = ConvexHull(pts_2d)
    except Exception as e:
        print(f"  [{label}] ConvexHull failed: {e}")
        return None

    hull_2d = pts_2d[hull.vertices]   # (K, 2)
    n_hull = len(hull_2d)
    print(f"  [{label}] convex hull: {n_hull} vertices")

    # Reconstruct 3D hull vertices on the fitted plane
    # p = centroid + s*u_axis + t*v_axis
    # (component along normal is zero since we project back onto the plane through centroid)
    hull_3d = (
        centroid
        + hull_2d[:, 0:1] * u_axis
        + hull_2d[:, 1:2] * v_axis
    ).astype(np.float32)

    # Fan triangulation: center + hull ring
    center = hull_3d.mean(axis=0, keepdims=True)          # (1, 3)
    vertices = np.vstack([center, hull_3d])               # (K+1, 3)
    triangles = [[0, j, j + 1] for j in range(1, n_hull)]
    triangles.append([0, n_hull, 1])

    avg_color = cols.mean(axis=0).astype(np.float64)
    color_arr = np.tile(avg_color, (len(vertices), 1))

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(color_arr)
    mesh.compute_vertex_normals()
    return mesh


# --------------------------------------------------------------------------- #
# Mesh post-processing
# --------------------------------------------------------------------------- #

def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Remove small disconnected clusters and rebuild as an independent mesh.

    Avoids copy.deepcopy which shares Open3D internal buffers and causes
    double-free crashes on destruction.
    """
    cc_result = mesh.cluster_connected_triangles()
    tri_clusters  = np.array(cc_result[0])
    cluster_n_tris = np.array(cc_result[1])
    del cc_result

    keep_mask = cluster_n_tris[tri_clusters] >= MESH_MIN_CLUSTER_TRIANGLES

    all_tris   = np.asarray(mesh.triangles)[keep_mask].copy()
    all_verts  = np.asarray(mesh.vertices).copy()
    has_colors = mesh.has_vertex_colors()
    all_colors = np.asarray(mesh.vertex_colors).copy() if has_colors else None

    used   = np.unique(all_tris)
    remap  = np.full(len(all_verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_tris  = remap[all_tris]
    new_verts = all_verts[used]

    is_degen = (
        (new_tris[:, 0] == new_tris[:, 1])
        | (new_tris[:, 1] == new_tris[:, 2])
        | (new_tris[:, 0] == new_tris[:, 2])
    )
    new_tris = new_tris[~is_degen]

    cleaned = o3d.geometry.TriangleMesh()
    cleaned.vertices  = o3d.utility.Vector3dVector(new_verts)
    cleaned.triangles = o3d.utility.Vector3iVector(new_tris)
    if has_colors and all_colors is not None:
        cleaned.vertex_colors = o3d.utility.Vector3dVector(all_colors[used])
    cleaned.compute_vertex_normals()
    return cleaned


def _merge_meshes(
    base: o3d.geometry.TriangleMesh,
    extras: list[o3d.geometry.TriangleMesh],
) -> o3d.geometry.TriangleMesh:
    """Concatenate meshes via numpy to avoid Open3D += double-free bugs."""
    verts_list  = [np.asarray(base.vertices).copy()]
    tris_list   = [np.asarray(base.triangles).copy()]
    colors_list = [
        np.asarray(base.vertex_colors).copy()
        if base.has_vertex_colors()
        else np.ones((len(verts_list[0]), 3), dtype=np.float64)
    ]

    offset = len(verts_list[0])
    for pm in extras:
        pm_verts  = np.asarray(pm.vertices).copy()
        pm_tris   = np.asarray(pm.triangles).copy() + offset
        pm_colors = (
            np.asarray(pm.vertex_colors).copy()
            if pm.has_vertex_colors()
            else np.ones((len(pm_verts), 3), dtype=np.float64)
        )
        verts_list.append(pm_verts)
        tris_list.append(pm_tris)
        colors_list.append(pm_colors)
        offset += len(pm_verts)

    merged = o3d.geometry.TriangleMesh()
    merged.vertices      = o3d.utility.Vector3dVector(np.vstack(verts_list))
    merged.triangles     = o3d.utility.Vector3iVector(np.vstack(tris_list))
    merged.vertex_colors = o3d.utility.Vector3dVector(np.vstack(colors_list))
    merged.compute_vertex_normals()
    return merged


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def export_mesh(
    colmap_dir: Path,
    frames_dir: Path,
    ply_path: Path,
    output_dir: Path,
) -> None:
    """Export mesh from trained Gaussians via TSDF Fusion.

    Args:
        colmap_dir : directory containing sparse/0 COLMAP reconstruction
        frames_dir : directory with input RGB frames
        ply_path   : trained Gaussian .ply file
        output_dir : where to write tsdf_fusion.ply and tsdf_fusion_post.ply
    """
    from PIL import Image as PILImage

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_dir = colmap_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load Gaussians ----
    print("Loading Gaussians …")
    means, scales, quats, opacities, sh_coeffs = _load_ply_gaussians(ply_path, device)
    print(f"  {len(means):,} Gaussians loaded")

    # ---- Gravity alignment ----
    R_align = torch.tensor(
        load_or_compute_gravity_rotation(sparse_dir), dtype=torch.float32, device=device
    )

    # ---- Grounded-SAM2 surface masking ----
    surface_prompts = _build_surface_prompts()
    use_surface_mask = len(surface_prompts) > 0
    gdino_processor = gdino_model = sam2_predictor = None

    if use_surface_mask:
        prompt_str = " . ".join(surface_prompts) + " ."
        active = (
            f"floor={MESH_MASK_FLOOR}  ceiling={MESH_MASK_CEILING}  "
            f"ground={MESH_MASK_GROUND}  sky={MESH_MASK_SKY}"
        )
        print(f"Loading Grounded-SAM2 for surface masking ({active}) …")
        gdino_processor, gdino_model, sam2_predictor = _load_gsam2_models(device)
    else:
        print("  Surface masking disabled (all flags are False)")

    # ---- Load COLMAP cameras ----
    model = pycolmap.Reconstruction(str(sparse_dir))

    # ---- TSDF Volume ----
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=MESH_VOXEL_SIZE,
        sdf_trunc=4.0 * MESH_VOXEL_SIZE,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    n_images = len(model.images)
    print(f"Rendering & integrating {n_images} views …")

    for i, image_id in enumerate(sorted(model.images)):
        img = model.images[image_id]
        cam = model.cameras[img.camera_id]

        fx = cam.focal_length_x if hasattr(cam, "focal_length_x") else cam.focal_length
        fy = cam.focal_length_y if hasattr(cam, "focal_length_y") else fx
        cx, cy = cam.principal_point_x, cam.principal_point_y

        img_path = frames_dir / img.name
        pil_img = PILImage.open(img_path).convert("RGB")
        W, H = pil_img.size

        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device
        )

        # Camera pose (c2w with gravity alignment)
        cfw = img.cam_from_world()
        R_np = np.array(cfw.rotation.matrix(), dtype=np.float32)
        t_np = np.array(cfw.translation, dtype=np.float32)
        w2c_np = np.eye(4, dtype=np.float32)
        w2c_np[:3, :3] = R_np
        w2c_np[:3, 3] = t_np
        c2w = apply_to_c2w(
            torch.tensor(np.linalg.inv(w2c_np), dtype=torch.float32, device=device),
            R_align,
        )

        # Render RGB + depth from Gaussians
        rgb, depth = _render_view(means, scales, quats, opacities, sh_coeffs, c2w, K, W, H)

        # Surface mask via Grounded-SAM2
        depth_masked = depth.copy()
        if use_surface_mask:
            image_np = np.array(pil_img)
            surface_mask = _surface_mask_gsam2(
                image_np, gdino_processor, gdino_model, sam2_predictor, prompt_str, device
            )
            depth_masked[surface_mask] = 0.0

        depth_masked[depth_masked > MESH_MAX_DEPTH] = 0.0

        # Integrate into TSDF
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        depth_uint16 = (depth_masked * 1000).clip(0, 65535).astype(np.uint16)

        o3d_color = o3d.geometry.Image(rgb_uint8)
        o3d_depth = o3d.geometry.Image(depth_uint16)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1000.0,
            depth_trunc=MESH_MAX_DEPTH,
            convert_rgb_to_intensity=False,
        )

        pose_o3d = np.eye(4, dtype=np.float64)
        pose_o3d[:3, :3] = R_np.astype(np.float64)
        pose_o3d[:3, 3] = t_np.astype(np.float64)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        volume.integrate(rgbd, intrinsic, pose_o3d)

        if (i + 1) % 50 == 0 or (i + 1) == n_images:
            print(f"  [{i + 1}/{n_images}]")

    # ---- Extract & save ----
    print("Extracting triangle mesh …")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    raw_path = output_dir / "tsdf_fusion.ply"
    o3d.io.write_triangle_mesh(
        str(raw_path), mesh,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"  Raw mesh → {raw_path}  ({len(mesh.triangles):,} triangles)")

    print("Cleaning mesh …")
    mesh_clean = _clean_mesh(mesh)
    del mesh  # release raw mesh to avoid shared-buffer double-free

    # ---- Convex-hull plane filling (from cleaned mesh vertices) ----
    if MESH_FILL_PLANES:
        R_align_np = load_or_compute_gravity_rotation(sparse_dir)
        world_up   = R_align_np.T @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
        world_up  /= np.linalg.norm(world_up)

        print("Building convex-hull planes from cleaned mesh vertices …")
        plane_meshes = _fill_planes_from_mesh(mesh_clean, world_up)
        print(f"  {len(plane_meshes)} plane(s) built")
        if plane_meshes:
            mesh_clean = _merge_meshes(mesh_clean, plane_meshes)
            del plane_meshes

    post_path = output_dir / "tsdf_fusion_post.ply"
    o3d.io.write_triangle_mesh(
        str(post_path), mesh_clean,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    n_tris = len(mesh_clean.triangles)
    del mesh_clean
    print(f"  Final mesh → {post_path}  ({n_tris:,} triangles)")
