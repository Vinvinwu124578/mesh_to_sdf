import argparse
from pathlib import Path

import numpy as np

from common import DEFAULT_SHAPENET_ROOT

try:
    import open3d as o3d
except Exception:
    o3d = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize adaptive-confidence tactile NPZ files. "
            "Supports patch confidence and band-width coloring."
        )
    )
    parser.add_argument("--npz-path", type=str, default=None)
    parser.add_argument("--view-index", type=int, default=None)
    parser.add_argument("--hide-surface", action="store_true")
    parser.add_argument("--show-query", action="store_true")
    parser.add_argument("--hide-query", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hide-patches", action="store_true")
    parser.add_argument("--hide-centers", action="store_true")
    parser.add_argument("--show-surface-normals", action="store_true", default=False)
    parser.add_argument("--hide-center-normals", action="store_true")
    parser.add_argument(
        "--color-mode",
        type=str,
        default="single",
        choices=["confidence", "band", "finger", "single"],
    )
    parser.add_argument("--max-surface-points", type=int, default=15000)
    parser.add_argument("--max-query-points", type=int, default=10000)
    parser.add_argument("--max-patch-points", type=int, default=50000)
    parser.add_argument("--normal-subsample", type=int, default=500)
    parser.add_argument("--normal-scale", type=float, default=0.03)
    parser.add_argument("--center-radius", type=float, default=0.02)
    parser.add_argument("--point-size", type=float, default=4.0)
    return parser.parse_args()


def find_default_npz():
    default_dir = DEFAULT_SHAPENET_ROOT / "tactistruct_progressive_attn_fix_adaptive_confidence_no_pose_onefolder"
    files = sorted(default_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {default_dir}")
    return str(files[0])


def subsample_array(arr, max_n):
    if arr is None:
        return None, None
    if len(arr) <= max_n:
        return arr, None
    idx = np.random.choice(len(arr), max_n, replace=False)
    return arr[idx], idx


def make_point_cloud(points, colors):
    if o3d is None:
        raise RuntimeError("open3d is not available in the current environment.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def make_normals_lineset(points, normals, scale=0.03, color=(1.0, 1.0, 0.0)):
    if o3d is None:
        raise RuntimeError("open3d is not available in the current environment.")
    points = points.astype(np.float64)
    normals = normals.astype(np.float64)
    normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    end_points = points + normals * scale
    line_points = np.vstack([points, end_points])
    n = len(points)
    lines = [[i, i + n] for i in range(n)]
    colors = [list(color) for _ in range(n)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def make_spheres_at_points(points, radius=0.02, colors=None, default_color=(1.0, 1.0, 1.0)):
    if o3d is None:
        raise RuntimeError("open3d is not available in the current environment.")
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return None
    meshes = []
    if colors is None:
        colors = np.tile(np.asarray(default_color, dtype=np.float32)[None, :], (len(points), 1))
    for point, color in zip(points, colors):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point.astype(np.float64))
        sphere.paint_uniform_color(np.asarray(color, dtype=np.float64))
        meshes.append(sphere)
    merged = meshes[0]
    for mesh in meshes[1:]:
        merged += mesh
    return merged


def color_query_sdf(points, sdf, eps=1e-3):
    colors = np.zeros((len(points), 3), dtype=np.float32)
    inside = sdf < -eps
    outside = sdf > eps
    near = np.abs(sdf) <= eps
    colors[inside] = [0.0, 0.0, 1.0]
    colors[outside] = [1.0, 0.0, 0.0]
    colors[near] = [1.0, 1.0, 1.0]
    return colors


def finger_palette():
    return np.asarray(
        [
            [0.00, 0.90, 0.30],
            [1.00, 0.85, 0.10],
            [1.00, 0.50, 0.05],
            [0.95, 0.15, 0.85],
            [0.10, 0.90, 0.95],
            [0.55, 0.95, 0.55],
            [0.55, 0.55, 1.00],
            [1.00, 0.30, 0.30],
            [0.70, 0.40, 1.00],
            [0.30, 0.75, 1.00],
        ],
        dtype=np.float32,
    )


def scalar_to_rgb(values, mode):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    values = np.clip(values, 0.0, 1.0) if mode == "confidence" else values
    vmin = float(values.min())
    vmax = float(values.max())
    span = max(vmax - vmin, 1e-8)
    t = (values - vmin) / span
    if mode == "confidence":
        # Green-scale map: low confidence is darker/desaturated green,
        # high confidence is brighter saturated green.
        r = 0.06 + 0.20 * t
        g = 0.35 + 0.60 * t
        b = 0.06 + 0.18 * t
    else:
        r = np.clip(2.2 * t, 0.0, 1.0)
        g = np.clip(1.4 - 1.8 * np.abs(t - 0.35), 0.0, 1.0)
        b = np.clip(1.3 - 2.0 * t, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def can_create_open3d_window():
    if o3d is None:
        return False
    vis = o3d.visualization.Visualizer()
    ok = False
    try:
        ok = vis.create_window(window_name="Open3D Probe", width=64, height=64, visible=False)
    except Exception as exc:
        print(f"[WARN] Open3D probe failed: {exc}")
        ok = False
    finally:
        try:
            vis.destroy_window()
        except Exception:
            pass
    return bool(ok)


def set_equal_axes_3d(ax, point_groups):
    valid_groups = [g for g in point_groups if g is not None and len(g) > 0]
    if not valid_groups:
        return
    all_points = np.concatenate(valid_groups, axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 1e-8:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def set_equal_axes_2d(ax, x, y):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size == 0 or y.size == 0:
        return
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    radius = max(xmax - xmin, ymax - ymin) * 0.55
    radius = max(radius, 1.0)
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_aspect("equal", adjustable="box")


def flatten_adaptive_patches(data, view_index, color_mode):
    patch_points_view = np.asarray(data["patch_points"], dtype=np.float32)[view_index]
    patch_mask_view = np.asarray(data["patch_mask"], dtype=bool)[view_index]
    patch_counts_view = np.asarray(data["patch_counts"], dtype=np.int32)[view_index]
    patch_centers_view = np.asarray(data["patch_centers"], dtype=np.float32)[view_index]
    patch_point_conf_view = np.asarray(data["patch_point_confidence"], dtype=np.float32)[view_index]
    patch_point_band_view = np.asarray(data["patch_point_band_width"], dtype=np.float32)[view_index]
    patch_conf_view = np.asarray(data["patch_confidence"], dtype=np.float32)[view_index]
    patch_band_view = np.asarray(data["patch_band_width"], dtype=np.float32)[view_index]

    num_fingers, num_rounds, points_per_patch, _ = patch_points_view.shape
    flat_points = []
    flat_colors = []
    flat_conf = []
    flat_band = []
    center_points = []
    center_colors = []
    center_conf = []
    center_band = []
    center_owner = []

    palette = finger_palette()
    for finger_idx in range(num_fingers):
        finger_center = patch_centers_view[finger_idx]
        finger_has_valid_patch = False
        finger_conf_values = []
        finger_band_values = []
        for round_idx in range(num_rounds):
            if not bool(patch_mask_view[finger_idx, round_idx]):
                continue
            kept_count = min(points_per_patch, max(1, int(patch_counts_view[finger_idx, round_idx])))
            patch_points = patch_points_view[finger_idx, round_idx][:kept_count]
            point_conf = patch_point_conf_view[finger_idx, round_idx][:kept_count]
            point_band = patch_point_band_view[finger_idx, round_idx][:kept_count]
            if color_mode == "single":
                colors = np.tile(np.asarray([[0.10, 0.85, 0.20]], dtype=np.float32), (len(patch_points), 1))
            elif color_mode == "finger":
                colors = np.tile(palette[finger_idx % len(palette)][None, :], (len(patch_points), 1))
            elif color_mode == "band":
                colors = scalar_to_rgb(point_band, mode="band")
            else:
                colors = scalar_to_rgb(point_conf, mode="confidence")

            flat_points.append(patch_points)
            flat_colors.append(colors)
            flat_conf.append(point_conf)
            flat_band.append(point_band)
            finger_conf_values.append(float(patch_conf_view[finger_idx, round_idx]))
            finger_band_values.append(float(patch_band_view[finger_idx, round_idx]))
            finger_has_valid_patch = True

        if finger_has_valid_patch:
            center_points.append(finger_center.astype(np.float32))
            center_owner.append(finger_idx)
            mean_conf = float(np.mean(finger_conf_values))
            mean_band = float(np.mean(finger_band_values))
            center_conf.append(mean_conf)
            center_band.append(mean_band)
            if color_mode == "single":
                center_colors.append(np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
            elif color_mode == "finger":
                center_colors.append(palette[finger_idx % len(palette)])
            elif color_mode == "band":
                center_colors.append(scalar_to_rgb(np.asarray([mean_band], dtype=np.float32), mode="band")[0])
            else:
                center_colors.append(scalar_to_rgb(np.asarray([mean_conf], dtype=np.float32), mode="confidence")[0])

    return {
        "patch_points": np.concatenate(flat_points, axis=0) if flat_points else np.zeros((0, 3), dtype=np.float32),
        "patch_colors": np.concatenate(flat_colors, axis=0) if flat_colors else np.zeros((0, 3), dtype=np.float32),
        "patch_confidence": np.concatenate(flat_conf, axis=0) if flat_conf else np.zeros((0,), dtype=np.float32),
        "patch_band_width": np.concatenate(flat_band, axis=0) if flat_band else np.zeros((0,), dtype=np.float32),
        "center_points": np.asarray(center_points, dtype=np.float32).reshape(-1, 3) if center_points else np.zeros((0, 3), dtype=np.float32),
        "center_colors": np.asarray(center_colors, dtype=np.float32).reshape(-1, 3) if center_colors else np.zeros((0, 3), dtype=np.float32),
        "center_confidence": np.asarray(center_conf, dtype=np.float32).reshape(-1) if center_conf else np.zeros((0,), dtype=np.float32),
        "center_band_width": np.asarray(center_band, dtype=np.float32).reshape(-1) if center_band else np.zeros((0,), dtype=np.float32),
        "center_owner": np.asarray(center_owner, dtype=np.int32).reshape(-1) if center_owner else np.zeros((0,), dtype=np.int32),
    }


def collect_view_visual_payload(
    data,
    view_index,
    show_surface=True,
    show_query=True,
    show_patches=True,
    show_centers=True,
    show_surface_normals=False,
    show_center_normals=True,
    color_mode="confidence",
    max_surface_points=7500,
    max_query_points=10000,
    max_patch_points=18000,
    normal_subsample=500,
    **_unused,
):
    payload = {
        "surface_points": None,
        "surface_colors": None,
        "surface_normal_points": None,
        "surface_normal_vectors": None,
        "query_points": None,
        "query_colors": None,
        "patch_points": None,
        "patch_colors": None,
        "patch_confidence": None,
        "patch_band_width": None,
        "center_points": None,
        "center_colors": None,
        "center_confidence": None,
        "center_band_width": None,
        "center_normal_points": None,
        "center_normal_vectors": None,
        "coverage_ratio": None,
        "reachable_fraction": None,
        "mean_patch_confidence": None,
        "mean_patch_band_width": None,
    }

    if "planning_view_coverage_ratio" in data and view_index < len(data["planning_view_coverage_ratio"]):
        payload["coverage_ratio"] = float(data["planning_view_coverage_ratio"][view_index])
    if "planning_reachable_surface_fraction" in data:
        reachable = np.asarray(data["planning_reachable_surface_fraction"]).reshape(-1)
        if reachable.size > 0:
            payload["reachable_fraction"] = float(reachable[0])

    if show_surface and "surface_points" in data:
        surface_points = np.asarray(data["surface_points"], dtype=np.float32)
        surface_points_vis, surface_idx = subsample_array(surface_points, max_surface_points)
        payload["surface_points"] = surface_points_vis
        payload["surface_colors"] = np.tile(np.asarray([[0.75, 0.75, 0.75]], dtype=np.float32), (len(surface_points_vis), 1))
        if show_surface_normals and "surface_normals" in data:
            surface_normals = np.asarray(data["surface_normals"], dtype=np.float32)
            if surface_idx is not None:
                surface_normals = surface_normals[surface_idx]
            if len(surface_points_vis) > normal_subsample:
                idx = np.random.choice(len(surface_points_vis), normal_subsample, replace=False)
                payload["surface_normal_points"] = surface_points_vis[idx]
                payload["surface_normal_vectors"] = surface_normals[idx]
            else:
                payload["surface_normal_points"] = surface_points_vis
                payload["surface_normal_vectors"] = surface_normals

    if show_query and "query_points" in data and "query_sdf" in data:
        query_points = np.asarray(data["query_points"], dtype=np.float32)
        query_sdf = np.asarray(data["query_sdf"], dtype=np.float32)
        query_points_vis, query_idx = subsample_array(query_points, max_query_points)
        if query_idx is not None:
            query_sdf = query_sdf[query_idx]
        payload["query_points"] = query_points_vis
        payload["query_colors"] = color_query_sdf(query_points_vis, query_sdf)

    if show_patches and "patch_points" in data and "patch_mask" in data:
        flattened = flatten_adaptive_patches(data=data, view_index=view_index, color_mode=color_mode)
        patch_points_vis, patch_idx = subsample_array(flattened["patch_points"], max_patch_points)
        patch_colors = flattened["patch_colors"]
        patch_conf = flattened["patch_confidence"]
        patch_band = flattened["patch_band_width"]
        if patch_idx is not None:
            patch_colors = patch_colors[patch_idx]
            patch_conf = patch_conf[patch_idx]
            patch_band = patch_band[patch_idx]
        payload["patch_points"] = patch_points_vis
        payload["patch_colors"] = patch_colors
        payload["patch_confidence"] = patch_conf
        payload["patch_band_width"] = patch_band
        if patch_conf is not None and len(patch_conf) > 0:
            payload["mean_patch_confidence"] = float(np.mean(flattened["patch_confidence"]))
        if patch_band is not None and len(patch_band) > 0:
            payload["mean_patch_band_width"] = float(np.mean(flattened["patch_band_width"]))

        if show_centers and len(flattened["center_points"]) > 0:
            payload["center_points"] = flattened["center_points"]
            payload["center_colors"] = flattened["center_colors"]
            payload["center_confidence"] = flattened["center_confidence"]
            payload["center_band_width"] = flattened["center_band_width"]
            if show_center_normals and "touch_center_normals" in data:
                center_normals_all = np.asarray(data["touch_center_normals"], dtype=np.float32)
                if view_index < len(center_normals_all):
                    center_normals = center_normals_all[view_index]
                    valid_center_normals = []
                    for finger_idx in flattened["center_owner"]:
                        if 0 <= int(finger_idx) < len(center_normals):
                            valid_center_normals.append(center_normals[int(finger_idx)])
                    if valid_center_normals:
                        payload["center_normal_points"] = flattened["center_points"]
                        payload["center_normal_vectors"] = np.asarray(valid_center_normals, dtype=np.float32)

    return payload


def payload_has_visuals(payload):
    for key in ("surface_points", "query_points", "patch_points", "center_points"):
        points = payload.get(key)
        if points is not None and len(points) > 0:
            return True
    return False


def build_open3d_geometries(payload, normal_scale=0.03, center_radius=0.02):
    geometries = []
    if payload["surface_points"] is not None:
        geometries.append(make_point_cloud(payload["surface_points"], payload["surface_colors"]))
    if payload["surface_normal_points"] is not None and payload["surface_normal_vectors"] is not None:
        geometries.append(make_normals_lineset(payload["surface_normal_points"], payload["surface_normal_vectors"], scale=normal_scale, color=(1.0, 1.0, 0.0)))
    if payload["query_points"] is not None:
        geometries.append(make_point_cloud(payload["query_points"], payload["query_colors"]))
    if payload["patch_points"] is not None:
        geometries.append(make_point_cloud(payload["patch_points"], payload["patch_colors"]))
    if payload["center_points"] is not None and len(payload["center_points"]) > 0:
        center_mesh = make_spheres_at_points(payload["center_points"], radius=center_radius, colors=payload["center_colors"])
        if center_mesh is not None:
            geometries.append(center_mesh)
    if payload["center_normal_points"] is not None and payload["center_normal_vectors"] is not None:
        geometries.append(make_normals_lineset(payload["center_normal_points"], payload["center_normal_vectors"], scale=normal_scale, color=(1.0, 0.9, 0.1)))
    return geometries


def visualize_one_view(data, view_index, **kwargs):
    payload = collect_view_visual_payload(data=data, view_index=view_index, **kwargs)
    if not payload_has_visuals(payload):
        print(f"No valid geometry found for view {view_index}.")
        return
    print(
        f"[VIEW {view_index}] "
        f"coverage={payload['coverage_ratio'] if payload['coverage_ratio'] is not None else 'n/a'} "
        f"mean_conf={payload['mean_patch_confidence'] if payload['mean_patch_confidence'] is not None else 'n/a'} "
        f"mean_band={payload['mean_patch_band_width'] if payload['mean_patch_band_width'] is not None else 'n/a'}"
    )
    geometries = build_open3d_geometries(
        payload,
        normal_scale=kwargs.get("normal_scale", 0.03),
        center_radius=kwargs.get("center_radius", 0.02),
    )
    if not geometries:
        print(f"No valid geometry found for view {view_index}.")
        return
    window_name = f"Adaptive Tactile Visualization - View {view_index}"
    if payload["coverage_ratio"] is not None:
        window_name += f" | coverage={payload['coverage_ratio']:.4f}"
    if payload["mean_patch_confidence"] is not None:
        window_name += f" | conf={payload['mean_patch_confidence']:.3f}"
    if payload["mean_patch_band_width"] is not None:
        window_name += f" | band={payload['mean_patch_band_width']:.4f}"
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name=window_name, width=1400, height=900, visible=True):
        raise RuntimeError("Failed to create Open3D visualization window.")
    try:
        for geometry in geometries:
            vis.add_geometry(geometry)
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        render_option.point_size = float(kwargs.get("point_size", 4.0))
        render_option.light_on = True
        vis.run()
    finally:
        vis.destroy_window()


def visualize_adaptive_npz_each_view(npz_path, view_index=None, **kwargs):
    data = np.load(npz_path, allow_pickle=True)
    print("========== NPZ Keys ==========")
    for key in data.files:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
    print("==============================")

    if "patch_points" not in data:
        print("No patch_points found in npz.")
        return

    num_views = int(data["patch_points"].shape[0])
    print(f"Total tactile views: {num_views}")
    selected_views = list(range(num_views)) if view_index is None else [int(view_index)]
    if o3d is None:
        raise RuntimeError("open3d is not available in the current environment.")
    if not can_create_open3d_window():
        raise RuntimeError(
            "Open3D window creation failed in the current environment. "
            "This script is now Open3D-only, so please run it in a desktop environment with OpenGL support."
        )

    for current_view in selected_views:
        if current_view < 0 or current_view >= num_views:
            raise IndexError(f"view_index={current_view} out of range for {num_views} views.")
        visualize_one_view(data=data, view_index=current_view, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    npz_path = args.npz_path or find_default_npz()
    visualize_adaptive_npz_each_view(
        npz_path=npz_path,
        view_index=args.view_index,
        show_surface=not bool(args.hide_surface),
        show_query=bool(args.show_query),
        show_patches=not bool(args.hide_patches),
        show_centers=not bool(args.hide_centers),
        show_surface_normals=bool(args.show_surface_normals),
        show_center_normals=not bool(args.hide_center_normals),
        color_mode=args.color_mode,
        max_surface_points=int(args.max_surface_points),
        max_query_points=int(args.max_query_points),
        max_patch_points=int(args.max_patch_points),
        normal_subsample=int(args.normal_subsample),
        normal_scale=float(args.normal_scale),
        center_radius=float(args.center_radius),
        point_size=float(args.point_size),
    )
