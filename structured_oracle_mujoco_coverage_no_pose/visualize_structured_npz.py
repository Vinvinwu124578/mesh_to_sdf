import argparse
import os
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
            "Visualize structured MuJoCo coverage-aware tactile NPZ files that store "
            "patch_points / patch_centers instead of flat touch_points."
        )
    )
    parser.add_argument("--npz-path", type=str, default=None)
    parser.add_argument("--view-index", type=int, default=None)
    parser.add_argument("--hide-surface", action="store_true")
    parser.add_argument("--hide-query", action="store_true")
    parser.add_argument("--hide-patches", action="store_true")
    parser.add_argument("--hide-centers", action="store_true")
    parser.add_argument("--show-surface-normals", action="store_true", default=False)
    parser.add_argument("--hide-center-normals", action="store_true")
    parser.add_argument(
        "--color-mode",
        type=str,
        default="single",
        choices=["single", "finger", "round", "combined"],
    )
    parser.add_argument("--max-surface-points", type=int, default=7500)
    parser.add_argument("--max-query-points", type=int, default=10000)
    parser.add_argument("--max-patch-points", type=int, default=15000)
    parser.add_argument("--normal-subsample", type=int, default=500)
    parser.add_argument("--normal-scale", type=float, default=0.03)
    parser.add_argument("--center-radius", type=float, default=0.02)
    parser.add_argument("--force-png", action="store_true")
    parser.add_argument("--png-output-dir", type=str, default=None)
    return parser.parse_args()


def find_default_npz():
    default_dir = DEFAULT_SHAPENET_ROOT / "tactistruct_structured_mujoco_coverage_no_pose_onefolder"
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
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norm, 1e-8, None)
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


def make_spheres_at_points(points, radius=0.02, color=(1.0, 1.0, 1.0)):
    if o3d is None:
        raise RuntimeError("open3d is not available in the current environment.")
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return None

    meshes = []
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point.astype(np.float64))
        sphere.paint_uniform_color(color)
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


def palette():
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


def patch_color(finger_idx, round_idx, color_mode):
    if color_mode == "single":
        return np.asarray([0.10, 0.90, 0.20], dtype=np.float32)
    pal = palette()
    if color_mode == "finger":
        return pal[finger_idx % len(pal)]
    if color_mode == "round":
        return pal[round_idx % len(pal)]
    return pal[(finger_idx * 7 + round_idx) % len(pal)]


def can_create_open3d_window():
    if o3d is None:
        return False
    vis = o3d.visualization.Visualizer()
    ok = False
    try:
        ok = vis.create_window(
            window_name="Open3D Probe",
            width=64,
            height=64,
            visible=False,
        )
    except Exception as exc:
        print(f"[WARN] Open3D probe failed: {exc}")
        ok = False
    finally:
        try:
            vis.destroy_window()
        except Exception:
            pass
    return bool(ok)


def default_png_output_dir(npz_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parts = os.path.normpath(npz_path).split(os.sep)
    tail_parts = parts[-4:] if len(parts) >= 4 else parts
    safe_tail = []
    for idx, part in enumerate(tail_parts):
        if idx == len(tail_parts) - 1:
            part = os.path.splitext(part)[0]
        safe_tail.append(part.replace(":", "_"))
    return os.path.join(script_dir, "visualization_exports_structured", "_".join(safe_tail))


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


def flatten_structured_patches(data, view_index, color_mode):
    patch_points_all = np.asarray(data["patch_points"], dtype=np.float32)
    patch_mask_all = np.asarray(data["patch_mask"], dtype=bool)
    patch_counts_all = np.asarray(data["patch_counts"], dtype=np.int32)
    patch_centers_all = np.asarray(data["patch_centers"], dtype=np.float32)

    patch_points_view = patch_points_all[view_index]
    patch_mask_view = patch_mask_all[view_index]
    patch_counts_view = patch_counts_all[view_index]
    patch_centers_view = patch_centers_all[view_index]

    if patch_points_view.ndim != 4:
        raise ValueError(
            f"Expected patch_points[view] with shape [F, R, P, 3], got {patch_points_view.shape}."
        )

    num_fingers, num_rounds, points_per_patch, _ = patch_points_view.shape

    flat_points = []
    flat_colors = []
    valid_centers = []
    center_owner = []

    for finger_idx in range(num_fingers):
        finger_center = patch_centers_view[finger_idx]
        finger_has_valid_patch = False

        for round_idx in range(num_rounds):
            if not bool(patch_mask_view[finger_idx, round_idx]):
                continue

            stored_patch = patch_points_view[finger_idx, round_idx]
            source_count = int(patch_counts_view[finger_idx, round_idx])
            kept_count = min(points_per_patch, max(1, source_count))
            patch_points = stored_patch[:kept_count]
            color = patch_color(finger_idx, round_idx, color_mode)

            flat_points.append(patch_points)
            flat_colors.append(np.tile(color[None, :], (len(patch_points), 1)))
            finger_has_valid_patch = True

        if finger_has_valid_patch:
            valid_centers.append(finger_center.astype(np.float32))
            center_owner.append(finger_idx)

    patch_points = np.concatenate(flat_points, axis=0) if flat_points else np.zeros((0, 3), dtype=np.float32)
    patch_colors = np.concatenate(flat_colors, axis=0) if flat_colors else np.zeros((0, 3), dtype=np.float32)
    center_points = np.asarray(valid_centers, dtype=np.float32).reshape(-1, 3) if valid_centers else np.zeros((0, 3), dtype=np.float32)
    center_owner = np.asarray(center_owner, dtype=np.int32).reshape(-1) if center_owner else np.zeros((0,), dtype=np.int32)
    return patch_points, patch_colors, center_points, center_owner


def collect_view_visual_payload(
    data,
    view_index,
    show_surface=True,
    show_query=True,
    show_patches=True,
    show_centers=True,
    show_surface_normals=False,
    show_center_normals=True,
    color_mode="finger",
    max_surface_points=7500,
    max_query_points=10000,
    max_patch_points=15000,
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
        "center_points": None,
        "center_colors": None,
        "center_normal_points": None,
        "center_normal_vectors": None,
        "coverage_ratio": None,
    }

    if "planning_view_coverage_ratio" in data and view_index < len(data["planning_view_coverage_ratio"]):
        payload["coverage_ratio"] = float(data["planning_view_coverage_ratio"][view_index])

    if show_surface and "surface_points" in data:
        surface_points = np.asarray(data["surface_points"], dtype=np.float32)
        surface_points_vis, surface_idx = subsample_array(surface_points, max_surface_points)
        surface_colors = np.tile(np.asarray([[0.75, 0.75, 0.75]], dtype=np.float32), (len(surface_points_vis), 1))
        payload["surface_points"] = surface_points_vis
        payload["surface_colors"] = surface_colors

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
        patch_points, patch_colors, center_points, center_owner = flatten_structured_patches(
            data=data,
            view_index=view_index,
            color_mode=color_mode,
        )

        patch_points_vis, patch_idx = subsample_array(patch_points, max_patch_points)
        if patch_idx is not None:
            patch_colors = patch_colors[patch_idx]

        payload["patch_points"] = patch_points_vis
        payload["patch_colors"] = patch_colors

        if show_centers and len(center_points) > 0:
            center_colors = np.asarray([patch_color(int(fid), 0, "finger") for fid in center_owner], dtype=np.float32)
            payload["center_points"] = center_points
            payload["center_colors"] = center_colors

            if show_center_normals and "touch_center_normals" in data:
                center_normals_all = np.asarray(data["touch_center_normals"], dtype=np.float32)
                if view_index < len(center_normals_all):
                    center_normals = center_normals_all[view_index]
                    valid_center_normals = []
                    for finger_idx in center_owner:
                        if 0 <= int(finger_idx) < len(center_normals):
                            valid_center_normals.append(center_normals[int(finger_idx)])
                    if valid_center_normals:
                        payload["center_normal_points"] = center_points
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
        geometries.append(
            make_normals_lineset(
                payload["surface_normal_points"],
                payload["surface_normal_vectors"],
                scale=normal_scale,
                color=(1.0, 1.0, 0.0),
            )
        )

    if payload["query_points"] is not None:
        geometries.append(make_point_cloud(payload["query_points"], payload["query_colors"]))

    if payload["patch_points"] is not None:
        geometries.append(make_point_cloud(payload["patch_points"], payload["patch_colors"]))

    if payload["center_points"] is not None and len(payload["center_points"]) > 0:
        center_mesh = make_spheres_at_points(
            payload["center_points"],
            radius=center_radius,
            color=(1.0, 1.0, 1.0),
        )
        if center_mesh is not None:
            geometries.append(center_mesh)

    if payload["center_normal_points"] is not None and payload["center_normal_vectors"] is not None:
        geometries.append(
            make_normals_lineset(
                payload["center_normal_points"],
                payload["center_normal_vectors"],
                scale=normal_scale,
                color=(1.0, 0.9, 0.1),
            )
        )

    return geometries


def save_view_matplotlib(payload, view_index, output_path, normal_scale=0.03):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[FAILED] matplotlib fallback unavailable: {exc}")
        return

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    title = f"Structured Tactile Visualization - View {view_index}"
    if payload["coverage_ratio"] is not None:
        title += f" (coverage={payload['coverage_ratio']:.4f})"
    ax.set_title(title)

    if payload["surface_points"] is not None and len(payload["surface_points"]) > 0:
        ax.scatter(
            payload["surface_points"][:, 0],
            payload["surface_points"][:, 1],
            payload["surface_points"][:, 2],
            c=payload["surface_colors"],
            s=1.0,
            alpha=0.18,
            depthshade=False,
        )

    if payload["query_points"] is not None and len(payload["query_points"]) > 0:
        ax.scatter(
            payload["query_points"][:, 0],
            payload["query_points"][:, 1],
            payload["query_points"][:, 2],
            c=payload["query_colors"],
            s=1.0,
            alpha=0.40,
            depthshade=False,
        )

    if payload["patch_points"] is not None and len(payload["patch_points"]) > 0:
        ax.scatter(
            payload["patch_points"][:, 0],
            payload["patch_points"][:, 1],
            payload["patch_points"][:, 2],
            c=payload["patch_colors"],
            s=4.0,
            alpha=0.95,
            depthshade=False,
        )

    if payload["center_points"] is not None and len(payload["center_points"]) > 0:
        ax.scatter(
            payload["center_points"][:, 0],
            payload["center_points"][:, 1],
            payload["center_points"][:, 2],
            c="white",
            s=55.0,
            edgecolors="black",
            linewidths=0.6,
            depthshade=False,
        )

    if payload["surface_normal_points"] is not None and len(payload["surface_normal_points"]) > 0:
        normals = payload["surface_normal_vectors"]
        normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
        ax.quiver(
            payload["surface_normal_points"][:, 0],
            payload["surface_normal_points"][:, 1],
            payload["surface_normal_points"][:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            length=normal_scale,
            normalize=True,
            color=(1.0, 1.0, 0.0),
            linewidth=0.4,
        )

    if payload["center_normal_points"] is not None and len(payload["center_normal_points"]) > 0:
        normals = payload["center_normal_vectors"]
        normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
        ax.quiver(
            payload["center_normal_points"][:, 0],
            payload["center_normal_points"][:, 1],
            payload["center_normal_points"][:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            length=normal_scale,
            normalize=True,
            color=(1.0, 0.9, 0.1),
            linewidth=0.7,
        )

    set_equal_axes_3d(
        ax,
        [
            payload["surface_points"],
            payload["query_points"],
            payload["patch_points"],
            payload["center_points"],
        ],
    )
    ax.view_init(elev=24, azim=38)
    ax.set_axis_off()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


def visualize_one_view(
    data,
    view_index,
    render_mode,
    output_path=None,
    **kwargs,
):
    payload = collect_view_visual_payload(
        data=data,
        view_index=view_index,
        **kwargs,
    )
    if not payload_has_visuals(payload):
        print(f"No valid geometry found for view {view_index}.")
        return

    if render_mode == "open3d":
        geometries = build_open3d_geometries(
            payload,
            normal_scale=kwargs.get("normal_scale", 0.03),
            center_radius=kwargs.get("center_radius", 0.02),
        )
        if not geometries:
            print(f"No valid geometry found for view {view_index}.")
            return

        window_name = f"Structured Tactile Visualization - View {view_index}"
        if payload["coverage_ratio"] is not None:
            window_name += f" (coverage={payload['coverage_ratio']:.4f})"
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1400,
            height=900,
        )
        return

    if output_path is None:
        raise ValueError("output_path must be provided when render_mode='matplotlib'.")
    save_view_matplotlib(
        payload,
        view_index=view_index,
        output_path=output_path,
        normal_scale=kwargs.get("normal_scale", 0.03),
    )


def visualize_structured_npz_each_view(
    npz_path,
    view_index=None,
    force_png=False,
    png_output_dir=None,
    **kwargs,
):
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

    if view_index is None:
        selected_views = list(range(num_views))
    else:
        if view_index < 0 or view_index >= num_views:
            raise IndexError(f"view_index={view_index} out of range for {num_views} views.")
        selected_views = [int(view_index)]

    render_mode = "open3d"
    if force_png or not can_create_open3d_window():
        render_mode = "matplotlib"
        if png_output_dir is None:
            png_output_dir = default_png_output_dir(npz_path)
        print(f"[INFO] Saving PNGs to: {png_output_dir}")

    for current_view in selected_views:
        if render_mode == "open3d":
            print(f"Opening window for view {current_view} ...")
            output_path = None
        else:
            output_path = os.path.join(png_output_dir, f"view_{current_view:02d}.png")
            print(f"Saving image for view {current_view} ...")

        visualize_one_view(
            data=data,
            view_index=current_view,
            render_mode=render_mode,
            output_path=output_path,
            **kwargs,
        )


if __name__ == "__main__":
    args = parse_args()
    npz_path = args.npz_path or find_default_npz()

    visualize_structured_npz_each_view(
        npz_path=npz_path,
        view_index=args.view_index,
        force_png=bool(args.force_png),
        png_output_dir=args.png_output_dir,
        show_surface=not bool(args.hide_surface),
        show_query=not bool(args.hide_query),
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
    )
