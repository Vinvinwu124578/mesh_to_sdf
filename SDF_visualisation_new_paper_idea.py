# import numpy as np
# import open3d as o3d


# def make_point_cloud(points, colors):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
#     pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
#     return pcd


# def subsample_array(arr, max_n):
#     if len(arr) <= max_n:
#         return arr, None
#     idx = np.random.choice(len(arr), max_n, replace=False)
#     return arr[idx], idx


# def make_normals_lineset(points, normals, scale=0.03):
#     points = points.astype(np.float64)
#     normals = normals.astype(np.float64)

#     norm = np.linalg.norm(normals, axis=1, keepdims=True)
#     normals = normals / np.clip(norm, 1e-8, None)

#     end_points = points + normals * scale

#     line_points = np.vstack([points, end_points])

#     n = len(points)
#     lines = [[i, i + n] for i in range(n)]
#     colors = [[1.0, 1.0, 0.0] for _ in range(n)]  # yellow

#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(line_points)
#     line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
#     line_set.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
#     return line_set


# def color_query_sdf(points, sdf, eps=1e-3):
#     colors = np.zeros((len(points), 3), dtype=np.float32)

#     inside = sdf < -eps
#     outside = sdf > eps
#     near = np.abs(sdf) <= eps

#     colors[inside] = [0.0, 0.0, 1.0]   # blue
#     colors[outside] = [1.0, 0.0, 0.0]  # red
#     colors[near] = [1.0, 1.0, 1.0]     # white

#     return colors


# def color_touch_by_round(touch_points, touch_round_ids):
#     palette = np.array([
#         [0.0, 1.0, 0.0],   # green
#         [1.0, 1.0, 0.0],   # yellow
#         [1.0, 0.5, 0.0],   # orange
#         [1.0, 0.0, 1.0],   # magenta
#         [0.0, 1.0, 1.0],   # cyan
#         [0.5, 1.0, 0.5],
#         [0.5, 0.5, 1.0],
#     ], dtype=np.float32)

#     colors = np.zeros((len(touch_points), 3), dtype=np.float32)
#     for i in range(len(touch_points)):
#         rid = int(touch_round_ids[i]) if i < len(touch_round_ids) else 0
#         colors[i] = palette[rid % len(palette)]
#     return colors


# def make_spheres_at_points(points, radius=0.02, color=(1, 1, 1)):
#     meshes = []
#     for p in points:
#         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
#         sphere.translate(p.astype(np.float64))
#         sphere.paint_uniform_color(color)
#         meshes.append(sphere)

#     if len(meshes) == 0:
#         return None

#     merged = meshes[0]
#     for m in meshes[1:]:
#         merged += m
#     return merged


# def visualize_tactistruct_npz_open3d(
#     npz_path,
#     show_surface=True,
#     show_touch=True,
#     show_query=True,
#     show_normals=False,
#     color_touch_by_rounds=True,
#     max_surface_points=15000,
#     max_touch_points=15000,
#     max_query_points=30000,
#     normal_scale=0.03,
#     normal_subsample=500,
#     center_radius=0.025
# ):
#     data = np.load(npz_path, allow_pickle=True)

#     print("========== NPZ Keys ==========")
#     for k in data.files:
#         arr = data[k]
#         print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
#     print("==============================")

#     geometries = []

#     # --------------------------------------------------
#     # 1. surface_points
#     # --------------------------------------------------
#     if show_surface and "surface_points" in data:
#         surface_points = data["surface_points"]
#         surface_points_vis, surface_idx = subsample_array(surface_points, max_surface_points)

#         surface_colors = np.tile(
#             np.array([[0.75, 0.75, 0.75]], dtype=np.float32),
#             (len(surface_points_vis), 1)
#         )

#         surface_pcd = make_point_cloud(surface_points_vis, surface_colors)
#         geometries.append(surface_pcd)

#         if show_normals and "surface_normals" in data:
#             surface_normals = data["surface_normals"]
#             if surface_idx is not None:
#                 surface_normals = surface_normals[surface_idx]

#             if len(surface_points_vis) > normal_subsample:
#                 idx = np.random.choice(len(surface_points_vis), normal_subsample, replace=False)
#                 n_points = surface_points_vis[idx]
#                 n_normals = surface_normals[idx]
#             else:
#                 n_points = surface_points_vis
#                 n_normals = surface_normals

#             normal_lines = make_normals_lineset(
#                 n_points,
#                 n_normals,
#                 scale=normal_scale
#             )
#             geometries.append(normal_lines)

#     # --------------------------------------------------
#     # 2. touch_points
#     # --------------------------------------------------
#     if show_touch and "touch_points" in data:
#         touch_points = data["touch_points"]
#         touch_points_vis, touch_idx = subsample_array(touch_points, max_touch_points)

#         if color_touch_by_rounds and "touch_round_ids" in data:
#             touch_round_ids = data["touch_round_ids"]
#             if touch_idx is not None:
#                 touch_round_ids = touch_round_ids[touch_idx]
#             touch_colors = color_touch_by_round(touch_points_vis, touch_round_ids)
#         else:
#             touch_colors = np.tile(
#                 np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
#                 (len(touch_points_vis), 1)
#             )

#         touch_pcd = make_point_cloud(touch_points_vis, touch_colors)
#         geometries.append(touch_pcd)

#     # --------------------------------------------------
#     # 3. query_points + query_sdf
#     # --------------------------------------------------
#     if show_query and "query_points" in data and "query_sdf" in data:
#         query_points = data["query_points"]
#         query_sdf = data["query_sdf"]

#         query_points_vis, query_idx = subsample_array(query_points, max_query_points)
#         if query_idx is not None:
#             query_sdf = query_sdf[query_idx]

#         query_colors = color_query_sdf(query_points_vis, query_sdf, eps=1e-3)
#         query_pcd = make_point_cloud(query_points_vis, query_colors)
#         geometries.append(query_pcd)

#     # --------------------------------------------------
#     # 4. touch centers (preferred)
#     # --------------------------------------------------
#     if "touch_centers" in data:
#         center_points = data["touch_centers"].astype(np.float32)
#         center_mesh = make_spheres_at_points(
#             center_points,
#             radius=center_radius,
#             color=(1.0, 1.0, 1.0)
#         )
#         if center_mesh is not None:
#             geometries.append(center_mesh)

#     # --------------------------------------------------
#     # 5. fallback: touch_center_ids
#     # --------------------------------------------------
#     elif "touch_center_ids" in data and "surface_points" in data:
#         center_ids = data["touch_center_ids"]
#         surface_points = data["surface_points"]

#         valid_centers = []
#         for cid in center_ids:
#             cid = int(cid)
#             if 0 <= cid < len(surface_points):
#                 valid_centers.append(surface_points[cid])

#         if len(valid_centers) > 0:
#             valid_centers = np.array(valid_centers, dtype=np.float32)
#             center_mesh = make_spheres_at_points(
#                 valid_centers,
#                 radius=center_radius,
#                 color=(1.0, 1.0, 1.0)
#             )
#             if center_mesh is not None:
#                 geometries.append(center_mesh)

#     if len(geometries) == 0:
#         print("No valid geometry found to visualize.")
#         return

#     o3d.visualization.draw_geometries(
#         geometries,
#         window_name="TactiStruct NPZ Visualization",
#         width=1400,
#         height=900
#     )


# if __name__ == "__main__":
#     npz_path = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/tactistruct_npz_train/airplane_0002.npz"

#     visualize_tactistruct_npz_open3d(
#         npz_path=npz_path,
#         show_surface=True,
#         show_touch=True,
#         show_query=True,
#         show_normals=True,
#         color_touch_by_rounds=True,
#         max_surface_points=15000,
#         max_touch_points=15000,
#         max_query_points=30000,
#         normal_scale=0.03,
#         normal_subsample=500,
#         center_radius=0.02
#     )





import os

import numpy as np

try:
    import open3d as o3d
except Exception:
    o3d = None


def make_point_cloud(points, colors):
    if o3d is None:
        raise RuntimeError("open3d is not available in the current environment.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def subsample_array(arr, max_n):
    if len(arr) <= max_n:
        return arr, None
    idx = np.random.choice(len(arr), max_n, replace=False)
    return arr[idx], idx


def make_normals_lineset(points, normals, scale=0.03):
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
    colors = [[1.0, 1.0, 0.0] for _ in range(n)]  # yellow

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return line_set


def color_query_sdf(points, sdf, eps=1e-3):
    colors = np.zeros((len(points), 3), dtype=np.float32)

    inside = sdf < -eps
    outside = sdf > eps
    near = np.abs(sdf) <= eps

    colors[inside] = [0.0, 0.0, 1.0]   # blue
    colors[outside] = [1.0, 0.0, 0.0]  # red
    colors[near] = [1.0, 1.0, 1.0]     # white

    return colors


def color_touch_by_round(touch_points, touch_round_ids):
    palette = np.array([
        [0.0, 1.0, 0.0],   # green
        [1.0, 1.0, 0.0],   # yellow
        [1.0, 0.5, 0.0],   # orange
        [1.0, 0.0, 1.0],   # magenta
        [0.0, 1.0, 1.0],   # cyan
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
    ], dtype=np.float32)

    touch_round_ids = np.asarray(touch_round_ids).reshape(-1)

    colors = np.zeros((len(touch_points), 3), dtype=np.float32)
    n = min(len(touch_points), len(touch_round_ids))
    for i in range(n):
        rid = int(touch_round_ids[i])
        colors[i] = palette[rid % len(palette)]

    if len(touch_points) > n:
        colors[n:] = [0.0, 1.0, 0.0]

    return colors


def make_spheres_at_points(points, radius=0.02, color=(1, 1, 1)):
    if o3d is None:
        raise RuntimeError("open3d is not available in the current environment.")
    points = np.asarray(points).reshape(-1, 3)

    meshes = []
    for p in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(p.astype(np.float64))
        sphere.paint_uniform_color(color)
        meshes.append(sphere)

    if len(meshes) == 0:
        return None

    merged = meshes[0]
    for m in meshes[1:]:
        merged += m
    return merged


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
            visible=False
        )
    except Exception as e:
        print(f"[WARN] Open3D probe failed: {e}")
        ok = False
    finally:
        try:
            vis.destroy_window()
        except Exception:
            pass

    return bool(ok)


def payload_has_visuals(payload):
    return any(
        payload.get(key) is not None and len(payload.get(key)) > 0
        for key in ["surface_points", "touch_points", "query_points", "center_points"]
    )


def default_png_output_dir(npz_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parts = os.path.normpath(npz_path).split(os.sep)
    tail_parts = parts[-4:] if len(parts) >= 4 else parts

    safe_tail = []
    for idx, part in enumerate(tail_parts):
        if idx == len(tail_parts) - 1:
            part = os.path.splitext(part)[0]
        safe_tail.append(part.replace(":", "_"))

    folder_name = "_".join(safe_tail)
    return os.path.join(script_dir, "visualization_exports", folder_name)



def resolve_npz_input_path(npz_path):
    npz_path = os.path.abspath(os.path.expanduser(str(npz_path)))
    if os.path.isfile(npz_path):
        if not npz_path.lower().endswith('.npz'):
            raise ValueError(f'Expected a .npz file, got: {npz_path}')
        return npz_path

    if os.path.isdir(npz_path):
        candidates = []
        for root, _, files in os.walk(npz_path):
            for name in files:
                if name.lower().endswith('.npz'):
                    candidates.append(os.path.join(root, name))
        candidates = sorted(candidates)
        if len(candidates) == 0:
            raise FileNotFoundError(f'No .npz files were found under directory: {npz_path}')
        if len(candidates) == 1:
            print(f'[INFO] npz_path is a directory; using the only .npz file: {candidates[0]}')
            return candidates[0]

        print(f'[INFO] npz_path is a directory; found {len(candidates)} .npz files. Using the first one:')
        for idx, candidate in enumerate(candidates[:10]):
            print(f'  [{idx}] {candidate}')
        if len(candidates) > 10:
            print(f'  ... ({len(candidates) - 10} more)')
        print(f'[INFO] selected: {candidates[0]}')
        return candidates[0]

    raise FileNotFoundError(f'Path does not exist: {npz_path}')

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


def collect_round_visual_payload(
    data,
    round_idx,
    show_surface=True,
    show_touch=True,
    show_query=True,
    show_normals=False,
    show_touch_normals=False,
    color_touch_by_rounds=True,
    max_surface_points=15000,
    max_touch_points=15000,
    max_query_points=300000,
    normal_subsample=500,
    touch_normal_subsample=1500,
):
    payload = {
        "surface_points": None,
        "surface_colors": None,
        "normal_points": None,
        "normal_vectors": None,
        "touch_points": None,
        "touch_colors": None,
        "touch_normal_points": None,
        "touch_normal_vectors": None,
        "query_points": None,
        "query_colors": None,
        "center_points": None,
    }

    if show_surface and "surface_points" in data:
        surface_points = data["surface_points"]
        surface_points_vis, surface_idx = subsample_array(surface_points, max_surface_points)

        surface_colors = np.tile(
            np.array([[0.75, 0.75, 0.75]], dtype=np.float32),
            (len(surface_points_vis), 1)
        )
        payload["surface_points"] = surface_points_vis
        payload["surface_colors"] = surface_colors

        if show_normals and "surface_normals" in data:
            surface_normals = data["surface_normals"]
            if surface_idx is not None:
                surface_normals = surface_normals[surface_idx]

            if len(surface_points_vis) > normal_subsample:
                idx = np.random.choice(len(surface_points_vis), normal_subsample, replace=False)
                payload["normal_points"] = surface_points_vis[idx]
                payload["normal_vectors"] = surface_normals[idx]
            else:
                payload["normal_points"] = surface_points_vis
                payload["normal_vectors"] = surface_normals

    if show_touch and "touch_points" in data:
        touch_points_all = data["touch_points"]

        if round_idx >= len(touch_points_all):
            print(f"round_idx={round_idx} out of range.")
            return None

        touch_points = touch_points_all[round_idx]
        touch_points_vis, touch_idx = subsample_array(touch_points, max_touch_points)

        if color_touch_by_rounds and "touch_round_ids" in data:
            touch_round_ids_all = data["touch_round_ids"]
            touch_round_ids = touch_round_ids_all[round_idx]
            if touch_idx is not None:
                touch_round_ids = touch_round_ids[touch_idx]
            touch_colors = color_touch_by_round(touch_points_vis, touch_round_ids)
        else:
            touch_colors = np.tile(
                np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
                (len(touch_points_vis), 1)
            )

        payload["touch_points"] = touch_points_vis
        payload["touch_colors"] = touch_colors

        if show_touch_normals and "touch_point_normals" in data:
            touch_normals_all = data["touch_point_normals"]
            if round_idx < len(touch_normals_all):
                touch_normals = touch_normals_all[round_idx]
                if touch_idx is not None:
                    touch_normals = touch_normals[touch_idx]

                if len(touch_points_vis) > touch_normal_subsample:
                    idx = np.random.choice(len(touch_points_vis), touch_normal_subsample, replace=False)
                    payload["touch_normal_points"] = touch_points_vis[idx]
                    payload["touch_normal_vectors"] = touch_normals[idx]
                else:
                    payload["touch_normal_points"] = touch_points_vis
                    payload["touch_normal_vectors"] = touch_normals

    if show_query and "query_points" in data and "query_sdf" in data:
        query_points = data["query_points"]
        query_sdf = data["query_sdf"]

        query_points_vis, query_idx = subsample_array(query_points, max_query_points)
        if query_idx is not None:
            query_sdf = query_sdf[query_idx]

        payload["query_points"] = query_points_vis
        payload["query_colors"] = color_query_sdf(query_points_vis, query_sdf, eps=1e-3)

    if "touch_centers" in data:
        all_centers = data["touch_centers"]
        if round_idx < len(all_centers):
            payload["center_points"] = np.asarray(all_centers[round_idx], dtype=np.float32).reshape(-1, 3)
    elif "touch_center_ids" in data and "surface_points" in data:
        all_center_ids = data["touch_center_ids"]
        surface_points = data["surface_points"]

        if round_idx < len(all_center_ids):
            center_ids = np.asarray(all_center_ids[round_idx]).reshape(-1)
            valid_centers = []
            for cid in np.unique(center_ids):
                cid = int(cid)
                if 0 <= cid < len(surface_points):
                    valid_centers.append(surface_points[cid])

            if valid_centers:
                payload["center_points"] = np.asarray(valid_centers, dtype=np.float32)

    return payload


def build_open3d_geometries(payload, normal_scale=0.03, touch_normal_scale=0.02, center_radius=0.025):
    geometries = []

    if payload["surface_points"] is not None:
        geometries.append(
            make_point_cloud(payload["surface_points"], payload["surface_colors"])
        )

    if payload["normal_points"] is not None and payload["normal_vectors"] is not None:
        geometries.append(
            make_normals_lineset(
                payload["normal_points"],
                payload["normal_vectors"],
                scale=normal_scale
            )
        )

    if payload["touch_points"] is not None:
        geometries.append(
            make_point_cloud(payload["touch_points"], payload["touch_colors"])
        )

    if payload["touch_normal_points"] is not None and payload["touch_normal_vectors"] is not None:
        touch_normal_lines = make_normals_lineset(
            payload["touch_normal_points"],
            payload["touch_normal_vectors"],
            scale=touch_normal_scale,
        )
        touch_normal_lines.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.0, 0.8, 0.2]], dtype=np.float64), (len(payload["touch_normal_points"]), 1))
        )
        geometries.append(touch_normal_lines)

    if payload["query_points"] is not None:
        geometries.append(
            make_point_cloud(payload["query_points"], payload["query_colors"])
        )

    if payload["center_points"] is not None and len(payload["center_points"]) > 0:
        center_mesh = make_spheres_at_points(
            payload["center_points"],
            radius=center_radius,
            color=(1.0, 1.0, 1.0)
        )
        if center_mesh is not None:
            geometries.append(center_mesh)

    return geometries


def save_round_matplotlib(payload, round_idx, output_path, normal_scale=0.03, touch_normal_scale=0.02):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[FAILED] matplotlib fallback unavailable: {e}")
        return

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"TactiStruct Visualization - Round {round_idx}")

    if payload["surface_points"] is not None and len(payload["surface_points"]) > 0:
        ax.scatter(
            payload["surface_points"][:, 0],
            payload["surface_points"][:, 1],
            payload["surface_points"][:, 2],
            c=payload["surface_colors"],
            s=1.0,
            alpha=0.20,
            depthshade=False,
        )

    if payload["query_points"] is not None and len(payload["query_points"]) > 0:
        ax.scatter(
            payload["query_points"][:, 0],
            payload["query_points"][:, 1],
            payload["query_points"][:, 2],
            c=payload["query_colors"],
            s=1.0,
            alpha=0.45,
            depthshade=False,
        )

    if payload["touch_points"] is not None and len(payload["touch_points"]) > 0:
        ax.scatter(
            payload["touch_points"][:, 0],
            payload["touch_points"][:, 1],
            payload["touch_points"][:, 2],
            c=payload["touch_colors"],
            s=3.0,
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

    if payload["normal_points"] is not None and len(payload["normal_points"]) > 0:
        normals = payload["normal_vectors"]
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.clip(norm, 1e-8, None)

        ax.quiver(
            payload["normal_points"][:, 0],
            payload["normal_points"][:, 1],
            payload["normal_points"][:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            length=normal_scale,
            normalize=True,
            color=(1.0, 1.0, 0.0),
            linewidth=0.4,
        )

    if payload["touch_normal_points"] is not None and len(payload["touch_normal_points"]) > 0:
        touch_normals = payload["touch_normal_vectors"]
        norm = np.linalg.norm(touch_normals, axis=1, keepdims=True)
        touch_normals = touch_normals / np.clip(norm, 1e-8, None)

        ax.quiver(
            payload["touch_normal_points"][:, 0],
            payload["touch_normal_points"][:, 1],
            payload["touch_normal_points"][:, 2],
            touch_normals[:, 0],
            touch_normals[:, 1],
            touch_normals[:, 2],
            length=touch_normal_scale,
            normalize=True,
            color=(0.0, 0.8, 0.2),
            linewidth=0.45,
        )

    set_equal_axes_3d(
        ax,
        [
            payload["surface_points"],
            payload["query_points"],
            payload["touch_points"],
            payload["center_points"],
        ]
    )
    ax.view_init(elev=24, azim=38)
    ax.set_axis_off()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


def visualize_one_round(
    data,
    round_idx,
    show_surface=True,
    show_touch=True,
    show_query=True,
    show_normals=False,
    show_touch_normals=False,
    color_touch_by_rounds=True,
    max_surface_points=15000,
    max_touch_points=15000,
    max_query_points=300000,
    normal_scale=0.03,
    normal_subsample=500,
    touch_normal_scale=0.02,
    touch_normal_subsample=1500,
    center_radius=0.025,
    render_mode="open3d",
    output_path=None,
):
    payload = collect_round_visual_payload(
        data=data,
        round_idx=round_idx,
        show_surface=show_surface,
        show_touch=show_touch,
        show_query=show_query,
        show_normals=show_normals,
        show_touch_normals=show_touch_normals,
        color_touch_by_rounds=color_touch_by_rounds,
        max_surface_points=max_surface_points,
        max_touch_points=max_touch_points,
        max_query_points=max_query_points,
        normal_subsample=normal_subsample,
        touch_normal_subsample=touch_normal_subsample,
    )

    if payload is None:
        return

    if render_mode == "open3d":
        geometries = build_open3d_geometries(
            payload,
            normal_scale=normal_scale,
            touch_normal_scale=touch_normal_scale,
            center_radius=center_radius,
        )
        if len(geometries) == 0:
            print(f"No valid geometry found for round {round_idx}.")
            return

        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"TactiStruct Visualization - Round {round_idx}",
            width=1400,
            height=900
        )
        return

    if output_path is None:
        raise ValueError("output_path must be provided when render_mode='matplotlib'.")

    if not payload_has_visuals(payload):
        print(f"No valid geometry found for round {round_idx}.")
        return

    save_round_matplotlib(
        payload,
        round_idx=round_idx,
        output_path=output_path,
        normal_scale=normal_scale,
        touch_normal_scale=touch_normal_scale,
    )


def visualize_tactistruct_npz_open3d_each_round(
    npz_path,
    show_surface=True,
    show_touch=True,
    show_query=True,
    show_normals=False,
    show_touch_normals=False,
    color_touch_by_rounds=True,
    max_surface_points=15000,
    max_touch_points=15000,
    max_query_points=30000,
    normal_scale=0.03,
    normal_subsample=500,
    touch_normal_scale=0.02,
    touch_normal_subsample=1500,
    center_radius=0.025,
    fallback_to_png=True,
    png_output_dir=None,
):
    npz_path = resolve_npz_input_path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    print("========== NPZ Keys ==========")
    for k in data.files:
        arr = data[k]
        print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
    print("==============================")

    if "touch_points" not in data:
        print("No touch_points found in npz.")
        return

    num_rounds = data["touch_points"].shape[0]
    print(f"Total tactile rounds: {num_rounds}")

    open3d_available = can_create_open3d_window()
    render_mode = "open3d"

    if not open3d_available:
        if not fallback_to_png:
            print("[WARN] Open3D window creation is unavailable and fallback_to_png=False.")
            return

        render_mode = "matplotlib"
        if png_output_dir is None:
            png_output_dir = default_png_output_dir(npz_path)
        print(f"[INFO] Open3D window unavailable. Saving PNGs to: {png_output_dir}")

    for round_idx in range(num_rounds):
        if render_mode == "open3d":
            print(f"Opening window for round {round_idx} ...")
            output_path = None
        else:
            output_path = os.path.join(png_output_dir, f"round_{round_idx:02d}.png")
            print(f"Saving image for round {round_idx} ...")

        visualize_one_round(
            data=data,
            round_idx=round_idx,
            show_surface=show_surface,
            show_touch=show_touch,
            show_query=show_query,
            show_normals=show_normals,
            show_touch_normals=show_touch_normals,
            color_touch_by_rounds=color_touch_by_rounds,
            max_surface_points=max_surface_points,
            max_touch_points=max_touch_points,
            max_query_points=max_query_points,
            normal_scale=normal_scale,
            normal_subsample=normal_subsample,
            touch_normal_scale=touch_normal_scale,
            touch_normal_subsample=touch_normal_subsample,
            center_radius=center_radius,
            render_mode=render_mode,
            output_path=output_path,
        )


if __name__ == "__main__":
    # npz_path = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/chair/tactistruct_npz_train/bench_0073.npz"
    # npz_path = r"C:/Users/wudaw/Downloads/tacti/tacti/bench_0077.npz"
    # npz_path = r"C:/Users/wudaw/Downloads/ShapeNetCore/ShapeNetCore/bottle/tactistruct_npz_shapenet/02876657/1071fa4cddb2da2fc8724d5673a063a6/models/model_normalized.npz"
    npz_path = r"C:/Users\wudaw\Downloads\ShapeNetCore\ShapeNetCore/tactistruct_npz_shapenet_tactip_mujoco_coverage_full_watertight_manifoldplus"
    visualize_tactistruct_npz_open3d_each_round(
        npz_path=npz_path,
        show_surface=True,
        show_touch=True,
        show_query=True,
        show_normals=True,
        show_touch_normals=True,
        color_touch_by_rounds=True,
        max_surface_points=7500,
        max_touch_points=15000,
        max_query_points=1000000,
        normal_scale=0.03,
        normal_subsample=500,
        touch_normal_scale=0.02,
        touch_normal_subsample=1200,
        center_radius=0.02
    )
