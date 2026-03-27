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





import numpy as np
import open3d as o3d


def make_point_cloud(points, colors):
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


def visualize_one_round(
    data,
    round_idx,
    show_surface=True,
    show_touch=True,
    show_query=True,
    show_normals=False,
    color_touch_by_rounds=True,
    max_surface_points=15000,
    max_touch_points=15000,
    max_query_points=300000,
    normal_scale=0.03,
    normal_subsample=500,
    center_radius=0.025
):
    geometries = []

    # --------------------------------------------------
    # 1. surface_points
    # --------------------------------------------------
    if show_surface and "surface_points" in data:
        surface_points = data["surface_points"]
        surface_points_vis, surface_idx = subsample_array(surface_points, max_surface_points)

        surface_colors = np.tile(
            np.array([[0.75, 0.75, 0.75]], dtype=np.float32),
            (len(surface_points_vis), 1)
        )
        surface_pcd = make_point_cloud(surface_points_vis, surface_colors)
        geometries.append(surface_pcd)

        if show_normals and "surface_normals" in data:
            surface_normals = data["surface_normals"]
            if surface_idx is not None:
                surface_normals = surface_normals[surface_idx]

            if len(surface_points_vis) > normal_subsample:
                idx = np.random.choice(len(surface_points_vis), normal_subsample, replace=False)
                n_points = surface_points_vis[idx]
                n_normals = surface_normals[idx]
            else:
                n_points = surface_points_vis
                n_normals = surface_normals

            normal_lines = make_normals_lineset(
                n_points,
                n_normals,
                scale=normal_scale
            )
            geometries.append(normal_lines)

    # --------------------------------------------------
    # 2. touch_points of one round
    # --------------------------------------------------
    if show_touch and "touch_points" in data:
        touch_points_all = data["touch_points"]   # (R, N, 3)

        if round_idx >= len(touch_points_all):
            print(f"round_idx={round_idx} out of range.")
            return

        touch_points = touch_points_all[round_idx]   # (N, 3)
        touch_points_vis, touch_idx = subsample_array(touch_points, max_touch_points)

        if color_touch_by_rounds and "touch_round_ids" in data:
            touch_round_ids_all = data["touch_round_ids"]  # (R, N)
            touch_round_ids = touch_round_ids_all[round_idx]  # (N,)
            if touch_idx is not None:
                touch_round_ids = touch_round_ids[touch_idx]
            touch_colors = color_touch_by_round(touch_points_vis, touch_round_ids)
        else:
            touch_colors = np.tile(
                np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
                (len(touch_points_vis), 1)
            )

        touch_pcd = make_point_cloud(touch_points_vis, touch_colors)
        geometries.append(touch_pcd)

    # --------------------------------------------------
    # 3. query_points + query_sdf
    # --------------------------------------------------
    if show_query and "query_points" in data and "query_sdf" in data:
        query_points = data["query_points"]
        query_sdf = data["query_sdf"]

        query_points_vis, query_idx = subsample_array(query_points, max_query_points)
        if query_idx is not None:
            query_sdf = query_sdf[query_idx]

        query_colors = color_query_sdf(query_points_vis, query_sdf, eps=1e-3)
        query_pcd = make_point_cloud(query_points_vis, query_colors)
        geometries.append(query_pcd)

    # --------------------------------------------------
    # 4. touch centers of one round
    # --------------------------------------------------
    if "touch_centers" in data:
        all_centers = data["touch_centers"]   # (R, K, 3)

        if round_idx < len(all_centers):
            center_points = np.asarray(all_centers[round_idx], dtype=np.float32)  # (K, 3)
            center_mesh = make_spheres_at_points(
                center_points,
                radius=center_radius,
                color=(1.0, 1.0, 1.0)
            )
            if center_mesh is not None:
                geometries.append(center_mesh)

    # --------------------------------------------------
    # 5. fallback: touch_center_ids of one round
    # --------------------------------------------------
    elif "touch_center_ids" in data and "surface_points" in data:
        all_center_ids = data["touch_center_ids"]   # (R, N) or maybe (R, K)
        surface_points = data["surface_points"]

        if round_idx < len(all_center_ids):
            center_ids = np.asarray(all_center_ids[round_idx]).reshape(-1)

            valid_centers = []
            unique_ids = np.unique(center_ids)
            for cid in unique_ids:
                cid = int(cid)
                if 0 <= cid < len(surface_points):
                    valid_centers.append(surface_points[cid])

            if len(valid_centers) > 0:
                valid_centers = np.array(valid_centers, dtype=np.float32)
                center_mesh = make_spheres_at_points(
                    valid_centers,
                    radius=center_radius,
                    color=(1.0, 1.0, 1.0)
                )
                if center_mesh is not None:
                    geometries.append(center_mesh)

    if len(geometries) == 0:
        print(f"No valid geometry found for round {round_idx}.")
        return

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"TactiStruct Visualization - Round {round_idx}",
        width=1400,
        height=900
    )


def visualize_tactistruct_npz_open3d_each_round(
    npz_path,
    show_surface=True,
    show_touch=True,
    show_query=True,
    show_normals=False,
    color_touch_by_rounds=True,
    max_surface_points=15000,
    max_touch_points=15000,
    max_query_points=30000,
    normal_scale=0.03,
    normal_subsample=500,
    center_radius=0.025
):
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

    for round_idx in range(num_rounds):
        print(f"Opening window for round {round_idx} ...")
        visualize_one_round(
            data=data,
            round_idx=round_idx,
            show_surface=show_surface,
            show_touch=show_touch,
            show_query=show_query,
            show_normals=show_normals,
            color_touch_by_rounds=color_touch_by_rounds,
            max_surface_points=max_surface_points,
            max_touch_points=max_touch_points,
            max_query_points=max_query_points,
            normal_scale=normal_scale,
            normal_subsample=normal_subsample,
            center_radius=center_radius
        )


if __name__ == "__main__":
    npz_path = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/sofa/tactistruct_npz_train/sofa_0001                                                                                                                                                                                   .npz"

    visualize_tactistruct_npz_open3d_each_round(
        npz_path=npz_path,
        show_surface=True,
        show_touch=True,
        show_query=True,
        show_normals=True,
        color_touch_by_rounds=True,
        max_surface_points=7500,
        max_touch_points=15000,
        max_query_points=0,
        normal_scale=0.03,
        normal_subsample=500,
        center_radius=0.02
    )