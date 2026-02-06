# # import numpy as np
# # import pyrender
# # import trimesh
# # import matplotlib.cm as cm


# # def visualize_tactile_npz(npz_path, point_size=3):
# #     data = np.load(npz_path)

# #     points = data["points"]        # (N, 3)
# #     finger_id = data["finger_id"]  # (N,)

# #     num_fingers = finger_id.max() + 1

# #     # 使用 matplotlib colormap 给不同手指上色
# #     cmap = cm.get_cmap("tab10", num_fingers)
# #     colors = cmap(finger_id)[:, :3]  # RGB

# #     cloud = pyrender.Mesh.from_points(points, colors=colors)

# #     scene = pyrender.Scene(bg_color=[255, 255, 255])
# #     scene.add(cloud)

# #     pyrender.Viewer(
# #         scene,
# #         use_raymond_lighting=True,
# #         point_size=point_size
# #     )


# # if __name__ == "__main__":
# #     npz_path = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/tactile_npz_train/airplane_0001.npz"
# #     visualize_tactile_npz(npz_path)


# import numpy as np
# import pyrender
# import matplotlib.pyplot as plt


# def visualize_tactile_npz(npz_path, point_size=3):
#     data = np.load(npz_path)

#     points = data["points"]        # (N, 3)
#     finger_id = data["finger_id"]  # (N,)

#     # ✅ 兼容所有 matplotlib 版本
#     cmap = plt.get_cmap("tab10")
#     colors = cmap(finger_id % 10)[:, :3]

#     cloud = pyrender.Mesh.from_points(points, colors=colors)

#     scene = pyrender.Scene(bg_color=[255, 255, 255])
#     scene.add(cloud)

#     # pyrender.Viewer(
#     #     scene,
#     #     use_raymond_lighting=True,
#     #     point_size=point_size
#     # )
#     renderer = pyrender.OffscreenRenderer(640, 480)
#     color, depth = renderer.render(scene)


# if __name__ == "__main__":
#     visualize_tactile_npz(
#         # r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/desk/tactile_npz_train/desk_0128.npz"
#         r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/tactile_npz_train/airplane_0001.npz"
#     )





import open3d as o3d
import numpy as np

def visualize_tactile_npz(npz_path):
    data = np.load(npz_path)

    if "points" not in data:
        raise ValueError("npz must contain 'points'")

    points = data["points"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 可选：上色
    pcd.paint_uniform_color([0.1, 0.6, 0.9])

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Tactile Point Cloud",
        width=800,
        height=600,
    )


if __name__ == "__main__":
    visualize_tactile_npz("C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/tactile_ellipsoid_npz_train/airplane_0012.npz")
    

