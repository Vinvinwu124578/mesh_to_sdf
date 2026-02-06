import numpy as np
import pyrender
import trimesh


def visualize_sdf_npz(npz_path, point_size=2):
    data = np.load(npz_path)
    points = data["points"]
    sdf = data["sdf"]

    colors = np.zeros((points.shape[0], 3))
    colors[sdf < 0] = [0, 0, 1]   # inside → blue
    colors[sdf > 0] = [1, 0, 0]   # outside → red

    cloud = pyrender.Mesh.from_points(points, colors=colors)

    scene = pyrender.Scene()
    scene.add(cloud)

    pyrender.Viewer(
        scene,
        use_raymond_lighting=True,
        point_size=point_size
    )


if __name__ == "__main__":
    npz_path = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/sdf_npz_train_same_as_ori_paper/airplane_0080.npz"
    # npz_path = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/sdf_npz_train/airplane_0001.npz"

    # npz_path = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/chair/sdf_npz_train/chair_0001.npz"
    # npz_path = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/car/sdf_npz_train/car_0196.npz"
    # npz_path = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/airplane/sdf_npz_train_norm/airplane_0006.npz"
    visualize_sdf_npz(npz_path)
