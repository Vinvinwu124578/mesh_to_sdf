# import os
# import trimesh
# import numpy as np
# from mesh_to_sdf import sample_sdf_near_surface

# TARGET_CATEGORIES = [
#     "cone",
#     "cup",
#     "curtain",
#     "desk",
#     "door",
#     "dresser",
#     "flower_pot",
#     "glass_box",
#     "guitar",
#     "keyboard",
#     "lamp",
#     "laptop",
#     "mantel",
#     "monitor",
#     "night_stand",
#     "person",
#     "piano",
#     "plant",
#     "radio",
#     "range_hood",
#     "sink",
#     "sofa",
#     "stairs",
#     "stool",
#     "table",
#     "tent",
#     "toilet",
#     "tv_stand",
#     "vase",
#     "wardrobe",
#     "xbox"
# ]

# def process_split(category_dir, split, num_points=250000):
#     """
#     category_dir: ModelNet40/airplane
#     split: 'train' or 'test'
#     """
#     obj_dir = os.path.join(category_dir, f"{split}_obj")
#     out_dir = os.path.join(category_dir, f"sdf_npz_{split}")

#     if not os.path.isdir(obj_dir):
#         return

#     os.makedirs(out_dir, exist_ok=True)

#     for name in os.listdir(obj_dir):
#         if not name.lower().endswith(".obj"):
#             continue

#         obj_path = os.path.join(obj_dir, name)
#         out_path = os.path.join(out_dir, name.replace(".obj", ".npz"))

#         if os.path.exists(out_path):
#             print("Skip (exists):", out_path)
#             continue

#         try:
#             mesh = trimesh.load(obj_path, force='mesh')

#             points, sdf = sample_sdf_near_surface(
#                 mesh,
#                 number_of_points=num_points
#             )

#             np.savez(
#                 out_path,
#                 points=points.astype(np.float32),
#                 sdf=sdf.astype(np.float32)
#             )

#             print("Saved:", out_path)

#         except Exception as e:
#             print("Failed:", obj_path, "Error:", e)


# def process_modelnet40(root_dir, num_points=250000):
#     for category in os.listdir(root_dir):
#         category_dir = os.path.join(root_dir, category)
#         if not os.path.isdir(category_dir):
#             continue

#         print(f"\nProcessing category: {category}")
#         process_split(category_dir, "train", num_points)
#         process_split(category_dir, "test", num_points)


# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
#     process_modelnet40(root_dir, num_points=250000)
#     print("\nAll SDF sampling done.")






# import os
# import trimesh
# import numpy as np
# from mesh_to_sdf import sample_sdf_near_surface

# TARGET_CATEGORIES = [
#     "cone", "cup", "curtain", "desk", "door", "dresser",
#     "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
#     "laptop", "mantel", "monitor", "night_stand", "person",
#     "piano", "plant", "radio", "range_hood", "sink", "sofa",
#     "stairs", "stool", "table", "tent", "toilet", "tv_stand",
#     "vase", "wardrobe", "xbox"
# ]

# def process_split(category_dir, split, num_points=235000):
#     obj_dir = os.path.join(category_dir, f"{split}_obj")
#     out_dir = os.path.join(category_dir, f"sdf_npz_{split}")

#     if not os.path.isdir(obj_dir):
#         print(f"Skip missing dir: {obj_dir}")
#         return

#     os.makedirs(out_dir, exist_ok=True)

#     for name in os.listdir(obj_dir):
#         if not name.lower().endswith(".obj"):
#             continue

#         obj_path = os.path.join(obj_dir, name)
#         out_path = os.path.join(out_dir, name.replace(".obj", ".npz"))

#         if os.path.exists(out_path):
#             print("Skip (exists):", out_path)
#             continue

#         try:
#             mesh = trimesh.load(obj_path, force="mesh")

#             points, sdf = sample_sdf_near_surface(
#                 mesh,
#                 number_of_points=num_points
#             )

#             np.savez(
#                 out_path,
#                 points=points.astype(np.float32),
#                 sdf=sdf.astype(np.float32)
#             )

#             print("Saved:", out_path)

#         except Exception as e:
#             print("Failed:", obj_path, "Error:", e)


# def process_modelnet40(root_dir, num_points=250000):
#     for category in TARGET_CATEGORIES:
#         category_dir = os.path.join(root_dir, category)

#         if not os.path.isdir(category_dir):
#             print(f"Category not found, skip: {category}")
#             continue

#         print(f"\nProcessing category: {category}")
#         process_split(category_dir, "train", num_points)
#         process_split(category_dir, "test", num_points)


# if __name__ == "__main__":
#     root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
#     process_modelnet40(root_dir, num_points=250000)
#     print("\nAll SDF sampling done.")













import os
import trimesh
import numpy as np
from mesh_to_sdf import sample_sdf_near_surface


def process_split(category_dir, split, num_points=235000, max_objects=80):
    obj_dir = os.path.join(category_dir, f"{split}_obj")
    out_dir = os.path.join(category_dir, f"sdf_npz_{split}_same_as_ori_paper")

    if not os.path.isdir(obj_dir):
        print(f"Skip missing dir: {obj_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 只取 .obj，排序后取前 max_objects（可复现）
    obj_files = sorted([f for f in os.listdir(obj_dir) if f.lower().endswith(".obj")])
    obj_files = obj_files[:max_objects]

    print(f"  {split}: found {len(obj_files)} obj files (limit={max_objects})")

    for name in obj_files:
        obj_path = os.path.join(obj_dir, name)
        out_path = os.path.join(out_dir, name[:-4] + ".npz")  # replace .obj -> .npz

        if os.path.exists(out_path):
            print("Skip (exists):", out_path)
            continue

        try:
            mesh = trimesh.load(obj_path, force="mesh")

            points, sdf = sample_sdf_near_surface(
                mesh,
                number_of_points=num_points
            )

            np.savez(
                out_path,
                points=points.astype(np.float32),
                sdf=sdf.astype(np.float32)
            )

            print("Saved:", out_path)

        except Exception as e:
            print("Failed:", obj_path, "Error:", e)


def process_all_categories(root_dir, num_points=235000, max_objects=80):
    # root_dir 下所有一级子文件夹都当作 category_dir
    subdirs = sorted([
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    if not subdirs:
        print("No subfolders found under:", root_dir)
        return

    print(f"Found {len(subdirs)} category folders under root.")

    for category_dir in subdirs:
        category_name = os.path.basename(category_dir)
        print(f"\nProcessing category folder: {category_name}")

        process_split(category_dir, "train", num_points=num_points, max_objects=max_objects)
        # process_split(category_dir, "test",  num_points=num_points, max_objects=max_objects)


if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40"
    process_all_categories(root_dir, num_points=250000, max_objects=80)
    print("\nAll SDF sampling done.")
