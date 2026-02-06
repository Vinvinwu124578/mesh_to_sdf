# import numpy as np
# import os

# # ================== 在这里填写两个 npz 文件路径 ==================
# NPZ_FILE_1 = "C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/sdf_npz_train/airplane_0001.npz"
# NPZ_FILE_2 = "C:/Users/wudaw/OneDrive - University of Bristol/Desktop/airplane/sdf_npz_train/airplane_0001.npz"
# # ==================================================================

# def compare_npz(file1, file2):
#     if not os.path.exists(file1):
#         print(f"❌ File not found: {file1}")
#         return False
#     if not os.path.exists(file2):
#         print(f"❌ File not found: {file2}")
#         return False

#     data1 = np.load(file1, allow_pickle=True)
#     data2 = np.load(file2, allow_pickle=True)

#     keys1 = set(data1.files)
#     keys2 = set(data2.files)

#     # 1. Compare keys
#     if keys1 != keys2:
#         print("❌ Keys are different")
#         print("Only in file1:", keys1 - keys2)
#         print("Only in file2:", keys2 - keys1)
#         return False

#     # 2. Compare data for each key
#     for key in keys1:
#         arr1 = data1[key]
#         arr2 = data2[key]

#         if arr1.shape != arr2.shape:
#             print(f"❌ Shape mismatch for key '{key}': {arr1.shape} vs {arr2.shape}")
#             return False

#         if arr1.dtype != arr2.dtype:
#             print(f"❌ Dtype mismatch for key '{key}': {arr1.dtype} vs {arr2.dtype}")
#             return False

#         if not np.array_equal(arr1, arr2):
#             print(f"❌ Value mismatch for key '{key}'")
#             diff_count = np.sum(arr1 != arr2)
#             print(f"   Different elements: {diff_count}")
#             return False

#     print("✅ The two NPZ files are completely identical.")
#     return True


# if __name__ == "__main__":
#     compare_npz(NPZ_FILE_1, NPZ_FILE_2)



import numpy as np
import os

# ================== 在这里填写两个 npz 文件路径 ==================
NPZ_FILE_1 = "C:/Users/wudaw/OneDrive - University of Bristol/Desktop/ModelNet40/airplane/sdf_npz_train/airplane_0001.npz"
NPZ_FILE_2 = "C:/Users/wudaw/OneDrive - University of Bristol/Desktop/airplane/sdf_npz_train/airplane_0001.npz"
# ==================================================================

# def print_npz_contents(file_path, name):
#     print(f"\n{'=' * 80}")
#     print(f"📦 Contents of {name}: {file_path}")
#     print(f"{'=' * 80}")

#     if not os.path.exists(file_path):
#         print(f"❌ File not found: {file_path}")
#         return

#     data = np.load(file_path, allow_pickle=True)

#     if len(data.files) == 0:
#         print("⚠️ NPZ file is empty")
#         return

#     for key in data.files:
#         arr = data[key]
#         print(f"\n🔑 Key: {key}")
#         print(f"   Shape: {arr.shape}")
#         print(f"   Dtype: {arr.dtype}")
#         print("   Data:")

#         # 关键设置：不省略任何数据
#         with np.printoptions(threshold=np.inf, linewidth=200):
#             print(arr)


# if __name__ == "__main__":
#     print_npz_contents(NPZ_FILE_1, "NPZ_FILE_1")
#     print_npz_contents(NPZ_FILE_2, "NPZ_FILE_2")

# import numpy as np
# import os


# def compare_shapes(file1, file2):
#     data1 = np.load(file1, allow_pickle=True)
#     data2 = np.load(file2, allow_pickle=True)

#     keys1 = set(data1.files)
#     keys2 = set(data2.files)

#     if keys1 != keys2:
#         print("❌ Key set mismatch")
#         print("Only in file1:", keys1 - keys2)
#         print("Only in file2:", keys2 - keys1)
#         return

#     print("✅ Keys are identical\n")

#     all_same = True
#     for key in sorted(keys1):
#         shape1 = data1[key].shape
#         shape2 = data2[key].shape

#         if shape1 == shape2:
#             print(f"✔ {key}: shape = {shape1}")
#         else:
#             print(f"❌ {key}: shape mismatch {shape1} vs {shape2}")
#             all_same = False

#     if all_same:
#         print("\n🎯 All array shapes are identical.")
#     else:
#         print("\n⚠️ Some array shapes differ.")

# if __name__ == "__main__":
#     compare_shapes(NPZ_FILE_1, NPZ_FILE_2)


import numpy as np


def get_bbox_extent(points):
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    extent = max_xyz - min_xyz
    return min_xyz, max_xyz, extent

data1 = np.load(NPZ_FILE_1)
data2 = np.load(NPZ_FILE_2)

p1 = data1["points"]
p2 = data2["points"]

min1, max1, ext1 = get_bbox_extent(p1)
min2, max2, ext2 = get_bbox_extent(p2)

print("=== File 1 ===")
print("Min:", min1)
print("Max:", max1)
print("Extent (X,Y,Z):", ext1)

print("\n=== File 2 ===")
print("Min:", min2)
print("Max:", max2)
print("Extent (X,Y,Z):", ext2)

print("\n=== Size Ratio (file2 / file1) ===")
print(ext2 / ext1)

