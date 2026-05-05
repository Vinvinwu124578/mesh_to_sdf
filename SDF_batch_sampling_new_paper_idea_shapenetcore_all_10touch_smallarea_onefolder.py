import os

from SDF_batch_sampling_new_paper_idea import process_single_obj_to_merged_npz
from SDF_batch_sampling_new_paper_idea_shapenetcore_all import (
    find_shapenet_obj_files,
    iter_category_dirs,
)


def build_flat_output_path(obj_path, root_dir, output_folder_name):
    rel_path = os.path.relpath(obj_path, root_dir)
    rel_without_ext = os.path.splitext(rel_path)[0]
    safe_name = rel_without_ext.replace("\\", "__").replace("/", "__").replace(":", "_")
    out_dir = os.path.join(root_dir, output_folder_name)
    return os.path.join(out_dir, safe_name + ".npz")


def process_shapenetcore_all_categories_custom_onefolder(
    root_dir,
    category_names=None,
    max_objects_per_category=None,
    num_tactile_samples=10,
    tactile_num_fingers=10,
    tactile_beam_radius=0.08,
    tactile_start_ratio=0.09,
    tactile_end_ratio=0.022,
    tactile_thickness_ratio=0.008,
    tactile_points_per_finger=3000,
    output_folder_name="tactistruct_npz_shapenet_10touch_smallarea_onefolder",
):
    category_dirs = list(iter_category_dirs(root_dir, category_names=category_names))

    if not category_dirs:
        print("[WARN] no category folders found under:", root_dir)
        return

    out_dir = os.path.join(root_dir, output_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] found {len(category_dirs)} category folders under root.")
    print(f"[INFO] flat output dir: {out_dir}")

    for category_dir in category_dirs:
        category_name = os.path.basename(category_dir)
        obj_paths = find_shapenet_obj_files(
            category_dir,
            max_objects=max_objects_per_category,
        )

        print(f"\n########## Processing category: {category_name} ##########")
        print(f"[INFO] found {len(obj_paths)} model_normalized.obj files")

        if not obj_paths:
            continue

        for obj_path in obj_paths:
            out_path = build_flat_output_path(
                obj_path=obj_path,
                root_dir=root_dir,
                output_folder_name=output_folder_name,
            )

            if os.path.exists(out_path):
                print("[SKIP exists]", out_path)
                continue

            try:
                process_single_obj_to_merged_npz(
                    obj_path=obj_path,
                    out_path=out_path,
                    num_tactile_samples=num_tactile_samples,
                    tactile_num_fingers=tactile_num_fingers,
                    tactile_beam_radius=tactile_beam_radius,
                    tactile_start_ratio=tactile_start_ratio,
                    tactile_end_ratio=tactile_end_ratio,
                    tactile_thickness_ratio=tactile_thickness_ratio,
                    tactile_points_per_finger=tactile_points_per_finger,
                )
            except Exception as e:
                print("[FAILED]", obj_path)
                print("Error:", e)


if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/Downloads/ShapeNetCore/ShapeNetCore"

    process_shapenetcore_all_categories_custom_onefolder(
        root_dir=root_dir,
        category_names=None,
        max_objects_per_category=275,
        num_tactile_samples=10,
        tactile_num_fingers=10,
        tactile_beam_radius=0.08,
        tactile_start_ratio=0.09,
        tactile_end_ratio=0.022,
        tactile_thickness_ratio=0.008,
        tactile_points_per_finger=3000,
        output_folder_name="tactistruct_npz_shapenet_10touch_smallarea_onefolder",
    )

    print("\nAll done.")
