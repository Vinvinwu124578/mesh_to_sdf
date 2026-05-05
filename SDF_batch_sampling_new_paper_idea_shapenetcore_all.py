import os

from SDF_batch_sampling_new_paper_idea import process_single_obj_to_merged_npz


def iter_category_dirs(root_dir, category_names=None):
    if category_names is None:
        for name in sorted(os.listdir(root_dir)):
            category_dir = os.path.join(root_dir, name)
            if os.path.isdir(category_dir) and not name.startswith("."):
                yield category_dir
        return

    for name in category_names:
        category_dir = os.path.join(root_dir, name)
        if os.path.isdir(category_dir):
            yield category_dir
        else:
            print(f"[WARN] category folder not found: {category_dir}")


def find_shapenet_obj_files(category_dir, max_objects=None):
    obj_paths = []

    for current_root, _, files in os.walk(category_dir):
        if os.path.basename(current_root) != "models":
            continue

        if "model_normalized.obj" in files:
            obj_paths.append(os.path.join(current_root, "model_normalized.obj"))

    obj_paths.sort()

    if max_objects is not None:
        obj_paths = obj_paths[:max_objects]

    return obj_paths


def build_output_path(obj_path, category_dir, output_folder_name):
    rel_path = os.path.relpath(obj_path, category_dir)
    rel_npz_path = os.path.splitext(rel_path)[0] + ".npz"
    return os.path.join(category_dir, output_folder_name, rel_npz_path)


def process_shapenetcore_all_categories(
    root_dir,
    category_names=None,
    max_objects_per_category=None,
    num_tactile_samples=10,
    output_folder_name="tactistruct_npz_all",
):
    category_dirs = list(iter_category_dirs(root_dir, category_names=category_names))

    if not category_dirs:
        print("[WARN] no category folders found under:", root_dir)
        return

    print(f"[INFO] found {len(category_dirs)} category folders under root.")

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
            out_path = build_output_path(
                obj_path=obj_path,
                category_dir=category_dir,
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
                )
            except Exception as e:
                print("[FAILED]", obj_path)
                print("Error:", e)


if __name__ == "__main__":
    root_dir = r"C:/Users/wudaw/Downloads/ShapeNetCore/ShapeNetCore"

    process_shapenetcore_all_categories(
        root_dir=root_dir,
        category_names=None,
        max_objects_per_category=275,
        num_tactile_samples=10,
        output_folder_name="tactistruct_npz_shapenet",
    )

    print("\nAll done.")
