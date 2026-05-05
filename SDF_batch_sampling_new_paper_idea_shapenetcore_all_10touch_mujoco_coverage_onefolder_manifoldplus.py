import runpy
import sys
from pathlib import Path


BASE_SCRIPT = (
    Path(__file__).resolve().parent
    / "SDF_batch_sampling_new_paper_idea_shapenetcore_all_10touch_mujoco_coverage_onefolder_paired_watertight.py"
)
DEFAULT_MANIFOLDPLUS_PATH = (
    "wsl:/mnt/c/Users/wudaw/Downloads/mesh_to_sdf-master/mesh_to_sdf-master/"
    "mesh_to_sdf/external_tools/ManifoldPlus/build_conda_path/manifold"
)


def has_arg(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def append_default(name, value):
    if not has_arg(name):
        sys.argv.extend([name, str(value)])


if __name__ == "__main__":
    append_default("--watertight-mesh-usage", "full_pipeline")
    append_default("--watertight-proxy-mode", "manifoldplus")
    append_default("--manifoldplus-path", DEFAULT_MANIFOLDPLUS_PATH)
    append_default("--non-watertight-policy", "skip")
    append_default(
        "--output-folder-name",
        "tactistruct_npz_shapenet_mujoco_coverage_full_watertight_manifoldplus",
    )
    append_default(
        "--asset-folder-name",
        "tactistruct_npz_shapenet_mujoco_coverage_full_watertight_manifoldplus_assets",
    )
    sys.argv[0] = str(BASE_SCRIPT)
    runpy.run_path(str(BASE_SCRIPT), run_name="__main__")
