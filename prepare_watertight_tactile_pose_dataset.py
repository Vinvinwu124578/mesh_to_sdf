from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_INPUT_DIR = (
    r"C:\Users\wudaw\OneDrive - University of Bristol\Desktop"
    r"\3D_Printing_Objects\watertight meshes"
)
DEFAULT_WORK_DIR = (
    r"C:\Users\wudaw\Downloads\mesh_to_sdf-master\mesh_to_sdf-master"
    r"\mesh_to_sdf\watertight_tactile_pose_work"
)
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\wudaw\OneDrive - University of Bristol\Desktop"
    r"\3D_Printing_Objects\watertight_tactile_pose_dataset_32"
)
DEFAULT_PYTHON = r"C:\Users\wudaw\anaconda3\envs\diffusionSDF\python.exe"
DEFAULT_SAMPLER_SCRIPT = (
    r"C:\Users\wudaw\Downloads\mesh_to_sdf-master\mesh_to_sdf-master"
    r"\mesh_to_sdf\SDF_batch_sampling_new_paper_idea_shapenetcore_all_10touch_mujoco_coverage_onefolder_manifoldplus.py"
)
DEFAULT_POSE_SCRIPT = (
    r"C:\Users\wudaw\Downloads\Tactistruct-main\Tactistruct-main"
    r"\prepare_progressive_attn_fix_pose_dataset.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tactile NPZ dataset from a folder of watertight meshes, then "
            "augment each base object with multiple rigid poses."
        )
    )
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python-exe", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--sampler-script", type=str, default=DEFAULT_SAMPLER_SCRIPT)
    parser.add_argument("--pose-script", type=str, default=DEFAULT_POSE_SCRIPT)
    parser.add_argument("--category-name", type=str, default="custom_watertight")
    parser.add_argument("--poses-per-object", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--max-objects", type=int, default=0, help="0 means all meshes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-identity", action="store_true", default=True)
    parser.add_argument("--no-include-identity", dest="include_identity", action="store_false")
    parser.add_argument("--full-rotation", action="store_true", default=True)
    parser.add_argument("--bounded-rotation", dest="full_rotation", action="store_false")
    parser.add_argument("--max-degrees", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-staging", action="store_true")
    parser.add_argument("--overwrite-base", action="store_true")
    parser.add_argument("--overwrite-pose", action="store_true")
    parser.add_argument(
        "--base-output-folder-name",
        type=str,
        default="tactistruct_npz_3dprint_manifoldplus_base",
    )
    parser.add_argument(
        "--asset-folder-name",
        type=str,
        default="tactistruct_npz_3dprint_manifoldplus_assets",
    )
    parser.add_argument("--num-tactile-samples", type=int, default=10)
    parser.add_argument("--tactile-num-fingers", type=int, default=10)
    parser.add_argument("--tactile-points-per-finger", type=int, default=3000)
    parser.add_argument("--num-surface-points", type=int, default=235000)
    parser.add_argument("--num-query-points", type=int, default=250000)
    parser.add_argument("--dense-surface-sample-n", type=int, default=120000)
    parser.add_argument("--candidate-touch-samples", type=int, default=6000)
    parser.add_argument("--mujoco-max-faces", type=int, default=190000)
    parser.add_argument("--manifoldplus-depth", type=int, default=7)
    parser.add_argument(
        "--watertight-mesh-usage",
        type=str,
        default="sign_proxy",
        choices=("sign_proxy", "full_pipeline"),
        help=(
            "sign_proxy keeps the staged mesh for tactile/surface sampling and uses "
            "ManifoldPlus for SDF signs. full_pipeline replaces the downstream mesh."
        ),
    )
    parser.add_argument("--paired-query-eps-min", type=float, default=0.003)
    parser.add_argument("--paired-query-eps-max", type=float, default=0.02)
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return sanitized or "mesh"


def collect_meshes(input_dir: Path, max_objects: int = 0) -> list[Path]:
    extensions = {".stl", ".obj", ".ply", ".off"}
    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in extensions)
    if max_objects and max_objects > 0:
        files = files[: int(max_objects)]
    if not files:
        raise FileNotFoundError(f"No mesh files found in {input_dir}")
    return files


def _deduplicate_vertices(raw_vertices: list[tuple[float, float, float]]):
    vertices: list[tuple[float, float, float]] = []
    vertex_to_index: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []
    for offset in range(0, len(raw_vertices), 3):
        face = []
        for vertex in raw_vertices[offset : offset + 3]:
            index = vertex_to_index.get(vertex)
            if index is None:
                index = len(vertices)
                vertex_to_index[vertex] = index
                vertices.append(vertex)
            face.append(index)
        if len(set(face)) == 3:
            faces.append((face[0], face[1], face[2]))
    return vertices, faces


def load_stl_as_indexed_triangles(path: Path):
    data = path.read_bytes()
    is_binary = False
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        is_binary = 84 + int(triangle_count) * 50 == len(data)

    raw_vertices: list[tuple[float, float, float]] = []
    if is_binary:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        offset = 84
        for _ in range(int(triangle_count)):
            values = struct.unpack_from("<12fH", data, offset)
            raw_vertices.extend(
                [
                    (values[3], values[4], values[5]),
                    (values[6], values[7], values[8]),
                    (values[9], values[10], values[11]),
                ]
            )
            offset += 50
    else:
        text = data.decode("utf-8", errors="ignore")
        pending: list[tuple[float, float, float]] = []
        for raw_line in text.splitlines():
            parts = raw_line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                pending.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(pending) == 3:
                    raw_vertices.extend(pending)
                    pending = []

    vertices, faces = _deduplicate_vertices(raw_vertices)
    if not vertices or not faces:
        raise RuntimeError(f"Could not read triangles from STL file: {path}")
    return vertices, faces


def write_obj(vertices, faces, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# staged watertight mesh for tactile preprocessing\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


def stage_mesh_as_shapenet_model(source_path: Path, staged_obj_path: Path) -> None:
    suffix = source_path.suffix.lower()
    if suffix == ".obj":
        staged_obj_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, staged_obj_path)
        return
    if suffix == ".stl":
        vertices, faces = load_stl_as_indexed_triangles(source_path)
        write_obj(vertices, faces, staged_obj_path)
        return

    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            f"Need trimesh to stage {source_path.suffix} meshes. "
            "Use STL/OBJ or install trimesh."
        ) from exc
    mesh = trimesh.load(str(source_path), force="mesh", process=False)
    if mesh.is_empty:
        raise RuntimeError(f"Empty mesh: {source_path}")
    staged_obj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(staged_obj_path)


def prepare_staging_root(args: argparse.Namespace, input_dir: Path, work_dir: Path) -> tuple[Path, list[dict]]:
    stage_root = work_dir / "shapenet_style_input"
    category_dir = stage_root / sanitize_name(args.category_name)
    if (args.overwrite or args.overwrite_staging) and stage_root.exists():
        shutil.rmtree(stage_root)
    category_dir.mkdir(parents=True, exist_ok=True)

    staged_entries = []
    for mesh_index, source_path in enumerate(collect_meshes(input_dir, args.max_objects)):
        object_name = sanitize_name(source_path.stem)
        object_dir = category_dir / object_name / "models"
        staged_obj_path = object_dir / "model_normalized.obj"
        if not staged_obj_path.exists() or args.overwrite or args.overwrite_staging:
            stage_mesh_as_shapenet_model(source_path, staged_obj_path)
        staged_entries.append(
            {
                "object_index": mesh_index,
                "object_name": object_name,
                "source_mesh": str(source_path),
                "staged_obj": str(staged_obj_path),
            }
        )
        print(f"[stage] {source_path.name} -> {staged_obj_path}")

    summary_path = work_dir / "staging_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(staged_entries, indent=2), encoding="utf-8")
    return stage_root, staged_entries


def run_command(command: list[str], cwd: Path, env_updates: dict[str, str] | None = None) -> None:
    print("\n[run]", " ".join(f'"{item}"' if " " in item else item for item in command))
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if env_updates:
        env.update(env_updates)
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)


def run_base_sampler(args: argparse.Namespace, stage_root: Path, script_dir: Path) -> Path:
    base_npz_dir = stage_root / args.base_output_folder_name
    command = [
        str(Path(args.python_exe)),
        "-u",
        str(Path(args.sampler_script)),
        "--root-dir",
        str(stage_root),
        "--category-names",
        sanitize_name(args.category_name),
        "--max-objects-per-category",
        str(max(1, int(args.max_objects)) if int(args.max_objects) > 0 else 1000000),
        "--max-workers",
        str(args.max_workers),
        "--output-folder-name",
        str(args.base_output_folder_name),
        "--asset-folder-name",
        str(args.asset_folder_name),
        "--num-tactile-samples",
        str(args.num_tactile_samples),
        "--tactile-num-fingers",
        str(args.tactile_num_fingers),
        "--tactile-points-per-finger",
        str(args.tactile_points_per_finger),
        "--num-surface-points",
        str(args.num_surface_points),
        "--num-query-points",
        str(args.num_query_points),
        "--dense-surface-sample-n",
        str(args.dense_surface_sample_n),
        "--candidate-touch-samples",
        str(args.candidate_touch_samples),
        "--mujoco-max-faces",
        str(args.mujoco_max_faces),
        "--watertight-mesh-usage",
        str(args.watertight_mesh_usage),
        "--manifoldplus-depth",
        str(args.manifoldplus_depth),
        "--paired-query-eps-min",
        str(args.paired_query_eps_min),
        "--paired-query-eps-max",
        str(args.paired_query_eps_max),
        "--base-seed",
        str(args.seed),
    ]
    if args.overwrite or args.overwrite_base:
        command.append("--overwrite")
    env_updates = {"MANIFOLDPLUS_TMP_ROOT": str(stage_root.parent / "manifoldplus_tmp")}
    run_command(command, cwd=script_dir, env_updates=env_updates)
    if not any(base_npz_dir.rglob("*.npz")):
        raise RuntimeError(
            "Base tactile sampling finished without producing any NPZ files. "
            f"Check the sampler summary under {base_npz_dir}."
        )
    return base_npz_dir


def run_pose_augmentation(args: argparse.Namespace, base_npz_dir: Path, output_dir: Path, script_dir: Path) -> None:
    command = [
        str(Path(args.python_exe)),
        "-u",
        str(Path(args.pose_script)),
        "--input-dir",
        str(base_npz_dir),
        "--output-dir",
        str(output_dir),
        "--poses-per-object",
        str(args.poses_per_object),
        "--seed",
        str(args.seed),
        "--max-degrees",
        str(args.max_degrees),
    ]
    if args.full_rotation:
        command.append("--full-rotation")
    if args.include_identity:
        command.append("--include-identity")
    if args.overwrite or args.overwrite_pose:
        command.append("--overwrite")
    run_command(command, cwd=script_dir)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent

    stage_root, staged_entries = prepare_staging_root(args, input_dir, work_dir)
    base_npz_dir = run_base_sampler(args, stage_root, script_dir)
    run_pose_augmentation(args, base_npz_dir, output_dir, script_dir)

    base_npz_count = len(list(base_npz_dir.rglob("*.npz")))
    pose_npz_count = len(list(output_dir.rglob("*.npz")))
    summary = {
        "input_dir": str(input_dir),
        "work_dir": str(work_dir),
        "stage_root": str(stage_root),
        "base_npz_dir": str(base_npz_dir),
        "output_dir": str(output_dir),
        "num_objects": len(staged_entries),
        "poses_per_object": int(args.poses_per_object),
        "base_npz_count": int(base_npz_count),
        "pose_npz_count": int(pose_npz_count),
        "full_rotation": bool(args.full_rotation),
        "include_identity": bool(args.include_identity),
    }
    summary_path = output_dir / "tactile_pose_pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[summary] saved {summary_path}")
    print(f"[summary] base_npz={base_npz_count} pose_npz={pose_npz_count}")


if __name__ == "__main__":
    main()
