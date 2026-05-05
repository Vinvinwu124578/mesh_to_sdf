param(
    [string]$RootDir = "C:\Users\wudaw\Downloads\ShapeNetCore\ShapeNetCore",
    [int]$MaxWorkers = 12,
    [int]$ManifoldDepth = 8,
    [int]$MaxObjectsPerCategory = 275,
    [switch]$OverwriteExisting
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\wudaw\anaconda3\envs\diffusionSDF\python.exe"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PreprocessScript = Join-Path $ScriptDir "SDF_batch_sampling_new_paper_idea_shapenetcore_all_10touch_mujoco_coverage_onefolder_manifoldplus.py"

$Arguments = @(
    $PreprocessScript,
    "--root-dir", $RootDir,
    "--max-objects-per-category", "$MaxObjectsPerCategory",
    "--max-workers", "$MaxWorkers",
    "--manifoldplus-depth", "$ManifoldDepth",
    "--paired-query-eps-min", "0.003",
    "--paired-query-eps-max", "0.02",
    "--paired-query-anchor-mode", "coverage_grid",
    "--paired-query-coverage-grid-size", "16",
    "--paired-query-coverage-min-per-cell", "2",
    "--paired-query-eps-retries", "4"
)

if ($OverwriteExisting) {
    $Arguments += "--overwrite"
}

Write-Host "[INFO] RootDir: $RootDir"
Write-Host "[INFO] MaxWorkers: $MaxWorkers"
Write-Host "[INFO] ManifoldPlus depth: $ManifoldDepth"
Write-Host "[INFO] MaxObjectsPerCategory: $MaxObjectsPerCategory"
Write-Host "[INFO] OverwriteExisting: $OverwriteExisting"
Write-Host "[INFO] Output folder: tactistruct_npz_shapenet_mujoco_coverage_full_watertight_manifoldplus"

& $Python @Arguments
