param(
    [string]$RootDir = "C:\Users\wudaw\Downloads\ShapeNetCore\ShapeNetCore",
    [int]$MaxWorkers = 12,
    [int]$MaxObjectsPerCategory = 275,
    [int]$ManifoldDepth = 8,
    [switch]$OverwriteExisting
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\wudaw\anaconda3\envs\diffusionSDF\python.exe"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConvertScript = Join-Path $ScriptDir "convert_shapenet_to_stl_assets.py"

$Arguments = @(
    $ConvertScript,
    "--root-dir", $RootDir,
    "--max-workers", "$MaxWorkers",
    "--max-objects-per-category", "$MaxObjectsPerCategory",
    "--watertight-proxy-mode", "manifoldplus",
    "--manifoldplus-depth", "$ManifoldDepth",
    "--mujoco-max-faces", "190000"
)

if ($OverwriteExisting) {
    $Arguments += "--overwrite"
}

Write-Host "[INFO] RootDir: $RootDir"
Write-Host "[INFO] MaxWorkers: $MaxWorkers"
Write-Host "[INFO] MaxObjectsPerCategory: $MaxObjectsPerCategory"
Write-Host "[INFO] ManifoldPlus depth: $ManifoldDepth"
Write-Host "[INFO] Output folder: shapenet_stl_assets_full_watertight_manifoldplus"

& $Python @Arguments
