$ErrorActionPreference = "Stop"

# Keep conda's captured child-process output readable on Windows.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$steps = @(
    "rebuild_material_cell_dataset.py",
    "export_tace_subtables.py",
    "prepare_bo_dataset.py",
    "mobo_demo.py",
    "validate_bo_contract.py"
)

foreach ($step in $steps) {
    Write-Host "[pipeline] $step"
    conda run --no-capture-output -n EGNN python $step
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline step failed: $step (exit code $LASTEXITCODE)"
    }
}

Write-Host "[pipeline] completed"
