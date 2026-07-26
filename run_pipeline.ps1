$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

conda run --no-capture-output -n EGNN python -m pip install -e ".[test]"
if ($LASTEXITCODE -ne 0) { throw "Environment installation failed." }

$steps = @(
    @("-m", "mg2si.cli", "ingest"),
    @("-m", "mg2si.cli", "validate-data"),
    @("-m", "mg2si.cli", "evaluate"),
    @("-m", "pytest"),
    @("-m", "mg2si.cli", "recommend", "--branch", "synthetic", "--tumor-cell-line", "Huh-7", "--normal-cell-line", "THLE", "--allow-direct-baseline"),
    @("-m", "mg2si.cli", "validate-candidates"),
    @("scripts/generate_figures.py"),
    @("-m", "mg2si.cli", "clean-derived")
)

foreach ($arguments in $steps) {
    Write-Host "[pipeline] python $arguments"
    & conda run --no-capture-output -n EGNN python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline step failed: python $arguments (exit code $LASTEXITCODE)" }
}

Write-Host "[pipeline] completed; canonical data and model outputs are in data/processed/mg2si.sqlite."
