<#
.SYNOPSIS
    PowerShell equivalent of `make reproduce-paper` for the semantic-novelty
    audit manuscript (no retraining).

.DESCRIPTION
    Regenerates the strengthening analyses (t2/t3/t6/t9/t10), the cross-
    benchmark figures, the macros, the identity-framework figure, and the
    compiled PDF.  Paths resolve relative to this script, so it can be run
    from any working directory.

.PARAMETER Python
    Python interpreter with numpy/scipy/matplotlib installed.  Optional:
    when omitted (or when "python" does not work), the script probes common
    conda installations (Anaconda/Miniconda "pytorch" and base envs) and
    uses the first one that passes the dependency check.

.PARAMETER AuditRoot
    Path to external_semantic_overlap_audit.  Defaults to the sibling
    directory of semantic_novelty_audit_paper.

.PARAMETER NoCompile
    Skip the final latexmk run (figures only).

.PARAMETER RebuildRootChecksums
    Also rebuild ANONYMIZED_SHA256SUMS at the supplement root (packaging
    step; run only when the tree is final).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\reproduce-paper.ps1

.EXAMPLE
    .\reproduce-paper.ps1 -Python "C:\Users\me\miniconda3\envs\audit\python.exe"
#>

[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$AuditRoot = "",
    [switch]$NoCompile,
    [switch]$RebuildRootChecksums
)

$ErrorActionPreference = "Stop"

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
$tmlrDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$paperDir = Split-Path -Parent $tmlrDir
$supplementRoot = Split-Path -Parent $paperDir

if ([string]::IsNullOrWhiteSpace($AuditRoot)) {
    $AuditRoot = Join-Path $supplementRoot "external_semantic_overlap_audit"
}
$AuditRoot = (Resolve-Path $AuditRoot).Path

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed ($Label): exit code $LASTEXITCODE"
    }
}

function Invoke-PythonScript {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Script not found: $Path"
    }
    & $Python $Path
    if ($LASTEXITCODE -ne 0) {
        throw "Python script failed: $Path (exit $LASTEXITCODE)"
    }
}

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
Write-Host "supplement root : $supplementRoot"
Write-Host "audit root      : $AuditRoot"

function Test-PythonDeps {
    param([string]$Exe)
    if ([string]::IsNullOrWhiteSpace($Exe)) { return $false }
    $null = & $Exe -c "import numpy, scipy, matplotlib" 2>$null
    return ($LASTEXITCODE -eq 0)
}

$pythonCandidates = @()
if ($Python -ne "python" -and -not [string]::IsNullOrWhiteSpace($Python)) {
    # Explicit interpreter: trust it but still verify.
    $pythonCandidates = @($Python)
}
else {
    $pythonCandidates = @("python")
    foreach ($base in @("$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3",
                        "$env:USERPROFILE\Anaconda3", "$env:USERPROFILE\Miniconda3")) {
        $pythonCandidates += (Join-Path $base "envs\pytorch\python.exe")
        $pythonCandidates += (Join-Path $base "envs\digress\python.exe")
        $pythonCandidates += (Join-Path $base "python.exe")
    }
    $pythonCandidates += "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe"
    $pythonCandidates += "python3"
}

$resolved = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -like "*\*" -and -not (Test-Path $candidate)) { continue }
    if (Test-PythonDeps $candidate) { $resolved = $candidate; break }
}

if (-not $resolved) {
    throw ("No Python with numpy/scipy/matplotlib found.  Install them, or pass " +
           "-Python <path to python.exe> (e.g. an Anaconda environment).")
}

$Python = $resolved
Write-Host "python          : $Python"

$Latexmk = $null
if (-not $NoCompile) {
    $Latexmk = (Get-Command latexmk -ErrorAction SilentlyContinue).Source
    if (-not $Latexmk) {
        $fallback = "D:\MiKTeX\miktex\bin\x64\latexmk.exe"
        if (Test-Path $fallback) { $Latexmk = $fallback }
    }
    if (-not $Latexmk) {
        throw "latexmk not found on PATH; install MiKTeX or pass -NoCompile."
    }
    Write-Host "latexmk         : $Latexmk"
}

# --------------------------------------------------------------------------
# 1) Strengthening analyses (t2, t3, t6, t9, t10)
# --------------------------------------------------------------------------
$strengthening = Join-Path $AuditRoot "scripts\tmlr_strengthening"
$scripts = @(
    "run_t2_clutrr_heterogeneity.py",
    "run_t3_internal_heterogeneity.py",
    "analyze_t6_semantic_surface.py",
    "run_t9_property_summary.py",
    "run_t10_novelty_bounds.py"
)
foreach ($name in $scripts) {
    Invoke-Step "Run $name" { Invoke-PythonScript (Join-Path $strengthening $name) }
}

# --------------------------------------------------------------------------
# 2) Cross-benchmark figures (SLR figure, boolean figure, framework figure)
# --------------------------------------------------------------------------
Invoke-Step "Run build_cross_benchmark_figures.py" {
    Invoke-PythonScript (Join-Path $AuditRoot "scripts\build_cross_benchmark_figures.py")
}

# --------------------------------------------------------------------------
# 3) Sync macros + figures into the manuscript, regenerate framework figure
# --------------------------------------------------------------------------
Invoke-Step "Sync artifacts into the manuscript" {
    & $Python (Join-Path $tmlrDir "scripts\copy_strengthening_artifacts.py") $AuditRoot
    if ($LASTEXITCODE -ne 0) { throw "copy_strengthening_artifacts.py failed" }
}

Invoke-Step "Regenerate identity-framework figure" {
    Invoke-PythonScript (Join-Path $tmlrDir "scripts\make_semantic_identity_figure.py")
}

# build_cross_benchmark_figures.py writes into semantic_novelty_audit_paper/figures,
# but LaTeX includes from tmlr/figures: copy the cross-benchmark PDFs across.
Invoke-Step "Copy cross-benchmark figures into tmlr/figures" {
    $srcFigures = Join-Path $paperDir "figures"
    $dstFigures = Join-Path $tmlrDir "figures"
    $names = @(
        "fig_slr_semantic_overlap.pdf",
        "fig_boolean_exposure_effect.pdf"
    )
    foreach ($name in $names) {
        $src = Join-Path $srcFigures $name
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $dstFigures $name) -Force
        }
        else {
            Write-Warning "missing cross-benchmark figure: $src"
        }
    }
}

# --------------------------------------------------------------------------
# 4) Compile
# --------------------------------------------------------------------------
if (-not $NoCompile) {
    Invoke-Step "Compile main.pdf (latexmk)" {
        Push-Location $tmlrDir
        try {
            & $Latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
            if ($LASTEXITCODE -ne 0) { throw "latexmk failed" }
        }
        finally {
            Pop-Location
        }
    }

    # Report overfull boxes so layout regressions are visible.
    $log = Join-Path $tmlrDir "main.log"
    if (Test-Path $log) {
        $overfull = Select-String -Path $log -Pattern "Overfull \\hbox \(([0-9.]+)pt"
        $count = @($overfull).Count
        if ($count -gt 0) {
            $max = ($overfull | ForEach-Object {
                [double]$_.Matches[0].Groups[1].Value
            } | Measure-Object -Maximum).Maximum
            Write-Warning ("Overfull hboxes: {0} (max {1}pt)" -f $count, $max)
        }
        else {
            Write-Host "No overfull hboxes." -ForegroundColor Green
        }
    }
}

# --------------------------------------------------------------------------
# 5) Optional: rebuild the root checksum manifest (packaging)
# --------------------------------------------------------------------------
if ($RebuildRootChecksums) {
    Invoke-Step "Rebuild ANONYMIZED_SHA256SUMS" {
        Push-Location $supplementRoot
        try {
            $root = (Get-Location).Path
            $lines = Get-ChildItem -Recurse -File |
                Where-Object {
                    $_.Name -ne "ANONYMIZED_SHA256SUMS" -and
                    $_.FullName -notmatch "[\\/]\.git[\\/]"
                } |
                ForEach-Object {
                    $rel = $_.FullName.Substring($root.Length + 1).Replace("\", "/")
                    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
                    "{0}  {1}" -f $hash, $rel
                } |
                Sort-Object { ($_ -split '  ', 2)[1] }
            Set-Content -Path "ANONYMIZED_SHA256SUMS" -Value $lines -Encoding ascii
            Write-Host ("Wrote {0} entries" -f @($lines).Count)
        }
        finally {
            Pop-Location
        }
    }
}

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "==> Figure inventory" -ForegroundColor Cyan
$paperFigures = Join-Path $tmlrDir "figures"
Get-ChildItem $paperFigures -File -Filter "*.pdf" |
    Sort-Object Name |
    ForEach-Object {
        "{0,-42} {1,8:N0} B  {2}" -f $_.Name, $_.Length,
            $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm")
    }

Write-Host ""
Write-Host "Done." -ForegroundColor Green
