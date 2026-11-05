# Edge simulation runner (Windows PowerShell). See SETUP.md.
#
#   .\run_all.ps1                 build + all x86 steps (about 2-3 hours)
#   .\run_all.ps1 -Arm            also the ARM64 staging-parity steps (adds 1-3 hours)
#   .\run_all.ps1 -Quick          smoke test: 2 nights per step (about 20 minutes after the build)
#   .\run_all.ps1 -Only check     one step: build, check, staging, reports, endtoend, arm, summary, pack
#
# Every step can be re-run; finished nights are not recomputed.
param(
    [switch]$Arm,
    [switch]$Quick,
    [string]$Only = ""
)
$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
$Out = Join-Path $Here "out"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Start-Transcript -Path (Join-Path $Out "run_log.txt") -Append | Out-Null

$Img = "sleep-sim:agent"
$ImgArm = "sleep-sim:arm"
$N = @()
if ($Quick) { $N = @("--nights", "2") }

# Pi-like limits: 4 CPUs, 16 GB, no swap beyond that, NO NETWORK. The project and data are read-only.
$Mounts = @(
    "-v", "${Here}\repo:/repo:ro",
    "-v", "${Here}\sim:/sim:ro",
    "-v", "${Here}\data:/data:ro",
    "-v", "${Out}:/out"
)
$Limits = @("--rm", "--network", "none", "--cpus", "4", "--memory", "16g", "--memory-swap", "16g")

function Step($name, [scriptblock]$body) {
    if ($Only -and $Only -ne $name) { return }
    Write-Host ""
    Write-Host "=== $name  ($(Get-Date -Format 'HH:mm:ss')) ===" -ForegroundColor Cyan
    & $body
    if ($LASTEXITCODE -ne 0) { Write-Host "Step '$name' exited with code $LASTEXITCODE (see out\run_log.txt)" -ForegroundColor Yellow }
}

function Sim([string]$image, [string[]]$extra, [string[]]$cmd) {
    $a = @("run") + $Limits + $extra + $Mounts + @($image, "/sim/measure.py") + $cmd
    & docker @a
}

docker info *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "Docker is not running. Start Docker Desktop, wait until it says 'Engine running', then re-run." -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }

# 1. Build the x86 image (needs internet; about 15-30 minutes the first time, cached afterwards)
Step "build" { docker build --target agent -t $Img (Join-Path $Here "sim") }

# 2. Environment: release pins, network really off, no secrets visible
Step "check" { Sim $Img @() @("check") }

# 3. Staging on all 31 raw dev recordings (no report model): speed, memory, and Linux-vs-reference parity
Step "staging" { Sim $Img @() (@("pathb", "--backend", "oracle") + $N) }

# 4. Report model on the 31 dev evidence packets: speed, memory, outcome, identical text to the reference?
Step "reports" { Sim $Img @() (@("patha") + $N) }

# 5. Full system on 3 raw recordings: recording -> staging -> packet -> report -> verifier -> delivery
Step "endtoend" { Sim $Img @() @("pathb", "--backend", "llama", "--nights", "3") }

# 6. Optional: ARM64 (the Pi's processor type) under emulation: staging parity only
if ($Arm -or $Only -eq "arm") {
    Step "arm" {
        docker run --privileged --rm tonistiigi/binfmt --install arm64
        docker buildx build --platform linux/arm64 --target staging -t $ImgArm --load (Join-Path $Here "sim")
        $nights = if ($Quick) { "2" } else { "8" }
        Sim $ImgArm @("--platform", "linux/arm64") @("pathb", "--backend", "oracle", "--nights", $nights)
    }
}

# 7. Summary and the zip to send back
Step "summary" { Sim $Img @() @("summary") }
Step "pack" {
    $zip = Join-Path $Here ("results_" + (Get-Date -Format "yyyyMMdd_HHmm") + ".zip")
    $items = Get-ChildItem $Out -File | Select-Object -ExpandProperty FullName
    $items += Join-Path $Out "logs"
    Stop-Transcript | Out-Null                        # the transcript holds run_log.txt open (run 1: zip failed)
    Compress-Archive -Path $items -DestinationPath $zip -Force
    Start-Transcript -Path (Join-Path $Out "run_log.txt") -Append | Out-Null
    Write-Host "Send this file back: $zip" -ForegroundColor Green
}
Stop-Transcript | Out-Null
