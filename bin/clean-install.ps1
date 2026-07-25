# Clean install script for codex-nim-proxy on Windows.
# Run this in PowerShell to NUKE all old config and set up fresh.
#
# Usage:
#   .\clean-install.ps1
#   .\clean-install.ps1 -ApiKey "nvapi-..."
#   .\clean-install.ps1 -BackendModel "thinkingmachines/inkling"

param(
    [string]$ApiKey = "",
    [string]$BackendModel = "thinkingmachines/inkling",
    [string]$InstallDir = "C:\Tools\codex-proxy"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  codex-nim-proxy - Clean Install" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill any running proxy
Write-Host "[1/7] Stopping any running proxy..." -ForegroundColor Yellow
Get-Process -Name "codex-nim-proxy" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1
Write-Host "  Done." -ForegroundColor Green

# Step 2: Download fresh binaries
Write-Host "[2/7] Downloading fresh binaries..." -ForegroundColor Yellow
New-Item -Path $InstallDir -ItemType Directory -Force | Out-Null
$exeUrl = "https://raw.githubusercontent.com/rohan1416242-sys/codex/main/bin/codex-nim-proxy.exe"
$ps1Url = "https://raw.githubusercontent.com/rohan1416242-sys/codex/main/bin/run-codex-nim-proxy.ps1"
Invoke-WebRequest -Uri $exeUrl -OutFile "$InstallDir\codex-nim-proxy.exe"
Invoke-WebRequest -Uri $ps1Url -OutFile "$InstallDir\run-codex-nim-proxy.ps1"
Unblock-File "$InstallDir\codex-nim-proxy.exe"
Write-Host "  Downloaded to $InstallDir" -ForegroundColor Green

# Step 3: Set API key
Write-Host "[3/7] Setting NVIDIA_API_KEY..." -ForegroundColor Yellow
if ($ApiKey -ne "") {
    [System.Environment]::SetEnvironmentVariable("NVIDIA_API_KEY", $ApiKey, "User")
    $env:NVIDIA_API_KEY = $ApiKey
    Write-Host "  Set to: $ApiKey" -ForegroundColor Green
} else {
    $existing = [System.Environment]::GetEnvironmentVariable("NVIDIA_API_KEY", "User")
    if ($existing) {
        $env:NVIDIA_API_KEY = $existing
        Write-Host "  Already set (reusing existing key)." -ForegroundColor Green
    } else {
        Write-Host "  ERROR: No API key provided." -ForegroundColor Red
        Write-Host "  Run: .\clean-install.ps1 -ApiKey 'nvapi-...'" -ForegroundColor Red
        exit 1
    }
}

# Step 4: Nuke ALL old codex config + auth
Write-Host "[4/7] Nuking old codex config + auth..." -ForegroundColor Yellow
$codexDir = "$env:USERPROFILE\.codex"
Remove-Item "$codexDir\auth.json" -Force -ErrorAction SilentlyContinue
Remove-Item "$codexDir\config.toml" -Force -ErrorAction SilentlyContinue
Remove-Item "$codexDir\config.lock.json" -Force -ErrorAction SilentlyContinue
Remove-Item "$codexDir\history.jsonl" -Force -ErrorAction SilentlyContinue
Write-Host "  Deleted auth.json, config.toml, config.lock.json, history.jsonl" -ForegroundColor Green

# Step 5: Write fresh config.toml - INVISIBLE BRIDGE config
Write-Host "[5/7] Writing fresh config.toml (invisible bridge)..." -ForegroundColor Yellow
$configLines = @(
    '# codex-nim-proxy invisible bridge config',
    '# All requests silently routed to local proxy, then to NVIDIA NIM.',
    '# Codex UI shows whatever model it picks (e.g. gpt-5.6-sol) - the proxy',
    '# overrides it invisibly with the configured backend NIM model.',
    'model_provider = "nvidia-nim"',
    '',
    '[model_providers.nvidia-nim]',
    'name = "NVIDIA NIM"',
    'base_url = "http://localhost:8765/v1"',
    'env_key = "NVIDIA_API_KEY"',
    'wire_api = "responses"',
    'requires_openai_auth = false',
    'supports_websockets = false',
    ''
)
$config = $configLines -join "`r`n"
[System.IO.File]::WriteAllText("$codexDir\config.toml", $config, [System.Text.Encoding]::ASCII)
Write-Host "  Written to $codexDir\config.toml" -ForegroundColor Green

# Step 6: Install codex CLI (if not already installed)
Write-Host "[6/7] Ensuring codex CLI is installed..." -ForegroundColor Yellow
$codexCmd = Get-Command codex -ErrorAction SilentlyContinue
if (-not $codexCmd) {
    Write-Host "  Installing @openai/codex via npm..." -ForegroundColor Yellow
    npm install -g @openai/codex
} else {
    Write-Host "  codex already installed: $($codexCmd.Source)" -ForegroundColor Green
}

# Step 7: Verify
Write-Host "[7/7] Verifying..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Config file contents:" -ForegroundColor Cyan
Get-Content "$codexDir\config.toml"
Write-Host ""
Write-Host "API key:" -ForegroundColor Cyan
Write-Host "  $env:NVIDIA_API_KEY"
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  INSTALL COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Backend model: $BackendModel" -ForegroundColor Cyan
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Start the proxy (leave this window open):" -ForegroundColor White
Write-Host "       cd $InstallDir" -ForegroundColor White
if ($BackendModel -ne "thinkingmachines/inkling") {
    Write-Host "       .\run-codex-nim-proxy.ps1 -BackendModel '$BackendModel'" -ForegroundColor White
} else {
    Write-Host "       .\run-codex-nim-proxy.ps1" -ForegroundColor White
}
Write-Host ""
Write-Host "  2. Open a NEW PowerShell window and run:" -ForegroundColor White
Write-Host "       codex" -ForegroundColor White
Write-Host ""
Write-Host "  The proxy silently routes all requests to $BackendModel" -ForegroundColor Gray
Write-Host "  on NVIDIA NIM. Codex UI shows gpt-5.6-sol (or whatever" -ForegroundColor Gray
Write-Host "  you pick with /model) - completely invisible." -ForegroundColor Gray
Write-Host ""
