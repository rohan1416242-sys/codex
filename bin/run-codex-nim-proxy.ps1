# codex-nim-proxy launcher for Windows PowerShell
param(
    [int]$Port = 8765,
    [int]$Rpm = 40,
    [string]$UpstreamBaseUrl,
    [string]$ApiKey,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Binary = Join-Path $ScriptDir "codex-nim-proxy.exe"

if (-not (Test-Path $Binary)) {
    Write-Error "ERROR: codex-nim-proxy.exe not found at $Binary"
    exit 1
}

$apiKeyPresent = -not [string]::IsNullOrEmpty($env:NVIDIA_API_KEY) -or -not [string]::IsNullOrEmpty($ApiKey)
if (-not $apiKeyPresent) {
    Write-Warning "NVIDIA_API_KEY is not set. Get one at https://developer.nvidia.com -> Build -> NVIDIA NIM"
}

$args = @("--port", $Port, "--rpm", $Rpm)
if ($UpstreamBaseUrl) { $args += @("--upstream-base-url", $UpstreamBaseUrl) }
if ($ApiKey)          { $args += @("--api-key", $ApiKey) }
if ($Verbose)         { $args += "--verbose" }

& $Binary @args
