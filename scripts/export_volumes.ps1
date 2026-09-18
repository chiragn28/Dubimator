<#
.SYNOPSIS
  Copies the two Docker volumes a server needs (Postgres data and the MLflow model store)
  plus .env to the server, ready for deploy/server_setup.sh.

.EXAMPLE
  .\scripts\export_volumes.ps1 -Server root@203.0.113.10

  Run from the repository root with Docker Desktop running. Stops the local stack first so the
  Postgres files are consistent; start it again afterwards with `docker compose up -d`.
#>
param(
    [Parameter(Mandatory = $true)] [string] $Server,
    [string] $RemoteDir = "/opt/dubimator"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".env")) { throw "No .env in $root - nothing to ship." }

$out = Join-Path $root "deploy\out"
New-Item -ItemType Directory -Force $out | Out-Null

Write-Host "Stopping the local stack so the volumes are consistent..."
docker compose stop

foreach ($v in "postgres_data", "mlflow_data") {
    Write-Host "Exporting dubimator_$v..."
    docker run --rm -v "dubimator_${v}:/v:ro" -v "${out}:/b" alpine tar czf "/b/$v.tgz" -C /v .
    if ($LASTEXITCODE -ne 0) { throw "export of dubimator_$v failed" }
}

Get-ChildItem $out | Format-Table Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB) } }

Write-Host "Copying to ${Server}:${RemoteDir} ..."
ssh $Server "mkdir -p $RemoteDir"
scp (Join-Path $out "postgres_data.tgz") (Join-Path $out "mlflow_data.tgz") ".env" "${Server}:${RemoteDir}/"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

Write-Host ""
Write-Host "Done. Next, on the server:"
Write-Host "  ssh $Server"
Write-Host "  cd $RemoteDir && git pull && bash deploy/server_setup.sh api.example.com demo.example.com"
Write-Host "Then edit $RemoteDir/.env on the server: new POSTGRES_PASSWORD and fresh API keys (see deploy/README.md)."
