$ErrorActionPreference = "Stop"

$runningGame = Get-Process | Where-Object { $_.ProcessName -in @("StardewModdingAPI", "Stardew Valley") }
if ($runningGame) {
    Write-Error "Close Stardew Valley/SMAPI first. The v8 bridge DLL cannot be activated inside a game process that is already running."
}

dotnet build .\src\JunimoKartRLBridge\JunimoKartRLBridge.csproj -c Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Bridge v8 built and deployed. Open Stardew Valley through SMAPI again before training."
