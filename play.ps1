[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$ProjectName,

    [Parameter(Position = 1)]
    [string]$MediaPlayer
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
$projectOutput = Join-Path $repoRoot "out\$ProjectName"

if (-not (Test-Path -LiteralPath $projectOutput -PathType Container)) {
    throw "No output directory exists for project '$ProjectName'."
}

$latest = Get-ChildItem -LiteralPath $projectOutput -Directory | Sort-Object Name | Select-Object -Last 1
if (-not $latest) { throw "No timestamped output exists for project '$ProjectName'." }

$video = @(
    (Join-Path $latest.FullName 'Video.mkv'),
    (Join-Path $latest.FullName 'Video.mp4')
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1

if (-not $video -or (Get-Item -LiteralPath $video).Length -lt 1024) {
    throw "The newest output video is missing or smaller than 1 KiB: $($latest.FullName)"
}

function Resolve-Player([string]$RequestedPlayer) {
    if ($RequestedPlayer) {
        $command = Get-Command $RequestedPlayer -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
        if (Test-Path -LiteralPath $RequestedPlayer -PathType Leaf) { return $RequestedPlayer }
        throw "Media player '$RequestedPlayer' was not found."
    }

    # ffplay can hang when PowerShell is running without an interactive WASAPI
    # endpoint (for example through SSH). Keep it available when explicitly
    # requested, but prefer the Windows file association as the automatic
    # fallback because native players handle endpoint changes more gracefully.
    foreach ($candidate in 'vlc', 'mpv') {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
    }
    foreach ($candidate in @(
        "$env:ProgramFiles\VideoLAN\VLC\vlc.exe",
        "${env:ProgramFiles(x86)}\VideoLAN\VLC\vlc.exe"
    )) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $candidate }
    }
    return $null
}

$player = Resolve-Player $MediaPlayer
if ($player) {
    Write-Host "play.ps1: Playing $video with $player"
    Start-Process -FilePath $player -ArgumentList @($video) | Out-Null
} else {
    Write-Host "play.ps1: No supported player found; opening the video with its Windows file association."
    Start-Process -FilePath $video | Out-Null
}
