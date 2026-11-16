<#
Sync local lab sources to a NEW GitHub branch, without touching original F: project.
Uses a separate checkout, whitelisted file types, and no automatic merge.
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\sync_local.ps1 -IncludeResults
#>
param(
  [string]$LocalSource = 'F:\0AI\lifelong_learning_lab_01',
  [string]$CheckoutDir = 'F:\0AI\lifelong-learning-lab-sync',
  [switch]$IncludeResults,
  [switch]$NoPush
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $LocalSource -PathType Container)) {
  throw "Local project not found: $LocalSource"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw 'Git executable not found.'
}
$remote = 'https://github.com/yixuanqi/lifelong-learning-lab.git'
if (-not (Test-Path -LiteralPath $CheckoutDir)) {
  & git clone $remote $CheckoutDir
  if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
} else {
  if (-not (Test-Path -LiteralPath (Join-Path $CheckoutDir '.git'))) {
    throw 'CheckoutDir exists but is not a git checkout.'
  }
  $changes = & git -C $CheckoutDir status --porcelain
  if ($LASTEXITCODE -ne 0 -or $changes) {
    throw 'CheckoutDir contains changes. Resolve them before syncing.'
  }
  & git -C $CheckoutDir switch main
  if ($LASTEXITCODE -ne 0) { throw 'git switch main failed' }
  & git -C $CheckoutDir pull --ff-only origin main
  if ($LASTEXITCODE -ne 0) { throw 'git pull failed' }
}
$branch = 'sync/local-labs-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
& git -C $CheckoutDir switch -c $branch
if ($LASTEXITCODE -ne 0) { throw 'git switch -c failed' }
$copied = 0

function Copy-One([string]$sourceRelative, [string]$targetRelative) {
  $sourcePath = Join-Path $LocalSource $sourceRelative
  if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    Write-Host "SKIP missing: $sourceRelative"
    return
  }
  $targetPath = Join-Path $CheckoutDir $targetRelative
  New-Item -ItemType Directory -Force -Path (Split-Path $targetPath -Parent) | Out-Null
  Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
  $script:copied++
  Write-Host "COPY $sourceRelative => $targetRelative"
}

Copy-One 'train.py'         'labs/lab01/train.py'
Copy-One 'smoke_test.py'    'labs/lab01/smoke_test.py'
Copy-One 'requirements.txt' 'labs/lab01/requirements.txt'
Copy-One 'README.md'        'labs/lab01/README_LOCAL_ORIGINAL.md'
Copy-One 'exp02.py'         'labs/lab02/exp02.py'
Copy-One 'exp02-report.html' 'labs/lab02/exp02-report.html'
Copy-One 'lab03.py'         'labs/lab03/lab03.py'
Copy-One 'lab03-report.html' 'labs/lab03/lab03-report.html'

function Copy-SelectedResults([string]$sourceRel, [string]$destinationRel) {
  $root = Join-Path $LocalSource $sourceRel
  if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    Write-Host "SKIP missing directory: $sourceRel"
    return
  }
  $root = (Get-Item -LiteralPath $root).FullName
  $extensions = @('.csv', '.json', '.png', '.md')
  $files = Get-ChildItem -LiteralPath $root -File -Recurse |
    Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() }
  foreach ($file in $files) {
    if ($file.Length -gt 15MB) {
      Write-Host "SKIP >15MB: $($file.FullName)"
      continue
    }
    $relative = $file.FullName.Substring($root.Length).TrimStart('\', '/')
    $targetRelative = ($destinationRel + '/' + $relative).Replace('\', '/')
    $targetPath = Join-Path $CheckoutDir $targetRelative
    New-Item -ItemType Directory -Force -Path (Split-Path $targetPath -Parent) | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $targetPath -Force
    $script:copied++
  }
}

if ($IncludeResults) {
  Copy-SelectedResults 'results'       'labs/lab01/curated/local-results'
  Copy-SelectedResults 'results_exp02' 'labs/lab02/curated'
  Copy-SelectedResults 'results_lab03' 'labs/lab03/curated'
}
if ($copied -eq 0) { throw 'Nothing found to copy. Verify LocalSource.' }
Write-Host "Copied $copied files. The GitHub repository is PUBLIC; review before pushing."
& git -C $CheckoutDir add -- labs/lab01 labs/lab02 labs/lab03
if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
& git -C $CheckoutDir diff --cached --stat
& git -C $CheckoutDir diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
  Write-Host 'No changes to commit.'
  exit 0
}
& git -C $CheckoutDir commit -m 'sync: import local experiment code and curated results'
if ($LASTEXITCODE -ne 0) { throw 'git commit failed (check git identity)' }
if ($NoPush) {
  Write-Host "Local branch ready: $branch (not pushed)"
} else {
  Write-Host 'REVIEW BEFORE PUSHING. GitHub repo is PUBLIC.'
  & git -C $CheckoutDir status --short
  $answer = Read-Host 'Type PUSH to publish this branch to PUBLIC GitHub'
  if ($answer -cne 'PUSH') {
    Write-Host "Not pushed. Your local sync branch is: $branch"
    exit 0
  }
  & git -C $CheckoutDir push -u origin $branch
  if ($LASTEXITCODE -ne 0) { throw 'git push failed (check GitHub authentication)' }
  Write-Host "Pushed branch: https://github.com/yixuanqi/lifelong-learning-lab/tree/$branch"
  Write-Host 'Branch is not merged into main. Send the link for PR review.'
}
