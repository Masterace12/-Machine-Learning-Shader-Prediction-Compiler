# Improved GitHub Download Script for Shader Predictive Compiler (PowerShell)
# Enhanced with better error handling, progress indication, and reliability

param(
    [switch]$Debug,
    [switch]$NoCache,
    [string]$Branch = "main",
    [switch]$Help
)

# Configuration
$GitHubUser = "Masterace12"
$GitHubRepo = "shader-prediction-compilation"
$RepoUrl = "https://github.com/$GitHubUser/$GitHubRepo"
$CacheDir = "$env:LOCALAPPDATA\shader-predict-compile\cache"
$TempDir = "$env:TEMP\shader-install-$(Get-Random)"
$MaxRetries = 3
$DownloadTimeout = 300 # seconds

# Show help if requested
if ($Help) {
    Write-Host @"
Usage: .\improved-github-download.ps1 [options]

Options:
  -Debug        Enable debug output
  -NoCache      Clear cache and force fresh download
  -Branch NAME  Use specific branch (default: main)
  -Help         Show this help message

Examples:
  .\improved-github-download.ps1
  .\improved-github-download.ps1 -Debug -NoCache
  .\improved-github-download.ps1 -Branch development
"@
    exit 0
}

# Color functions
function Write-Info { param($Message) Write-Host "[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline; Write-Host "[INFO]" -ForegroundColor Blue -NoNewline; Write-Host " $Message" }
function Write-Success { param($Message) Write-Host "[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline; Write-Host "[✓]" -ForegroundColor Green -NoNewline; Write-Host " $Message" }
function Write-Warning { param($Message) Write-Host "[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline; Write-Host "[!]" -ForegroundColor Yellow -NoNewline; Write-Host " $Message" }
function Write-Error { param($Message) Write-Host "[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline; Write-Host "[✗]" -ForegroundColor Red -NoNewline; Write-Host " $Message" }
function Write-Debug { param($Message) if ($Debug) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] " -NoNewline; Write-Host "[DEBUG]" -ForegroundColor Magenta -NoNewline; Write-Host " $Message" } }

# Cleanup function
function Cleanup {
    if (Test-Path $TempDir) {
        Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Register cleanup
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Cleanup } | Out-Null
trap { Cleanup; break }

# Progress bar function
function Show-Progress {
    param(
        [int]$Current,
        [int]$Total,
        [string]$Activity = "Processing"
    )
    
    $percent = if ($Total -gt 0) { [math]::Round(($Current / $Total) * 100) } else { 0 }
    Write-Progress -Activity $Activity -Status "$percent% Complete" -PercentComplete $percent
}

# Check system requirements
function Test-Requirements {
    Write-Info "Checking system requirements..."
    
    $missing = @()
    $optional = @()
    
    # Check PowerShell version
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        $missing += "PowerShell 5.0 or higher"
    }
    
    # Check for .NET Framework
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
    } catch {
        $missing += ".NET Framework 4.5 or higher"
    }
    
    # Check for optional tools
    $optionalTools = @('git', '7z', 'tar')
    foreach ($tool in $optionalTools) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            $optional += $tool
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Error "Missing required components: $($missing -join ', ')"
        return $false
    }
    
    if ($optional.Count -gt 0) {
        Write-Warning "Optional tools missing: $($optional -join ', ')"
        Write-Info "Consider installing for better functionality"
    }
    
    Write-Success "All required components available"
    return $true
}

# Check cache
function Test-Cache {
    param([string]$Branch)
    
    $cacheFile = Join-Path $CacheDir "repo-$Branch.zip"
    $cacheInfo = Join-Path $CacheDir "repo-$Branch.info"
    
    if ((Test-Path $cacheFile) -and (Test-Path $cacheInfo)) {
        $cacheDate = Get-Content $cacheInfo -ErrorAction SilentlyContinue
        if ($cacheDate) {
            $cacheAge = (Get-Date) - [DateTime]::Parse($cacheDate)
            if ($cacheAge.TotalHours -lt 24) {
                Write-Info "Found recent cached download ($([int]$cacheAge.TotalHours) hours old)"
                return $cacheFile
            } else {
                Write-Info "Cache expired, will download fresh copy"
                Remove-Item $cacheFile, $cacheInfo -Force -ErrorAction SilentlyContinue
            }
        }
    }
    
    return $null
}

# Download with progress
function Download-WithProgress {
    param(
        [string]$Url,
        [string]$OutFile,
        [int]$TimeoutSec = 300
    )
    
    Write-Info "Downloading from $Url..."
    
    try {
        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("User-Agent", "PowerShell")
        
        # Setup progress event
        $progressActivity = "Downloading $([System.IO.Path]::GetFileName($OutFile))"
        Register-ObjectEvent -InputObject $webClient -EventName DownloadProgressChanged -Action {
            $percent = $Event.SourceEventArgs.ProgressPercentage
            Write-Progress -Activity $progressActivity -Status "$percent% Complete" -PercentComplete $percent
        } | Out-Null
        
        # Setup completion event
        $downloadComplete = $false
        Register-ObjectEvent -InputObject $webClient -EventName DownloadFileCompleted -Action {
            $script:downloadComplete = $true
        } | Out-Null
        
        # Start async download
        $webClient.DownloadFileAsync($Url, $OutFile)
        
        # Wait for completion or timeout
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        while (-not $downloadComplete -and $stopwatch.Elapsed.TotalSeconds -lt $TimeoutSec) {
            Start-Sleep -Milliseconds 100
        }
        
        $webClient.CancelAsync()
        $webClient.Dispose()
        
        if (-not $downloadComplete) {
            throw "Download timeout after $TimeoutSec seconds"
        }
        
        Write-Progress -Activity $progressActivity -Completed
        return $true
    }
    catch {
        Write-Error "Download failed: $_"
        return $false
    }
    finally {
        Get-EventSubscriber | Where-Object { $_.SourceObject -eq $webClient } | Unregister-Event
    }
}

# Alternative download method using Invoke-WebRequest
function Download-Simple {
    param(
        [string]$Url,
        [string]$OutFile
    )
    
    try {
        Write-Info "Downloading (simple method)..."
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -TimeoutSec 300
        $ProgressPreference = 'Continue'
        return $true
    }
    catch {
        Write-Error "Simple download failed: $_"
        return $false
    }
}

# Verify download
function Test-Download {
    param([string]$File)
    
    if (-not (Test-Path $File)) {
        Write-Error "Download file not found: $File"
        return $false
    }
    
    $fileInfo = Get-Item $File
    if ($fileInfo.Length -lt 1000) {
        Write-Error "Downloaded file too small ($($fileInfo.Length) bytes)"
        return $false
    }
    
    # Check if it's a valid archive
    $validExtensions = @('.zip', '.tar.gz', '.tar', '.7z')
    $isValid = $false
    foreach ($ext in $validExtensions) {
        if ($File.EndsWith($ext)) {
            $isValid = $true
            break
        }
    }
    
    if (-not $isValid) {
        # Try to check file content
        $bytes = [System.IO.File]::ReadAllBytes($File) | Select-Object -First 4
        $signature = [System.BitConverter]::ToString($bytes)
        
        # Check for common archive signatures
        $archiveSignatures = @(
            "50-4B-03-04", # ZIP
            "50-4B-05-06", # ZIP
            "50-4B-07-08", # ZIP
            "1F-8B-08",    # GZIP
            "37-7A-BC-AF"  # 7Z
        )
        
        foreach ($sig in $archiveSignatures) {
            if ($signature.StartsWith($sig)) {
                $isValid = $true
                break
            }
        }
    }
    
    if ($isValid) {
        Write-Success "Valid archive detected (Size: $([math]::Round($fileInfo.Length / 1MB, 2)) MB)"
        
        # Calculate hash
        $hash = Get-FileHash $File -Algorithm SHA256
        Write-Info "SHA256: $($hash.Hash.Substring(0, 16))..."
        
        return $true
    } else {
        Write-Error "Invalid archive format"
        return $false
    }
}

# Download repository
function Download-Repository {
    Write-Info "Preparing to download repository..."
    
    # Clear cache if requested
    if ($NoCache -and (Test-Path $CacheDir)) {
        Remove-Item -Path $CacheDir -Recurse -Force
    }
    
    # Check cache first
    $cachedFile = Test-Cache -Branch $Branch
    if ($cachedFile) {
        Write-Success "Using cached download"
        return $cachedFile
    }
    
    # Create directories
    New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    
    $outputFile = Join-Path $TempDir "repo.zip"
    $success = $false
    
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        Write-Info "Download attempt $attempt of $MaxRetries"
        
        # Method 1: Git clone (if available)
        if ((Get-Command git -ErrorAction SilentlyContinue) -and $attempt -eq 1) {
            Write-Info "Trying git clone method..."
            $gitDir = Join-Path $TempDir "repo-git"
            $gitResult = & git clone --depth 1 --branch $Branch "$RepoUrl.git" $gitDir 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                Write-Info "Creating archive from git clone..."
                
                if (Get-Command tar -ErrorAction SilentlyContinue) {
                    Push-Location $gitDir
                    & tar -czf "$outputFile.tar.gz" .
                    Pop-Location
                    $outputFile = "$outputFile.tar.gz"
                } else {
                    # Use .NET compression
                    Add-Type -AssemblyName System.IO.Compression.FileSystem
                    [System.IO.Compression.ZipFile]::CreateFromDirectory($gitDir, $outputFile)
                }
                
                if (Test-Download $outputFile) {
                    $success = $true
                    break
                }
            }
        }
        
        # Method 2: Direct ZIP download
        $zipUrl = "$RepoUrl/archive/refs/heads/$Branch.zip"
        
        # Try advanced download first
        if (Download-WithProgress -Url $zipUrl -OutFile $outputFile) {
            if (Test-Download $outputFile) {
                $success = $true
                break
            }
        }
        
        # Try simple download as fallback
        if (Download-Simple -Url $zipUrl -OutFile $outputFile) {
            if (Test-Download $outputFile) {
                $success = $true
                break
            }
        }
        
        if ($attempt -lt $MaxRetries) {
            Write-Warning "Download failed, retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }
    
    if (-not $success) {
        Write-Error "All download attempts failed"
        return $null
    }
    
    # Cache the successful download
    $cacheFile = Join-Path $CacheDir "repo-$Branch.zip"
    Copy-Item $outputFile $cacheFile -Force
    (Get-Date).ToString() | Set-Content (Join-Path $CacheDir "repo-$Branch.info")
    
    Write-Success "Repository downloaded successfully"
    return $outputFile
}

# Extract archive
function Extract-Archive {
    param(
        [string]$Archive,
        [string]$DestDir
    )
    
    Write-Info "Extracting archive..."
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    
    try {
        # Try different extraction methods based on file type and available tools
        $extracted = $false
        
        if ($Archive.EndsWith('.zip')) {
            # Use .NET extraction
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            [System.IO.Compression.ZipFile]::ExtractToDirectory($Archive, $DestDir)
            $extracted = $true
        }
        elseif ($Archive.EndsWith('.tar.gz') -or $Archive.EndsWith('.tgz')) {
            if (Get-Command tar -ErrorAction SilentlyContinue) {
                & tar -xzf $Archive -C $DestDir
                $extracted = $true
            }
            elseif (Get-Command 7z -ErrorAction SilentlyContinue) {
                & 7z x $Archive -o"$DestDir" -y | Out-Null
                $tarFile = Get-ChildItem $DestDir -Filter "*.tar" | Select-Object -First 1
                if ($tarFile) {
                    & 7z x $tarFile.FullName -o"$DestDir" -y | Out-Null
                    Remove-Item $tarFile.FullName
                }
                $extracted = $true
            }
        }
        
        if (-not $extracted) {
            throw "No suitable extraction method found for $Archive"
        }
        
        # Find the actual project directory
        $projectDir = Get-ChildItem $DestDir -Recurse -Directory | 
                      Where-Object { $_.Name -eq "shader-predict-compile" } | 
                      Select-Object -First 1
        
        if (-not $projectDir) {
            # Try to find by looking for key files
            $installFile = Get-ChildItem $DestDir -Recurse -File | 
                          Where-Object { $_.Name -eq "install" -or $_.Name -eq "INSTALL.sh" } | 
                          Select-Object -First 1
            
            if ($installFile) {
                $projectDir = $installFile.Directory
            }
        }
        
        if (-not $projectDir) {
            Write-Warning "Could not find shader-predict-compile directory, using root"
            $projectDir = Get-Item $DestDir
        }
        
        Write-Success "Extracted to: $($projectDir.FullName)"
        return $projectDir.FullName
    }
    catch {
        Write-Error "Extraction failed: $_"
        return $null
    }
}

# Fix GitHub issues (Windows-specific)
function Repair-GitHubIssues {
    param([string]$Dir)
    
    Write-Info "Fixing GitHub download issues..."
    Push-Location $Dir
    
    try {
        # Find all relevant files
        $files = Get-ChildItem -Recurse -Include "*.sh", "*.py", "install*" -File
        $total = $files.Count
        $current = 0
        
        Write-Info "Processing $total files..."
        
        foreach ($file in $files) {
            $current++
            Show-Progress -Current $current -Total $total -Activity "Fixing files"
            
            # Read file content
            $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
            if ($content) {
                # Fix line endings (CRLF to LF)
                $fixedContent = $content -replace "`r`n", "`n"
                
                # Fix shebang if present
                if ($fixedContent.StartsWith("#!")) {
                    $lines = $fixedContent -split "`n"
                    $shebang = $lines[0]
                    
                    # Fix common shebang issues
                    $shebang = $shebang -replace '^#![\s]*/', '#!/'
                    $shebang = $shebang -replace '^#!/bin/bash.*$', '#!/bin/bash'
                    $shebang = $shebang -replace '^#!/usr/bin/env python.*$', '#!/usr/bin/env python3'
                    
                    $lines[0] = $shebang
                    $fixedContent = $lines -join "`n"
                }
                
                # Write back with UTF8 encoding and Unix line endings
                $utf8NoBom = New-Object System.Text.UTF8Encoding $false
                [System.IO.File]::WriteAllText($file.FullName, $fixedContent, $utf8NoBom)
                
                # Mark as executable (set attribute)
                if ($file.Extension -eq '.sh' -or $file.Extension -eq '.py' -or $file.BaseName -eq 'install') {
                    $file.Attributes = $file.Attributes -band (-bnot [System.IO.FileAttributes]::ReadOnly)
                }
            }
        }
        
        Write-Progress -Activity "Fixing files" -Completed
        
        # Create necessary directories
        @('logs', 'cache') | ForEach-Object {
            New-Item -ItemType Directory -Force -Path $_ | Out-Null
        }
        
        Write-Success "GitHub issues fixed"
    }
    finally {
        Pop-Location
    }
}

# Main function
function Main {
    Write-Host "`n" -NoNewline
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
    Write-Host "     🚀 Shader Predictive Compiler - Enhanced Installer 🚀" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Blue
    Write-Host ""
    
    # Check requirements
    if (-not (Test-Requirements)) {
        exit 1
    }
    
    # Download repository
    $archiveFile = Download-Repository
    if (-not $archiveFile) {
        Write-Error "Failed to download repository"
        exit 1
    }
    
    # Extract archive
    $projectDir = Extract-Archive -Archive $archiveFile -DestDir (Join-Path $TempDir "extracted")
    if (-not $projectDir) {
        Write-Error "Failed to extract archive"
        exit 1
    }
    
    # Fix GitHub issues
    Repair-GitHubIssues -Dir $projectDir
    
    Write-Host "`n" -NoNewline
    Write-Success "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Success "Download and preparation completed successfully! 🎉"
    Write-Success "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host ""
    Write-Info "Project prepared at: $projectDir"
    Write-Info "Next steps:"
    Write-Host "  1. Copy to Linux/Steam Deck: " -NoNewline
    Write-Host "$projectDir" -ForegroundColor Yellow
    Write-Host "  2. Run on Linux: " -NoNewline
    Write-Host "cd shader-predict-compile && ./install" -ForegroundColor Yellow
    Write-Host ""
    
    # Offer to open the directory
    $openDir = Read-Host "Open project directory in Explorer? (Y/N)"
    if ($openDir -eq 'Y' -or $openDir -eq 'y') {
        Start-Process explorer.exe -ArgumentList $projectDir
    }
}

# Run main function
try {
    Main
}
finally {
    Cleanup
}