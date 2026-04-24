param(
    [string]$VenvPath = ".venv",
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$PipArgs
)

$ErrorActionPreference = "Stop"

$pipExe = Join-Path $PSScriptRoot "$VenvPath/Scripts/pip.exe"
if (-not (Test-Path $pipExe)) {
    Write-Error "venv pip not found: $pipExe"
}

function Invoke-Pip {
    param(
        [string[]]$PipArguments
    )
    & $pipExe @PipArguments
    return $LASTEXITCODE
}

$originalHttpProxy = $env:HTTP_PROXY
$originalHttpsProxy = $env:HTTPS_PROXY
$originalAllProxy = $env:ALL_PROXY
$originalNoProxy = $env:NO_PROXY

try {
    Write-Host "pip-smart: try install with current proxy"
    $proxyTryArgs = @("--retries", "1", "--timeout", "8") + $PipArgs
    $code = Invoke-Pip -PipArguments $proxyTryArgs
    if ($code -eq 0) {
        Write-Host "pip-smart: install succeeded with proxy"
        exit 0
    }

    Write-Warning "pip-smart: proxy install failed, retry direct in current process only"
    Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
    Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
    Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
    $env:NO_PROXY = "pypi.org,files.pythonhosted.org,pypi.tuna.tsinghua.edu.cn"

    $code = Invoke-Pip -PipArguments $PipArgs
    if ($code -eq 0) {
        Write-Host "pip-smart: direct retry succeeded"
        exit 0
    }

    exit $code
}
finally {
    if ($null -ne $originalHttpProxy) { $env:HTTP_PROXY = $originalHttpProxy } else { Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue }
    if ($null -ne $originalHttpsProxy) { $env:HTTPS_PROXY = $originalHttpsProxy } else { Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue }
    if ($null -ne $originalAllProxy) { $env:ALL_PROXY = $originalAllProxy } else { Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue }
    if ($null -ne $originalNoProxy) { $env:NO_PROXY = $originalNoProxy } else { Remove-Item Env:NO_PROXY -ErrorAction SilentlyContinue }
}
