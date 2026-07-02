$ErrorActionPreference = "Stop"

$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $AppRoot
$ServerLog = Join-Path $AppRoot "server.out.log"
$ServerErr = Join-Path $AppRoot "server.err.log"
$TunnelLog = Join-Path $AppRoot "tunnel.out.log"
$TunnelErr = Join-Path $AppRoot "tunnel.err.log"
$Port = if ($env:BIRD_WEB_PORT) { [int]$env:BIRD_WEB_PORT } else { 8000 }

function Read-LogText {
  param([string]$Path)

  if (Test-Path $Path) {
    $text = Get-Content -Raw $Path
    if ($null -ne $text) {
      return $text
    }
  }
  return ""
}

Write-Host "正在启动鸟类识别服务，端口：$Port ..."
try {
  $healthUrl = "http://127.0.0.1:" + $Port + "/api/health"
  $health = Invoke-WebRequest -UseBasicParsing $healthUrl -TimeoutSec 3
  if ($health.StatusCode -eq 200) {
    Write-Host "服务已在端口 $Port 正常响应。"
    $server = $null
  }
} catch {
  $serverScript = Join-Path $ProjectRoot "bird_web_app\server.py"
  $serverArgs = New-Object System.Collections.Generic.List[string]
  $serverArgs.Add($serverScript)
  $serverArgs.Add("--host")
  $serverArgs.Add("0.0.0.0")
  $serverArgs.Add("--port")
  $serverArgs.Add([string]$Port)
  $serverArgs.Add("--warmup")
  $server = Start-Process -FilePath "python" -ArgumentList $serverArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $ServerLog -RedirectStandardError $ServerErr -PassThru
  Start-Sleep -Seconds 8
}

Write-Host "正在启动公网 HTTPS 隧道..."
$tunnelTarget = "80:127.0.0.1:" + $Port
$tunnelArgs = New-Object System.Collections.Generic.List[string]
$tunnelArgs.Add("-o")
$tunnelArgs.Add("StrictHostKeyChecking=no")
$tunnelArgs.Add("-o")
$tunnelArgs.Add("ServerAliveInterval=60")
$tunnelArgs.Add("-R")
$tunnelArgs.Add($tunnelTarget)
$tunnelArgs.Add("nokey@localhost.run")
$tunnel = Start-Process -FilePath "ssh" -ArgumentList $tunnelArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $TunnelLog -RedirectStandardError $TunnelErr -PassThru

$match = $null
for ($i = 0; $i -lt 20; $i++) {
  Start-Sleep -Seconds 2
  $content = (Read-LogText $TunnelLog) + "`n" + (Read-LogText $TunnelErr)
  $match = [regex]::Match($content, "https://[^\s]+\.lhr\.life")
  if ($match.Success) {
    break
  }
}

if ($server) {
  Write-Host "服务进程编号：$($server.Id)"
}
Write-Host "隧道进程编号：$($tunnel.Id)"
if ($null -ne $match -and $match.Success) {
  Write-Host "公网地址：$($match.Value)"
} else {
  Write-Host "暂时没有读取到公网地址，请稍后查看 $TunnelLog 和 $TunnelErr。"
}
