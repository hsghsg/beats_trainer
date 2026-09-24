# 从 Windows 调用指定 Ubuntu WSL，不通过字符串拼接构造 Linux shell 命令。
param(
    [ValidateSet('doctor', 'pt-to-onnx', 'onnx-to-om', 'all', 'atc-command')]
    [string]$Action = 'doctor',
    [string]$Distribution = 'Ubuntu-22.04',
    [string]$Config = (Join-Path $PSScriptRoot 'transfer.yaml')
)
$ErrorActionPreference = 'Stop'
$configPath = (Resolve-Path -LiteralPath $Config).Path
$linuxProject = (& wsl.exe -d $Distribution --exec wslpath -a $PSScriptRoot).Trim()
if ($LASTEXITCODE -ne 0) { throw '无法将子项目目录转换为 WSL 路径' }
$linuxConfig = (& wsl.exe -d $Distribution --exec wslpath -a $configPath).Trim()
if ($LASTEXITCODE -ne 0) { throw '无法将配置文件转换为 WSL 路径' }
& wsl.exe -d $Distribution --exec "$linuxProject/.venv-wsl/bin/python" "$linuxProject/convert.py" $Action --config $linuxConfig
exit $LASTEXITCODE
