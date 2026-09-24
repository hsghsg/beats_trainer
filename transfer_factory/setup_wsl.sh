#!/usr/bin/env bash
# 在当前子项目创建独立 CPU 导出环境，不修改系统 Python，也不安装 CANN。
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$project_dir/.venv-wsl"
"$project_dir/.venv-wsl/bin/python" -m pip install --upgrade pip
"$project_dir/.venv-wsl/bin/python" -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"$project_dir/.venv-wsl/bin/python" -m pip install -r "$project_dir/requirements.txt"
printf '%s\n' '转换环境安装完成；CANN Toolkit 请按目标设备版本另行安装。'
