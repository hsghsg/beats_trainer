"""使用昇腾 CANN ATC 将固定输入 ONNX 编译为 OM。"""

import logging
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)
RESERVED = {
    "model",
    "framework",
    "output",
    "input_shape",
    "input_format",
    "soc_version",
    "precision_mode",
    "mode",
    "dynamic_batch_size",
    "dynamic_image_size",
    "dynamic_dims",
}


def build_atc_command(config: dict, output_prefix: Path | None = None) -> list[str]:
    """构造 shell=False 的 ATC 参数数组，固定 framework=5 并禁止额外参数覆盖核心配置。

    output_prefix 可指向临时目录，以便成功后才发布 .om；型号必须显式配置。
    """
    atc, inp = config["atc"], config["input"]
    soc = atc["soc_version"]
    if not isinstance(soc, str) or not re.fullmatch(r"Ascend[A-Za-z0-9]+", soc):
        raise ValueError("必须配置目标设备的 atc.soc_version，例如 Ascend310B1")
    if not isinstance(atc["executable"], str) or not atc["executable"].strip():
        raise ValueError("atc.executable 必须为可执行文件名或绝对路径")
    if atc["input_format"] not in {"ND", "NCHW", "NHWC", "NCDHW", "NDHWC"}:
        raise ValueError("atc.input_format 无效")
    prefix = output_prefix if output_prefix is not None else atc["output"]
    shape = ",".join(str(size) for size in inp["shape"])
    command = [
        atc["executable"],
        f"--model={config['onnx']['path']}",
        "--framework=5",
        f"--output={prefix}",
        f"--input_shape={inp['name']}:{shape}",
        f"--input_format={atc['input_format']}",
        f"--soc_version={soc}",
    ]
    if atc["precision_mode"] is not None:
        command.append(f"--precision_mode={atc['precision_mode']}")
    for key, value in atc["extra_args"].items():
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", key)
            or key in RESERVED
        ):
            raise ValueError(f"额外 ATC 参数不合法或覆盖核心参数：{key}")
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise TypeError(f"ATC 参数 {key} 必须为文本或数值")
        command.append(f"--{key}={value}")
    return command


def cann_environment(
    script: Path | None, devlib_path: Path | None = None
) -> dict[str, str]:
    """读取 CANN set_env.sh 的导出环境；脚本路径通过位置参数传递，避免命令拼接。

    返回完整子进程环境而不修改父进程；缺少脚本或脚本失败抛出异常。
    可选 devlib_path 为无 NPU 主机添加离线编译库；编译器使用当前 Python 环境。
    """
    environment = os.environ.copy()
    if devlib_path is not None and not devlib_path.is_dir():
        raise FileNotFoundError(f"CANN 离线编译库目录不存在：{devlib_path}")
    if script is None:
        environment["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
        )
        if devlib_path is not None:
            environment["LD_LIBRARY_PATH"] = (
                str(devlib_path) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
            )
        return environment
    if not script.is_file():
        raise FileNotFoundError(
            f"CANN 环境脚本不存在：{script}；请安装 Toolkit 或设置正确路径"
        )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -e; source "$1" >&2; /usr/bin/env -0',
            "transfer-factory",
            str(script),
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "CANN 环境初始化失败：" + completed.stderr.decode(errors="replace")
        )
    for entry in completed.stdout.split(b"\0"):
        key, separator, value = entry.partition(b"=")
        if separator:
            environment[os.fsdecode(key)] = os.fsdecode(value)
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    if devlib_path is not None:
        environment["LD_LIBRARY_PATH"] = (
            str(devlib_path) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
    return environment


def terminate_process_group(process: subprocess.Popen) -> None:
    """在 Linux 终止 ATC 所在进程组并回收进程，防止超时/中断后遗留编译子进程。"""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def compile_om(config: dict) -> dict:
    """校验 ONNX，初始化 CANN 后执行真实 ATC；成功且生成非空 .om 才发布产物。

    超时、中断、ATC 非零退出或缺失产物都视为失败；原有 .om 不会被失败覆盖。
    ATC 输出持续写入独立日志，适合长时间编译且不无限占用 Python 内存。
    """
    from .onnx_export import check_onnx_contract, sha256_file

    if platform.system() != "Linux":
        raise RuntimeError("ATC 转换请在 Ubuntu/WSL 的 Linux 环境执行")
    atc = config["atc"]
    command = build_atc_command(config)
    output = Path(str(atc["output"]) + ".om")
    if output.exists() and not config["runtime"]["overwrite"]:
        raise FileExistsError(f"OM 已存在，请配置 overwrite 或更换输出路径：{output}")
    contract = check_onnx_contract(config["onnx"]["path"], config)
    environment = cann_environment(atc["env_script"], atc.get("devlib_path"))
    executable = shutil.which(command[0], path=environment.get("PATH"))
    if executable is None:
        raise FileNotFoundError(
            "未找到 ATC。请在 WSL 安装 CANN Toolkit，并核对 env_script/executable"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(str(atc["output"]) + ".atc.log")
    with tempfile.TemporaryDirectory(prefix=".atc-", dir=output.parent) as directory:
        prefix = Path(directory) / "model"
        command = build_atc_command(config, prefix)
        command[0] = executable
        LOGGER.info("开始 ATC 编译，目标芯片=%s；日志=%s", atc["soc_version"], log_path)
        LOGGER.info("ATC 命令：%s", shlex.join(command))
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=atc["timeout_seconds"])
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                terminate_process_group(process)
                raise
        if code != 0:
            raise RuntimeError(f"ATC 转换失败，退出码 {code}，请查看 {log_path}")
        generated = Path(str(prefix) + ".om")
        if not generated.is_file() or generated.stat().st_size == 0:
            raise RuntimeError(f"ATC 未生成非空 OM 文件，请查看 {log_path}")
        if output.exists() and not config["runtime"]["overwrite"]:
            raise FileExistsError(f"编译期间目标 OM 文件被创建：{output}")
        generated.replace(output)
    LOGGER.info("OM 转换成功：%s", output)
    return {
        "stage": "onnx-to-om",
        "status": "success",
        "output": str(output),
        "soc_version": atc["soc_version"],
        "sha256": sha256_file(output),
        "atc_log": str(log_path),
        "command": command,
        "contract": contract,
    }
