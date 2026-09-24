#!/usr/bin/env python3
"""转换工厂 CLI，支持分阶段转换、串联转换和环境检查。"""

import argparse
import importlib.util
import json
import logging
import platform
import shlex
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from transfer_factory.atc import build_atc_command, cann_environment
    from transfer_factory.config import load_config
    from transfer_factory.factory import TransferFactory
else:
    from .atc import build_atc_command, cann_environment
    from .config import load_config
    from .factory import TransferFactory

LOGGER = logging.getLogger(__name__)


def doctor(config: dict) -> dict:
    """检查 Linux、Python 依赖、CANN 环境和 ATC 路径，不导入或加载大模型。"""
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "torch",
            "torchaudio",
            "numpy",
            "onnx",
            "onnxruntime",
            "yaml",
            "scipy",
            "sklearn",
        )
    }
    result = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": packages,
        "soc_version": config["atc"]["soc_version"],
        "atc": None,
        "atc_error": None,
    }
    try:
        if platform.system() != "Linux":
            raise RuntimeError("请通过 run_wsl.ps1 或在 Ubuntu 内运行转换")
        build_atc_command(config)
        environment = cann_environment(
            config["atc"]["env_script"], config["atc"].get("devlib_path")
        )
        result["atc"] = shutil.which(
            config["atc"]["executable"], path=environment.get("PATH")
        )
        if result["atc"] is None:
            raise FileNotFoundError("CANN 环境中未找到 atc")
    except (OSError, ValueError, RuntimeError) as error:
        result["atc_error"] = str(error)
    result["onnx_ready"] = all(packages.values())
    result["om_ready"] = bool(result["atc"] and packages["onnx"])
    return result


def write_report(path: Path, result: dict) -> None:
    """将当前操作的状态、阶段结果及错误原子写入 UTF-8 JSON 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".report-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    """解析 CLI 并运行转换阶段；成功返回 0，失败返回 1，doctor 未就绪返回 2。

    all 在 ONNX 成功后才执行 OM；OM 失败仍保留 ONNX 与失败日志。
    中断返回 130。report 始终标明操作及真实阶段状态，不将命令预览当成编译成功。
    """
    parser = argparse.ArgumentParser(description="Ubuntu/WSL 模型格式转换工厂")
    parser.add_argument(
        "action", choices=("doctor", "pt-to-onnx", "onnx-to-om", "all", "atc-command")
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("transfer.yaml")
    )
    args = parser.parse_args()
    config = None
    report = {
        "action": args.action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stages": [],
    }
    code = 0
    try:
        config = load_config(args.config)
        log_path = config["runtime"]["log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            force=True,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding="utf-8"),
            ],
        )
        logging.getLogger("asd.vendor").setLevel(logging.WARNING)
        if args.action == "doctor":
            report["environment"] = doctor(config)
            code = (
                0
                if report["environment"]["onnx_ready"]
                and report["environment"]["om_ready"]
                else 2
            )
            report["status"] = "ready" if code == 0 else "not_ready"
        elif args.action == "atc-command":
            report["command"] = shlex.join(build_atc_command(config))
            report["status"] = "preview_only"
        else:
            factory = TransferFactory()
            if args.action in {"all", "pt-to-onnx"}:
                report["stages"].append(factory.convert("pt", "onnx", config))
            if args.action in {"all", "onnx-to-om"}:
                report["stages"].append(factory.convert("onnx", "om", config))
            report["status"] = "success"
    except KeyboardInterrupt:
        report.update(status="interrupted", error="用户中断转换")
        code = 130
    except Exception as error:
        report.update(
            status="failed", error=str(error), error_type=type(error).__name__
        )
        LOGGER.exception("模型转换未完成")
        code = 1
    if config is not None:
        write_report(config["runtime"]["report_path"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
