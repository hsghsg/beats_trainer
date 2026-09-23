#!/usr/bin/env python3
"""独立 ASD 命令行入口：python infer.py --config infer.yaml。"""

import argparse
import json
import logging
import sys
from contextlib import ExitStack, closing
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from asd.config import load_config
else:
    from .config import load_config


LOGGER = logging.getLogger(__name__)


def configure_logging(config: dict) -> None:
    """设置中文终端日志和可选 UTF-8 文件日志，日志文件按追加方式写入。"""
    handlers = [logging.StreamHandler(sys.stderr)]
    if config["log_path"] is not None:
        config["log_path"].parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config["log_path"], encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, config["log_level"]),
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.getLogger("asd.vendor").setLevel(logging.WARNING)


def main() -> int:
    """读取配置并连续输出 Result JSON，推理失败返回 1，中断返回 130。

    --check-config 仅校验 YAML，不加载权重；业务参数全部放在 infer.yaml。
    结果逐窗口写入 stdout，并可追加到 JSONL 文件以避免长时间运行积累内存。
    """
    parser = argparse.ArgumentParser(description="BEATs 异常声音检测推理")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("infer.yaml"),
        help="集中配置文件路径",
    )
    parser.add_argument("--check-config", action="store_true", help="仅校验 YAML 配置")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.check_config:
            print("infer.yaml 配置校验通过")
            return 0
        configure_logging(config["output"])
        if __package__ in (None, ""):
            from asd.engine import ASDInference
        else:
            from .engine import ASDInference
        inference = ASDInference(args.config)
        with ExitStack() as stack:
            output = None
            if config["output"]["jsonl_path"] is not None:
                path = config["output"]["jsonl_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                output = stack.enter_context(path.open("a", encoding="utf-8"))
            iterator = stack.enter_context(closing(inference.run()))
            count = 0
            for result in iterator:
                line = json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
                print(line, flush=True)
                if output is not None:
                    output.write(line + "\n")
                    output.flush()
                count += 1
            LOGGER.info("推理完成，共处理 %d 个十秒窗口", count)
        return 0
    except KeyboardInterrupt:
        LOGGER.info("用户中断推理")
        return 130
    except Exception:
        LOGGER.exception("ASD 推理失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
