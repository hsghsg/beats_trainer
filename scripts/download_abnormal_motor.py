#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 IDMT 官方 Zenodo ZIP 按需下载电机 broken 录音及原始说明。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from remotezip import RemoteZip

SOURCE_PAGE = "https://zenodo.org/records/7551261"
ARCHIVE_URL = SOURCE_PAGE + "/files/IDMT-ISA-ELECTRIC-ENGINE.zip"
MEMBERS = {
    "train/engine2_broken/pure.wav": "motor_broken_original.wav",
    "IDMT_ISA_ELECTRIC_ENGINE_discription.pdf": "IDMT_description.pdf",
}


def download(output_dir: Path) -> Path:
    """下载固定白名单成员，校验 ZIP CRC 后写入目标目录，并返回来源清单路径。

    HTTP Range 仅读取 ZIP 目录和选中成员；每个成员最多 64 MiB。
    已有结果不覆盖，成员完整读取后才落盘，来源清单记录路径、CRC 和 SHA-256。
    网络、CRC 或文件错误向调用方传播。
    """
    targets = [output_dir / name for name in MEMBERS.values()]
    manifest_path = output_dir / "sources.json"
    for path in [*targets, manifest_path]:
        if path.exists():
            raise FileExistsError(f"文件已存在，下载不会覆盖：{path}")
    downloaded = []
    with RemoteZip(ARCHIVE_URL, initial_buffer_size=524288, timeout=45) as archive:
        for member, name in MEMBERS.items():
            info = archive.getinfo(member)
            if info.file_size > 64 * 1024 * 1024:
                raise ValueError(f"源文件超出预期大小：{member}")
            print(
                f"正在下载：{member}，{info.compress_size / 1024**2:.1f} MiB",
                flush=True,
            )
            content = archive.read(member)
            if len(content) != info.file_size:
                raise ValueError(f"下载内容长度不匹配：{member}")
            downloaded.append(
                (
                    name,
                    content,
                    {
                        "file": name,
                        "archive_member": member,
                        "bytes": len(content),
                        "zip_crc32": f"{info.CRC:08x}",
                        "sha256": hashlib.sha256(content).hexdigest(),
                    },
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content, _ in downloaded:
        (output_dir / name).write_bytes(content)
    manifest = {
        "dataset": "IDMT-ISA-Electric-Engine",
        "record_url": SOURCE_PAGE,
        "archive_url": ARCHIVE_URL,
        "license": "CC-BY-NC-ND-4.0",
        "license_source": "https://zenodo.org/api/records/7551261",
        "label": "broken",
        "split": "train",
        "machine": "engine2",
        "label_context": "通过改变供电电压与负载模拟异常声学工况，非特定故障机理标签。",
        "files": [record for _, _, record in downloaded],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"下载完成：{manifest_path}")
    return manifest_path


def main() -> None:
    """解析下载目录并执行固定数据源下载，默认保存到本地实验素材目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/abnormal_audio/sources")
    )
    download(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
