#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 MIMII public 1.0 的 fan/pump，校验并按信噪比分目录解压，成功后删除 ZIP。"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
import hashlib
import http.client
import json
import logging
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib

RECORD_URL = "https://zenodo.org/records/3384388"
BUFFER_SIZE = 1024 * 1024
SEGMENT_SIZE = 64 * 1024**2
RANGE_REQUEST_SIZE = 8 * 1024**2
RESERVE_BYTES = 1024**3
LOGGER = logging.getLogger("mimii")
STOP_EVENT = threading.Event()


@dataclass(frozen=True)
class ArchiveSpec:
    """保存官方压缩包名称、字节数和 MD5；数值来自固定 Zenodo 记录的文件清单。"""

    name: str
    size: int
    md5: str


# 2026-09-09 核对 https://zenodo.org/api/records/3384388。
ARCHIVES = (
    ArchiveSpec("-6_dB_fan.zip", 10878096548, "f02ae808a58d84b6815b7ec38ff30879"),
    ArchiveSpec("-6_dB_pump.zip", 8236951723, "d20b783a0ff9c93d58f452f98c37b112"),
    ArchiveSpec("0_dB_fan.zip", 10411902283, "6354d1cc2165c52168f9ef1bcd9c7c52"),
    ArchiveSpec("0_dB_pump.zip", 7869431302, "488748295c3f60b25de07b58fe75b049"),
    ArchiveSpec("6_dB_fan.zip", 10158673161, "0890f7d3c2fd8448634e69ff1d66dd47"),
    ArchiveSpec("6_dB_pump.zip", 7659077508, "a09ba6060c10fc09cd4c8770213b0b9f"),
)


def confined_path(root: Path, relative: str) -> Path:
    """将相对路径限制在根目录内，拒绝绝对路径、上级跳转及链接逃逸，返回安全路径。"""
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
        or ".." in parts
    ):
        raise ValueError(f"路径不安全：{relative}")
    candidate = root.joinpath(*parts)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()) or candidate.is_symlink():
        raise ValueError(f"路径超出目标目录或为符号链接：{candidate}")
    return candidate


def check_stopped() -> None:
    """在收到中断请求时抛出异常，使下载和解压循环保留现有文件后退出。"""
    if STOP_EVENT.is_set():
        raise InterruptedError("任务已中断，保留断点文件和未完成压缩包")


def file_digest(file_path: Path, algorithm: str = "md5") -> str:
    """分块读取文件并返回 MD5 或 CRC32，支持中断，不修改文件；读取失败向上抛出。"""
    checksum = hashlib.md5(usedforsecurity=False) if algorithm == "md5" else 0
    with file_path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            check_stopped()
            if algorithm == "md5":
                checksum.update(chunk)
            else:
                checksum = zlib.crc32(chunk, checksum)
    return checksum.hexdigest() if algorithm == "md5" else f"{checksum:08x}"


def verify_archive(file_path: Path, spec: ArchiveSpec) -> None:
    """核对压缩包字节数与官方 MD5；任一不符即报错，并保留原文件供检查。"""
    LOGGER.info("校验压缩包：%s", spec.name)
    if file_path.stat().st_size != spec.size:
        raise ValueError(f"压缩包大小不符，已保留：{file_path}")
    actual = file_digest(file_path)
    if actual != spec.md5:
        raise ValueError(f"MD5 不符，已保留 {file_path}：期望 {spec.md5}，实际 {actual}")
    LOGGER.info("官方 MD5 校验通过：%s", spec.name)


def download_archive(root: Path, spec: ArchiveSpec, retries: int) -> Path:
    """复用已通过校验的 ZIP，或续传 .part；网络失败退避重试，校验成功后才改为 ZIP。"""
    archive_path = confined_path(root, spec.name)
    partial_path = confined_path(root, spec.name + ".part")
    if archive_path.exists():
        verify_archive(archive_path, spec)
        return archive_path
    if partial_path.exists() and partial_path.stat().st_size > spec.size:
        raise ValueError(f"断点文件大于官方大小，已保留：{partial_path}")

    url = f"https://zenodo.org/api/records/3384388/files/{spec.name}/content"
    for attempt in range(retries + 1):
        check_stopped()
        offset = partial_path.stat().st_size if partial_path.exists() else 0
        if offset == spec.size:
            break
        if shutil.disk_usage(root).free < spec.size - offset + RESERVE_BYTES:
            raise OSError(f"磁盘空间不足，无法下载：{spec.name}")
        headers = {
            "User-Agent": "MIMII-Dataset-Downloader/1.0",
            "Accept-Encoding": "identity",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if (
                        not match
                        or int(match[1]) != offset
                        or int(match[3]) != spec.size
                    ):
                        raise ValueError(f"续传响应范围不符：{content_range}")
                elif response.status == 200:
                    if offset:
                        LOGGER.warning("服务器未接受续传，重新下载：%s", spec.name)
                    offset = 0
                else:
                    raise OSError(f"下载响应状态异常：{response.status}")
                LOGGER.info(
                    "下载 %s：从 %.2f / %.2f GiB 开始",
                    spec.name, offset / 1024**3, spec.size / 1024**3,
                )
                last_time = time.monotonic()
                last_offset = offset
                with partial_path.open("ab" if offset else "wb") as destination:
                    while chunk := response.read(BUFFER_SIZE):
                        check_stopped()
                        if offset + len(chunk) > spec.size:
                            raise ValueError(f"下载数据超出官方大小：{spec.name}")
                        destination.write(chunk)
                        offset += len(chunk)
                        now = time.monotonic()
                        if now - last_time >= 30:
                            LOGGER.info(
                                "下载 %s：%.1f%%，%.2f / %.2f GiB，%.2f MiB/s",
                                spec.name, 100 * offset / spec.size,
                                offset / 1024**3, spec.size / 1024**3,
                                (offset - last_offset) / (now - last_time) / 1024**2,
                            )
                            last_time, last_offset = now, offset
                if offset != spec.size:
                    raise OSError(f"连接提前结束：{offset} / {spec.size} 字节")
            break
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            check_stopped()
            if attempt == retries:
                raise RuntimeError(f"下载重试耗尽，已保留断点：{spec.name}") from error
            delay = min(60, 5 * 2 ** min(attempt, 4))
            LOGGER.warning(
                "下载失败 %s：%s；%s 秒后重试（%s/%s）",
                spec.name, error, delay, attempt + 1, retries,
            )
            STOP_EVENT.wait(delay)
    verify_archive(partial_path, spec)
    partial_path.rename(archive_path)
    return archive_path



def download_segment(
    spec: ArchiveSpec, segment: dict, retries: int,
    cancel_event: threading.Event | None = None,
) -> None:
    """用最多 8 MiB 的范围请求续传一个分段；校验响应边界，连续网络失败时退避重试。"""
    target = Path(segment["file"])
    length = segment["end"] - segment["start"] + 1
    url = f"https://zenodo.org/api/records/3384388/files/{spec.name}/content"
    failures = 0
    while True:
        check_stopped()
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("本包分段下载已停止，保留断点")
        downloaded = target.stat().st_size if target.exists() else 0
        if downloaded == length:
            return
        if downloaded > length:
            raise ValueError(f"分段文件超出指定范围：{target}")
        start = segment["start"] + downloaded
        request_end = min(segment["end"], start + RANGE_REQUEST_SIZE - 1)
        request = urllib.request.Request(url, headers={
            "Range": f"bytes={start}-{request_end}",
            "Accept-Encoding": "identity",
            "User-Agent": "MIMII-Dataset-Downloader/1.0",
        })
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = f"bytes {start}-{request_end}/{spec.size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError(f"服务端未返回请求的分段范围：{expected}")
                expected_length = request_end - segment["start"] + 1
                with target.open("ab") as output:
                    while chunk := response.read(BUFFER_SIZE):
                        check_stopped()
                        if cancel_event is not None and cancel_event.is_set():
                            raise InterruptedError("本包分段下载已停止，保留断点")
                        if downloaded + len(chunk) > expected_length:
                            raise ValueError(f"分段返回内容过长：{target}")
                        output.write(chunk)
                        downloaded += len(chunk)
                if downloaded != expected_length:
                    raise OSError(f"分段连接提前结束：{downloaded}/{expected_length}")
            failures = 0
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            check_stopped()
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("本包分段下载已停止，保留断点") from error
            if failures >= retries:
                raise RuntimeError(f"分段连续重试耗尽，已保留：{target}") from error
            delay = min(60, 5 * 2 ** min(failures, 4))
            failures += 1
            if isinstance(error, urllib.error.HTTPError):
                retry_after = error.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
            LOGGER.warning("分段重试 %s [%s]：%s；等待 %s 秒", spec.name, segment["start"], error, delay)
            while delay > 0:
                check_stopped()
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("本包分段下载已停止，保留断点")
                interval = min(1, delay)
                STOP_EVENT.wait(interval)
                delay -= interval


def download_segmented(
    root: Path, spec: ArchiveSpec, retries: int, connections: int,
) -> Path:
    """用有限连接下载固定分段，复用旧单流断点，合并并验证官方 MD5 后清理已合并分段。"""
    archive_path = confined_path(root, spec.name)
    partial_path = confined_path(root, spec.name + ".part")
    if archive_path.exists():
        verify_archive(archive_path, spec)
        return archive_path
    if partial_path.exists() and partial_path.stat().st_size == spec.size:
        verify_archive(partial_path, spec)
        partial_path.rename(archive_path)
        return archive_path
    segment_root = confined_path(root, f".mimii_state/segments/{spec.name}")
    plan_path = confined_path(segment_root, "plan.json")
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["size"] != spec.size or plan["md5"] != spec.md5:
            raise ValueError(f"分段计划与官方清单不符：{plan_path}")
        base_offset = plan["base_offset"]
    else:
        base_offset = partial_path.stat().st_size if partial_path.exists() else 0
        if not 0 <= base_offset < spec.size:
            raise ValueError(f"已有断点大小异常：{partial_path}")
        segment_root.mkdir(parents=True, exist_ok=True)
        plan = {"size": spec.size, "md5": spec.md5, "base_offset": base_offset}
        temporary = confined_path(segment_root, "plan.json.tmp")
        temporary.write_text(json.dumps(plan), encoding="utf-8")
        temporary.replace(plan_path)
    if base_offset and (not partial_path.exists() or partial_path.stat().st_size < base_offset):
        raise ValueError(f"原有断点缺失或长度不足：{partial_path}")
    segment_size = SEGMENT_SIZE
    segments = [
        {
            "start": start,
            "end": min(start + segment_size, spec.size) - 1,
            "file": str(confined_path(segment_root, f"{start:012d}.part")),
        }
        for start in range(base_offset, spec.size, segment_size)
    ]
    # 合并时同时保留分段和完整临时包，按最坏情况预留磁盘空间。
    if shutil.disk_usage(root).free < 2 * (spec.size - base_offset) + RESERVE_BYTES:
        raise OSError(f"磁盘空间不足，无法进行分段下载：{spec.name}")
    LOGGER.info("分段下载 %s：%s 个连接，复用 %.2f GiB，分为 %s 段", spec.name, connections, base_offset / 1024**3, len(segments))
    last_time = time.monotonic()
    last_bytes = base_offset + sum(Path(item["file"]).stat().st_size for item in segments if Path(item["file"]).exists())
    cancel_event = threading.Event()
    with ThreadPoolExecutor(max_workers=connections) as executor:
        pending = {executor.submit(download_segment, spec, item, retries, cancel_event) for item in segments}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except Exception:
                    cancel_event.set()
                    for remaining in pending:
                        remaining.cancel()
                    raise
            now = time.monotonic()
            if now - last_time >= 30 or not pending:
                downloaded = base_offset + sum(Path(item["file"]).stat().st_size for item in segments if Path(item["file"]).exists())
                LOGGER.info(
                    "分段下载 %s：%.1f%%，%.2f / %.2f GiB，%.2f MiB/s",
                    spec.name, 100 * downloaded / spec.size, downloaded / 1024**3,
                    spec.size / 1024**3, (downloaded - last_bytes) / max(now - last_time, 0.001) / 1024**2,
                )
                last_time, last_bytes = now, downloaded
    LOGGER.info("合并已下载分段：%s", spec.name)
    with partial_path.open("r+b" if partial_path.exists() else "w+b") as output:
        # 上次合并若中断，从已保留的原始前缀重新合并；分段仍全部保留。
        output.truncate(base_offset)
        output.seek(base_offset)
        for item in segments:
            check_stopped()
            with Path(item["file"]).open("rb") as source:
                shutil.copyfileobj(source, output, BUFFER_SIZE)
    verify_archive(partial_path, spec)
    partial_path.rename(archive_path)
    for item in segments:
        confined_path(segment_root, Path(item["file"]).name).unlink()
    plan_path.unlink()
    segment_root.rmdir()
    return archive_path



def archive_entries(zipped: zipfile.ZipFile, machine: str) -> list[dict]:
    """检查 ZIP 成员路径及类型，返回相对机器目录的文件大小、CRC 清单，拒绝重复路径。"""
    entries = []
    seen = set()
    for entry in zipped.infolist():
        name = entry.filename
        parts = PurePosixPath(name).parts
        mode = entry.external_attr >> 16
        if (
            not parts
            or parts[0] != machine
            or "/".join(parts) != name.rstrip("/")
            or "\\" in name
            or ":" in name
            or ".." in parts
            or name.startswith("/")
            or stat.S_ISLNK(mode)
        ):
            raise ValueError(f"ZIP 中存在不安全或非目标机器条目：{name}")
        if entry.is_dir():
            continue
        if len(parts) < 2:
            raise ValueError(f"ZIP 文件路径异常：{name}")
        relative = "/".join(parts[1:])
        if relative.casefold() in seen:
            raise ValueError(f"ZIP 中存在重复文件路径：{name}")
        seen.add(relative.casefold())
        entries.append({
            "path": relative,
            "size": entry.file_size,
            "crc32": f"{entry.CRC:08x}",
        })
    if not entries:
        raise ValueError("ZIP 中没有可解压的文件")
    return entries


def verify_files(target: Path, entries: list[dict]) -> None:
    """逐个核对解压文件的存在性、长度和 CRC32；任何缺失或损坏都会阻止 ZIP 删除。"""
    for entry in entries:
        check_stopped()
        file_path = confined_path(target, entry["path"])
        if not file_path.is_file() or file_path.stat().st_size != entry["size"]:
            raise ValueError(f"解压文件缺失或大小不符：{file_path}")
        if file_digest(file_path, "crc32") != entry["crc32"]:
            raise ValueError(f"解压文件 CRC32 不符：{file_path}")


def extract_archive(root: Path, spec: ArchiveSpec, archive_path: Path) -> tuple[Path, list]:
    """在独立暂存目录解压，验证文件后发布到信噪比目录；失败保留 ZIP 和已解压内容。"""
    snr, machine = spec.name.removesuffix(".zip").rsplit("_", 1)
    target = confined_path(root, f"{snr}/{machine}")
    staging = confined_path(root, f".mimii_state/staging/{spec.name}")
    with zipfile.ZipFile(archive_path) as zipped:
        entries = archive_entries(zipped, machine)
        if target.exists():
            LOGGER.info("检查已有解压目录：%s", target)
            verify_files(target, entries)
            return target, entries
        required = sum(entry["size"] for entry in entries)
        if shutil.disk_usage(root).free < required + RESERVE_BYTES:
            raise OSError(f"磁盘空间不足，无法解压：{spec.name}")
        staging.mkdir(parents=True, exist_ok=True)
        LOGGER.info("解压 %s：%s 个文件，%.2f GiB", spec.name, len(entries), required / 1024**3)
        last_time = time.monotonic()
        for index, entry in enumerate(entries, 1):
            check_stopped()
            destination = confined_path(staging, entry["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = confined_path(staging, entry["path"] + ".extracting")
            # 暂存目录由本脚本管理；中断后重新解压对应文件，不触碰最终目录。
            with zipped.open(f"{machine}/{entry['path']}") as source:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, BUFFER_SIZE)
            temporary.replace(destination)
            now = time.monotonic()
            if now - last_time >= 30:
                LOGGER.info("解压 %s：%s/%s 个文件", spec.name, index, len(entries))
                last_time = now
        LOGGER.info("复核全部解压文件的大小和 CRC32：%s", spec.name)
        verify_files(staging, entries)
        target.parent.mkdir(parents=True, exist_ok=True)
        # rename 不覆盖已有目录；最终路径已验证位于用户指定的输出目录中。
        staging.rename(target)
    return target, entries


def process_archive(root: Path, spec: ArchiveSpec, retries: int, connections: int = 1) -> dict:
    """执行单包的续传、MD5 校验、解压和 CRC 复核；写入完成清单后仅删除对应 ZIP。"""
    state_file = confined_path(root, f".mimii_state/{spec.name}.json")
    snr, machine = spec.name.removesuffix(".zip").rsplit("_", 1)
    target = confined_path(root, f"{snr}/{machine}")
    archive_path = confined_path(root, spec.name)
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if (
            state.get("archive") != spec.name
            or state.get("md5") != spec.md5
            or state.get("archive_size") != spec.size
            or not state.get("files")
        ):
            raise ValueError(f"完成清单与官方压缩包不符：{state_file}")
        LOGGER.info("复核已完成的数据集：%s", spec.name)
        verify_files(target, state["files"])
        if archive_path.exists():
            verify_archive(archive_path, spec)
            archive_path.unlink()
            LOGGER.info("已删除完成解压的 ZIP：%s", archive_path)
        LOGGER.info("已完成且复核通过，跳过下载：%s", spec.name)
        return state

    archive_path = (
        download_segmented(root, spec, retries, connections)
        if connections > 1
        else download_archive(root, spec, retries)
    )
    target, entries = extract_archive(root, spec, archive_path)
    state = {
        "record_url": RECORD_URL,
        "archive": spec.name,
        "archive_size": spec.size,
        "md5": spec.md5,
        "target": str(target),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file_count": len(entries),
        "extracted_bytes": sum(entry["size"] for entry in entries),
        "files": entries,
    }
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = confined_path(root, f".mimii_state/{spec.name}.json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(state_file)
    archive_path.unlink()
    LOGGER.info("完成 %s：%s 个文件，已删除 ZIP，目录 %s", spec.name, len(entries), target)
    return state


def parse_args() -> argparse.Namespace:
    """解析目标目录、机器类型、信噪比、并发数及重试次数；非法参数由 argparse 拒绝。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\dataset\MIMII"), help="下载与解压根目录")
    parser.add_argument("--machines", nargs="+", choices=("fan", "pump"), default=["fan", "pump"], help="机器类型")
    parser.add_argument("--snrs", nargs="+", type=int, choices=(-6, 0, 6), default=[-6, 0, 6], help="信噪比，单位 dB")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2, help="同时处理的压缩包数")
    parser.add_argument("--retries", type=int, default=30, help="每包最多重试次数")
    parser.add_argument("--dry-run", action="store_true", help="只列出计划，不下载或修改目录")
    parser.add_argument("--connections", type=int, choices=(1, 2, 4, 8), default=4, help="每包分段连接数；1 表示单连接续传")
    args = parser.parse_args()
    if args.retries < 0:
        parser.error("--retries 不能为负数")
    return args


def main() -> int:
    """运行所选包的处理队列并记录中文日志；全部成功返回 0，失败返回 1，中断返回 130。"""
    args = parse_args()
    root = args.output_dir.resolve()
    selected = [
        spec for spec in ARCHIVES
        if spec.name.rsplit("_", 1)[-1].removesuffix(".zip") in args.machines
        and int(spec.name.split("_")[0]) in args.snrs
    ]
    # 已有 ZIP 优先进入处理队列，避免重复下载，并尽早释放压缩包空间。
    selected.sort(key=lambda spec: not (root / spec.name).exists())
    if args.dry_run:
        print(f"目标目录：{root}")
        for spec in selected:
            exists = (root / spec.name).exists()
            print(f"{spec.name}：{spec.size:,} 字节，MD5={spec.md5}，已有 ZIP={exists}")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    log_path = confined_path(root, "mimii_download.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    LOGGER.info("开始处理 %s 个压缩包，目标 %s，来源 %s", len(selected), root, RECORD_URL)
    failures = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_archive, root, spec, args.retries, args.connections): spec for spec in selected}
            try:
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        future.result()
                    except Exception:
                        failures.append(spec.name)
                        LOGGER.exception("处理失败，保留压缩包或断点文件：%s", spec.name)
            except KeyboardInterrupt:
                STOP_EVENT.set()
                for future in futures:
                    future.cancel()
                raise
    except KeyboardInterrupt:
        LOGGER.warning("收到中断，已保留续传和解压状态，可重新执行相同命令")
        return 130
    if failures:
        LOGGER.error("未完成的压缩包：%s", "、".join(failures))
        return 1
    LOGGER.info("全部完成：%s 个压缩包均已解压、复核并删除 ZIP", len(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

