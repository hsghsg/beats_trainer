"""验证 MIMII 下载器的断点续传、校验失败保护、分目录解压及重复执行行为。"""

import hashlib
import io
from pathlib import Path
import zipfile

import tempfile
import unittest
from unittest.mock import patch

from scripts import download_mimii_dataset as downloader


def build_archive(root: Path, name: str = "-6_dB_pump.zip") -> downloader.ArchiveSpec:
    """在测试临时目录创建含一条示例文件的 ZIP，返回与其字节数、MD5 一致的描述。"""
    archive_path = root / name
    machine = name.removesuffix(".zip").rsplit("_", 1)[1]
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{machine}/id_00/normal/00000000.wav", b"test audio data")
    return downloader.ArchiveSpec(
        name, archive_path.stat().st_size,
        hashlib.md5(archive_path.read_bytes(), usedforsecurity=False).hexdigest(),
    )


class TestMimiiDownloader(unittest.TestCase):
    """在隔离的临时目录中测试下载器，禁止触碰真实 MIMII 数据。"""

    def setUp(self):
        """为每个测试创建独立临时目录，并注册清理操作。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_extract_delete_and_repeat(self):
        """成功解压后仅删除对应 ZIP，保留其他文件，并保证重复执行不再次下载。"""
        spec = build_archive(self.root)
        unrelated = self.root / "unrelated.zip"
        unrelated.write_bytes(b"do not delete")
        state = downloader.process_archive(self.root, spec, 0)
        assert not (self.root / spec.name).exists()
        assert unrelated.read_bytes() == b"do not delete"
        assert (self.root / "-6_dB/pump/id_00/normal/00000000.wav").read_bytes() == b"test audio data"
        assert state["file_count"] == 1

        def reject_network(*args, **kwargs):
            """重复执行已完成任务时禁止联网，若被调用则使测试失败。"""
            raise AssertionError("已完成的数据不应再次下载")

        patcher = patch.object(downloader.urllib.request, "urlopen", reject_network)
        patcher.start()
        self.addCleanup(patcher.stop)
        assert downloader.process_archive(self.root, spec, 0)["md5"] == spec.md5


    def test_bad_md5_keeps_archive(self):
        """官方 MD5 不匹配时必须保留 ZIP，且不创建最终解压目录。"""
        spec = build_archive(self.root)
        invalid = downloader.ArchiveSpec(spec.name, spec.size, "0" * 32)
        with self.assertRaisesRegex(ValueError, "MD5"):
            downloader.process_archive(self.root, invalid, 0)
        assert (self.root / spec.name).exists()
        assert not (self.root / "-6_dB/pump").exists()


    def test_conflicting_target_keeps_archive(self):
        """已有解压文件内容冲突时不覆盖该文件，并保留已通过 MD5 的 ZIP。"""
        spec = build_archive(self.root)
        target = self.root / "-6_dB/pump/id_00/normal/00000000.wav"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * len(b"test audio data"))
        with self.assertRaisesRegex(ValueError, "CRC32"):
            downloader.process_archive(self.root, spec, 0)
        assert target.read_bytes() == b"x" * len(b"test audio data")
        assert (self.root / spec.name).exists()


    def test_snr_directories_do_not_overlap(self):
        """不同信噪比 ZIP 的相同内部文件名必须解压至不同目录。"""
        for name in ("-6_dB_pump.zip", "0_dB_pump.zip"):
            downloader.process_archive(self.root, build_archive(self.root, name), 0)
        assert (self.root / "-6_dB/pump/id_00/normal/00000000.wav").exists()
        assert (self.root / "0_dB/pump/id_00/normal/00000000.wav").exists()


    def test_unsafe_zip_member_is_rejected(self):
        """拒绝目录跳转、绝对路径和反斜杠成员路径，并保留相应源 ZIP。"""
        for name in ["pump/../../outside.wav", "/pump/file.wav", r"pump\\file.wav"]:
            with self.subTest(name=name):
                archive_path = self.root / "-6_dB_pump.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(name, b"unsafe")
                spec = downloader.ArchiveSpec(
                    archive_path.name, archive_path.stat().st_size,
                    hashlib.md5(archive_path.read_bytes(), usedforsecurity=False).hexdigest(),
                )
                with self.assertRaisesRegex(ValueError, "不安全"):
                    downloader.process_archive(self.root, spec, 0)
                assert archive_path.exists()


    def test_resume_download(self):
        """已有 .part 时发送正确 Range，并将返回片段追加为通过官方校验的完整文件。"""
        payload = b"complete archive bytes"
        spec = downloader.ArchiveSpec(
            "-6_dB_pump.zip", len(payload),
            hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        )
        (self.root / (spec.name + ".part")).write_bytes(payload[:5])

        def open_partial_response(request, timeout):
            """模拟支持 Range 的服务端，返回与请求偏移和总长度一致的 206 响应。"""
            assert request.get_header("Range") == "bytes=5-"
            response = io.BytesIO(payload[5:])
            response.status = 206
            response.headers = {"Content-Range": f"bytes 5-{len(payload) - 1}/{len(payload)}"}
            return response

        patcher = patch.object(downloader.urllib.request, "urlopen", open_partial_response)
        patcher.start()
        self.addCleanup(patcher.stop)
        result = downloader.download_archive(self.root, spec, 0)
        assert result.read_bytes() == payload
        assert not (self.root / (spec.name + ".part")).exists()


    def test_ignored_range_restarts_safely(self):
        """服务端忽略 Range 并返回 200 时重新写入完整 .part，不能把响应错误追加。"""
        payload = b"complete archive bytes"
        spec = downloader.ArchiveSpec(
            "-6_dB_pump.zip", len(payload),
            hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        )
        (self.root / (spec.name + ".part")).write_bytes(payload[:5])

        def open_full_response(request, timeout):
            """模拟忽略 Range 的服务端，返回完整正文和 200 状态。"""
            response = io.BytesIO(payload)
            response.status = 200
            response.headers = {}
            return response

        patcher = patch.object(downloader.urllib.request, "urlopen", open_full_response)
        patcher.start()
        self.addCleanup(patcher.stop)
        assert downloader.download_archive(self.root, spec, 0).read_bytes() == payload


if __name__ == "__main__":
    unittest.main()

