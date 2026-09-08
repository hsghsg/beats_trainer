"""验证设备声音融合的能量关系、重采样、异常输入和端到端输出。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.fuse_abnormal_audio import (
    file_digest,
    load_audio,
    mix_components,
    parse_args,
    peak_limit,
    rms,
    run,
)


class AudioFusionTests(unittest.TestCase):
    """采用可控的互异频率信号验证算法，不依赖模型权重或网络数据。"""

    def setUp(self) -> None:
        """构建两秒等长但不同幅值、不同频率的泵和电机测试输入。"""
        self.rate = 16000
        time = np.arange(self.rate * 2) / self.rate
        self.pump = 0.8 * np.sin(2 * np.pi * 317 * time)
        self.motor = 0.1 * np.sin(2 * np.pi * 733 * time)

    def test_snr_and_common_limiting(self) -> None:
        """验证 -6、0、6 dB 两路功率比，并确认防削波只使用统一线性增益。"""
        for target in (-6, 0, 6):
            with self.subTest(target=target):
                first, second, _ = mix_components(
                    self.pump,
                    self.motor,
                    "snr",
                    self.rate,
                    snr_db=target,
                )
                self.assertAlmostEqual(20 * np.log10(rms(first) / rms(second)), target)
                limited, gain = peak_limit(first + second, 0.95)
                self.assertLessEqual(np.max(np.abs(limited)), 0.95 + 1e-12)
                np.testing.assert_allclose(
                    limited, gain * first + gain * second, atol=1e-12
                )

    def test_weighted_keeps_physical_amplitude_ratio(self) -> None:
        """验证加权叠加不会暗中对两路分别归一化，保留输入幅度差异。"""
        first, second, _ = mix_components(
            self.pump,
            self.motor,
            "weighted",
            self.rate,
            alpha=0.25,
        )
        self.assertAlmostEqual(rms(first) / rms(second), 8 / 3)

    def test_crossfade_endpoints_and_power(self) -> None:
        """以恒定输入检查渐变端点和等功率包络，不要求相关音频合成能量恒定。"""
        ones = np.ones(100)
        first, second, _ = mix_components(ones, ones, "crossfade", self.rate)
        np.testing.assert_allclose(first**2 + second**2, 1, atol=1e-12)
        self.assertAlmostEqual(first[0], 1)
        self.assertAlmostEqual(second[0], 0)
        self.assertAlmostEqual(first[-1], 0)
        self.assertAlmostEqual(second[-1], 1)

    def test_intermittent_off_intervals_and_active_snr(self) -> None:
        """验证电机关闭区间完全静音、边界平滑到零及开启区间指定功率比。"""
        first, second, info = mix_components(
            self.pump,
            self.motor,
            "intermittent",
            self.rate,
            period=1,
            snr_db=6,
        )
        np.testing.assert_array_equal(first, self.pump)
        np.testing.assert_array_equal(second[8000:16000], 0)
        self.assertEqual(second[7999], 0)
        selected = ((np.arange(len(first)) % 16000) > 0) & (
            (np.arange(len(first)) % 16000) < 7999
        )
        self.assertAlmostEqual(
            20 * np.log10(rms(first[selected]) / rms(second[selected])), 6
        )
        self.assertEqual(info["snr_reference"], "active_mask_samples")

    def test_reject_invalid_inputs(self) -> None:
        """验证静音、非有限值、长度不一致及无法产生间歇淡变的参数被拒绝。"""
        for bad in (np.zeros_like(self.motor), self.motor[:-1], self.motor * np.nan):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                mix_components(self.pump, bad, "snr", self.rate)
        for options in ({"alpha": 1}, {"alpha": float("nan")}):
            with self.assertRaises(ValueError):
                mix_components(self.pump, self.motor, "weighted", self.rate, **options)
        with self.assertRaises(ValueError):
            mix_components(
                self.pump, self.motor, "intermittent", self.rate, fade_seconds=2
            )

    def test_resampling_suppresses_aliasing(self) -> None:
        """验证 44.1 kHz 到 16 kHz 重采样抑制超过目标奈奎斯特频率的信号。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "high.wav"
            time = np.arange(44100) / 44100
            sf.write(
                path, 0.5 * np.sin(2 * np.pi * 12000 * time), 44100, subtype="FLOAT"
            )
            result, _ = load_audio(path, self.rate, 0, 1, 0)
            self.assertEqual(len(result), 16000)
            self.assertLess(rms(result[100:-100]), 0.001)

    def test_channel_crop_and_duration_validation(self) -> None:
        """验证首通道选取、起点裁剪、去直流，以及超长请求和越界通道拒绝。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(
                path,
                np.column_stack((self.pump + 0.1, self.motor)),
                self.rate,
                subtype="FLOAT",
            )
            audio, info = load_audio(path, self.rate, 0.5, 1, 0)
            self.assertEqual(info["start_frame"], 8000)
            np.testing.assert_allclose(audio, self.pump[8000:24000], atol=1e-7)
            with self.assertRaises(ValueError):
                load_audio(path, self.rate, 1, 2, 0)
            with self.assertRaises(ValueError):
                load_audio(path, self.rate, 0, 1, 2)

    def test_end_to_end_manifest_and_no_overwrite(self) -> None:
        """验证两路不同时长产生六种默认结果、正确 PCM 格式和追溯信息且不覆盖。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pump_path, motor_path = root / "pump.wav", root / "motor.wav"
            sf.write(pump_path, self.pump, self.rate)
            sf.write(motor_path, np.tile(self.motor, 2), self.rate)
            args = parse_args(
                [
                    "--pump",
                    str(pump_path),
                    "--motor",
                    str(motor_path),
                    "--output-dir",
                    str(root / "result"),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                manifest_path = run(args)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["frames"], 32000)
            self.assertEqual(len(manifest["outputs"]), 8)
            for record in manifest["outputs"]:
                path = manifest_path.parent / record["file"]
                info = sf.info(path)
                self.assertEqual(
                    (info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16")
                )
                self.assertEqual(info.frames, 32000)
                self.assertEqual(record["sha256"], file_digest(path))
                self.assertGreater(record["rms"], 0)
                self.assertLessEqual(record["peak"], 0.95 + 1 / 32768)
            original = file_digest(manifest_path)
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual(file_digest(manifest_path), original)


if __name__ == "__main__":
    unittest.main()
