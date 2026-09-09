"""使用临时音频验证平圩预处理的采样内容、命名、边界条件及输出保护。"""

import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from scripts.preprocess_pingwei_audio import (
    DEFAULT_CONFIG,
    build_plan,
    load_config,
    load_resampled_audio,
    run,
)


class PingweiAudioTests(unittest.TestCase):
    """以可控信号执行独立端到端测试，不访问真实数据目录或加载模型。"""

    def setUp(self) -> None:
        """建立独立临时工作区并加载配套 YAML，所有测试输出在用例结束后清理。"""
        temporary = tempfile.TemporaryDirectory(prefix="pingwei_audio_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.config_path = self.root / "config.yaml"
        self.values = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8-sig"))
        self.values.update(input_dir="input", output_dir="output", log_level="ERROR")

    def config(self, **overrides):
        """将用例参数写入临时 YAML 并通过正式加载器校验，返回处理配置。"""
        values = copy.deepcopy(self.values)
        values.update(overrides)
        self.config_path.write_text(
            yaml.safe_dump(values, allow_unicode=True), encoding="utf-8"
        )
        return load_config(self.config_path)

    def write_source(self, name, audio, rate=16000):
        """把给定数组保存为临时浮点 WAV，返回原始文件路径用于结果对照。"""
        path = self.input_dir / name
        sf.write(path, audio, rate, subtype="FLOAT")
        return path

    def test_resampling_content_filtering_and_exact_names(self) -> None:
        """验证三条件筛选、22 秒立体声降采样混音及两个完整 10 秒片段的真实频率。"""
        rate = 48000
        time = np.arange(rate * 22) / rate
        tone = np.sin(2 * np.pi * 440 * time)
        original = self.write_source(
            "sync_20260824_163405_000001_7fa13.wav",
            np.column_stack((0.6 * tone, 0.2 * tone)),
            rate,
        )
        original_size = original.stat().st_size
        for name in (
            "other_20260824_163405_7fa13.wav",
            "sync_20260824_163405_2bv2.wav",
            "sync_20260824_163405_7fa13.json",
        ):
            (self.input_dir / name).write_bytes(b"excluded")
        nested = self.input_dir / "nested"
        nested.mkdir()
        (nested / "sync_bad_7fa13.wav").write_bytes(b"excluded")
        config = self.config()
        summary = run(config)
        self.assertEqual(
            summary, {"source_files": 1, "planned_clips": 2, "written_clips": 2}
        )
        files = sorted(config.output_dir.glob("*.wav"))
        self.assertEqual(
            [p.name for p in files],
            [
                f"pw-7fa13-7.闭式水泵.C.mid-260824.1634-normal-{index:04d}.wav"
                for index in (1, 2)
            ],
        )
        for path in files:
            info = sf.info(path)
            self.assertEqual(
                (info.samplerate, info.frames, info.channels, info.subtype),
                (16000, 160000, 1, "PCM_16"),
            )
            audio, _ = sf.read(path)
            expected = 0.4 * np.sin(2 * np.pi * 440 * np.arange(160000) / 16000)
            self.assertLess(
                np.sqrt(np.mean((audio[100:-100] - expected[100:-100]) ** 2)), 0.001
            )
        self.assertEqual(original.stat().st_size, original_size)
        self.assertEqual(sf.info(original).samplerate, 48000)

    def test_padding_preserves_tail_and_adds_silence(self) -> None:
        """验证 12.5 秒录音补零后生成两段，末段前 2.5 秒保留原值，其后为静音。"""
        self.write_source("sync_20260824_163405_7fa13.wav", np.full(200000, 0.25))
        config = self.config(tail_policy="pad")
        self.assertEqual(run(config)["written_clips"], 2)
        tail, _ = sf.read(sorted(config.output_dir.glob("*.wav"))[-1])
        self.assertEqual(len(tail), 160000)
        np.testing.assert_array_equal(tail[:40000], 0.25)
        np.testing.assert_array_equal(tail[40000:], 0)

    def test_short_and_empty_recordings(self) -> None:
        """验证短录音的丢弃与补零策略，空录音在两种策略下均不产生切片。"""
        self.write_source("sync_20260824_163405_7fa13.wav", np.full(8000, 0.25))
        self.write_source("sync_20260824_163505_7fa13.wav", np.empty(0))
        config = self.config()
        self.assertEqual(run(config)["written_clips"], 0)
        self.assertFalse(config.output_dir.exists())
        config = self.config(tail_policy="pad")
        self.assertEqual(run(config)["written_clips"], 1)
        audio, _ = sf.read(next(config.output_dir.glob("*.wav")))
        np.testing.assert_array_equal(audio[:8000], 0.25)
        np.testing.assert_array_equal(audio[8000:], 0)

    def test_same_minute_continues_sequence_and_new_minute_resets(self) -> None:
        """验证同一分钟多个源文件不会互相覆盖，而新一分钟恢复从 0001 编号。"""
        for time in ("163401", "163450", "163501"):
            self.write_source(f"sync_20260824_{time}_7fa13.WAV", np.zeros(160000))
        config = self.config()
        self.assertEqual(run(config)["written_clips"], 3)
        self.assertEqual(
            sorted(path.name for path in config.output_dir.glob("*.wav")),
            [
                "pw-7fa13-7.闭式水泵.C.mid-260824.1634-normal-0001.wav",
                "pw-7fa13-7.闭式水泵.C.mid-260824.1634-normal-0002.wav",
                "pw-7fa13-7.闭式水泵.C.mid-260824.1635-normal-0001.wav",
            ],
        )

    def test_late_conflict_prevents_all_writes_and_overwrite_is_explicit(self) -> None:
        """验证最后一个目标重名时前面的切片也不会写入，显式覆盖可完成重跑。"""
        self.write_source("sync_20260824_163405_7fa13.wav", np.zeros(320000))
        config = self.config()
        config.output_dir.mkdir()
        existing = (
            config.output_dir / "pw-7fa13-7.闭式水泵.C.mid-260824.1634-normal-0002.wav"
        )
        existing.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            run(config)
        self.assertEqual(list(config.output_dir.iterdir()), [existing])
        self.assertEqual(existing.read_bytes(), b"original")
        self.assertEqual(run(self.config(overwrite=True))["written_clips"], 2)
        self.assertEqual(sf.info(existing).frames, 160000)

    def test_dry_run_and_relative_paths(self) -> None:
        """验证路径相对于 YAML 定位，预览仅返回数量且不创建输出目录。"""
        self.write_source("sync_20260824_163405_7fa13.wav", np.zeros(160000))
        config = self.config(dry_run=True)
        self.assertEqual(config.input_dir, self.input_dir)
        self.assertEqual(config.output_dir, self.root / "output")
        self.assertEqual(
            run(config), {"source_files": 1, "planned_clips": 1, "written_clips": 0}
        )
        self.assertFalse(config.output_dir.exists())

    def test_invalid_configuration_is_rejected(self) -> None:
        """验证错误类型、无效时长采样率、列号、尾段策略、路径字段和未知参数均报错。"""
        for overrides in (
            {"target_sample_rate": 0},
            {"target_sample_rate": True},
            {"clip_seconds": -1},
            {"clip_seconds": float("nan")},
            {"clip_seconds": 0.00001},
            {"clip_seconds": True},
            {"date_column": 0},
            {"sequence_width": 0},
            {"sequence_start": -1},
            {"extensions": "wav"},
            {"extensions": []},
            {"extensions": [None]},
            {"mono": "true"},
            {"recursive": 1},
            {"tail_policy": "keep"},
            {"wav_subtype": "INVALID"},
            {"dataset": "bad/name"},
            {"filename_separator": ""},
            {"unknown": 1},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.config(**overrides)

    def test_invalid_timestamp_aborts_before_output(self) -> None:
        """验证不完整时间和无效日期均在预检失败，且不会留下此前源文件的切片。"""
        self.write_source("sync_20260824_163405_7fa13.wav", np.zeros(160000))
        for name in (
            "sync_20260824_164_7fa13.wav",
            "sync_20260230_163405_7fa13.wav",
            "sync_7fa13.wav",
        ):
            invalid = self.write_source(name, np.zeros(160000))
            with self.subTest(name=name), self.assertRaises(ValueError):
                run(self.config())
            self.assertFalse((self.root / "output").exists())
            invalid.unlink()

    def test_antialias_filter_and_preserved_stereo(self) -> None:
        """验证保留双声道时沿时间轴重采样，1 kHz 信号保留而 12 kHz 被抗混叠滤除。"""
        time = np.arange(480000) / 48000
        source = np.column_stack(
            (
                0.4 * np.sin(2 * np.pi * 1000 * time),
                0.4 * np.sin(2 * np.pi * 12000 * time),
            )
        )
        self.write_source("sync_20260824_163405_7fa13.wav", source, 48000)
        config = self.config(mono=False, wav_subtype="FLOAT")
        run(config)
        audio, rate = sf.read(next(config.output_dir.glob("*.wav")))
        self.assertEqual((rate, audio.shape), (16000, (160000, 2)))
        rms = np.sqrt(np.mean(audio[100:-100] ** 2, axis=0))
        self.assertAlmostEqual(rms[0], 0.4 / np.sqrt(2), delta=0.001)
        self.assertLess(rms[1], 0.001)

    def test_custom_yaml_and_noninteger_resampling_ratio(self) -> None:
        """验证 44.1 kHz 转 8 kHz、2 秒分段、自定义命名和日期列号均由 YAML 生效。"""
        self.write_source(
            "capture_235959_20260824_sensor.wav", np.full(44100 * 4 + 1, 0.25), 44100
        )
        config = self.config(
            filename_prefix="capture",
            filename_suffix="sensor",
            target_sample_rate=8000,
            clip_seconds=2,
            dataset="custom",
            collection_device="sensor",
            target_device="motor.mid",
            label="abnormal",
            date_column=3,
            time_column=2,
            sequence_start=7,
            sequence_width=3,
        )
        self.assertEqual(run(config)["written_clips"], 2)
        self.assertEqual(
            sorted(path.name for path in config.output_dir.glob("*.wav")),
            [
                "custom-sensor-motor.mid-260824.2359-abnormal-007.wav",
                "custom-sensor-motor.mid-260824.2359-abnormal-008.wav",
            ],
        )
        for path in config.output_dir.glob("*.wav"):
            self.assertEqual(
                (sf.info(path).frames, sf.info(path).samplerate), (16000, 8000)
            )

    def test_invalid_audio_and_changed_metadata(self) -> None:
        """验证浮点 NaN 及预检后源音频长度变化会被拒绝，防止输出不可信数据。"""
        path = self.write_source(
            "sync_20260824_163405_7fa13.wav", np.full(160000, np.nan)
        )
        config = self.config()
        with self.assertRaises(ValueError):
            run(config)
        self.assertFalse(list(config.output_dir.glob("*.wav")))
        sf.write(path, np.zeros(160000), 16000)
        plan = build_plan(config)[0]
        sf.write(path, np.zeros(160001), 16000)
        with self.assertRaises(ValueError):
            load_resampled_audio(plan, config)

    def test_no_matches_and_recursive_selection(self) -> None:
        """验证无匹配输入时报错，启用递归后才会选择子目录内满足条件的 WAV。"""
        config = self.config()
        with self.assertRaises(ValueError):
            run(config)
        nested = self.input_dir / "nested"
        nested.mkdir()
        sf.write(nested / "sync_20260824_163405_7fa13.wav", np.zeros(160000), 16000)
        with self.assertRaises(ValueError):
            run(config)
        self.assertEqual(run(self.config(recursive=True))["written_clips"], 1)

    def test_cli_from_another_working_directory(self) -> None:
        """从另一工作目录调用真实 CLI，验证配置定位、预览及无效配置的退出状态。"""
        self.write_source("sync_20260824_163405_7fa13.wav", np.zeros(160000))
        config = self.config(dry_run=True)
        command = [
            sys.executable,
            str(DEFAULT_CONFIG.with_suffix(".py")),
            "--config",
            str(self.config_path),
        ]
        result = subprocess.run(command, cwd=self.root, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(config.output_dir.exists())
        self.config_path.write_text("unknown: 1\n", encoding="utf-8")
        result = subprocess.run(command, cwd=self.root, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
