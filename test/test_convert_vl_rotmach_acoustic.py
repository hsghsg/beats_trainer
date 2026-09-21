"""验证 VL-RotMach 转换的频率、抗混叠、声压、异常输入及输出保护。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import savemat

from scripts.convert_vl_rotmach_acoustic import convert_file, load_acoustic, run


class AcousticConversionTests(unittest.TestCase):
    """使用真实格式的合成 MAT 验证转换，不依赖外部数据集。"""

    def setUp(self):
        """建立临时目录和 51.2 kHz 时间轴，测试结束自动清理。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.time = np.arange(51200) / 51200

    def write_mat(self, samples, name="sample.mat", count=None):
        """按 Test.Lab Signal 结构写入声压及元数据，返回测试文件路径。"""
        path = self.input_dir / name
        savemat(path, {"Signal": {
            "x_values": {
                "increment": 1 / 51200, "start_value": 0.001,
                "number_of_values": len(samples) if count is None else count,
                "quantity": {"label": "s"},
            },
            "y_values": {"values": samples, "quantity": {"label": "Pa"}},
        }})
        return path

    def test_float_frequency_amplitude_and_antialias(self):
        """验证 1 kHz 声压和时长保留，并抑制超过 8 kHz 的 12 kHz 信号。"""
        samples = 2 * np.sin(2 * np.pi * 1000 * self.time)
        samples += np.sin(2 * np.pi * 12000 * self.time)
        source = self.write_mat(samples)
        output = self.root / "float.wav"
        metadata = convert_file(source, output, "FLOAT")
        audio, rate = sf.read(output)
        self.assertEqual((rate, audio.size), (16000, 16000))
        self.assertEqual(sf.info(output).subtype, "FLOAT")
        self.assertEqual(metadata["gain_applied"], 1)
        self.assertEqual(metadata["source_start_seconds"], 0.001)
        spectrum = np.abs(np.fft.rfft(audio)) * 2 / audio.size
        self.assertEqual(np.argmax(spectrum), 1000)
        self.assertAlmostEqual(spectrum[1000], 2, delta=0.01)
        self.assertLess(spectrum[4000], 0.01)

    def test_pcm_and_silence(self):
        """验证 PCM 峰值缩放不会削波，静音不会除零或产生非零输出。"""
        for name, samples in (
            ("tone.mat", 3 * np.sin(2 * np.pi * 1000 * self.time)),
            ("silent.mat", np.zeros(51200)),
        ):
            source = self.write_mat(samples, name)
            output = self.root / (name + ".wav")
            metadata = convert_file(source, output, "PCM_16")
            audio, _ = sf.read(output)
            self.assertEqual(sf.info(output).subtype, "PCM_16")
            if name == "silent.mat":
                self.assertTrue(np.all(audio == 0))
                self.assertEqual(metadata["gain_applied"], 1)
            else:
                self.assertAlmostEqual(np.max(np.abs(audio)), 0.99, delta=1 / 32768)

    def test_invalid_samples_and_count(self):
        """拒绝非有限声压及样本数不一致的元数据，避免产生错误音频。"""
        source = self.write_mat(np.array([0.0, np.nan]))
        with self.assertRaises(ValueError):
            load_acoustic(source)
        source = self.write_mat(np.zeros(10), count=20)
        with self.assertRaises(ValueError):
            load_acoustic(source)

    def test_batch_skip_failure_and_overwrite(self):
        """验证错误文件不阻断后续文件、已有输出保护及显式覆盖功能。"""
        self.write_mat(np.sin(2 * np.pi * 1000 * self.time))
        bad = self.write_mat(np.zeros(5), "bad.mat", count=8)
        output_dir = self.root / "output"
        self.assertEqual(run(self.input_dir, output_dir, "FLOAT", False), 1)
        output = output_dir / "sample.wav"
        before = output.read_bytes()
        bad.unlink()
        self.assertEqual(run(self.input_dir, output_dir, "PCM_16", False), 0)
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(run(self.input_dir, output_dir, "PCM_16", True), 0)
        self.assertEqual(sf.info(output).subtype, "PCM_16")
        records = [
            record for path in output_dir.glob("conversion_*.json")
            for record in json.loads(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(
            {record["status"] for record in records}, {"converted", "failed", "skipped"},
        )


if __name__ == "__main__":
    unittest.main()

