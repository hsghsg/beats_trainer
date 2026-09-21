"""验证 CUMTB CSV 声道提取、重采样、输出保护和异常处理。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.convert_cumtb_acoustic import convert_file, load_acoustic, run


class CumtbConversionTests(unittest.TestCase):
    """用可控 CSV 覆盖实际格式和重采样关键行为。"""

    def setUp(self):
        """建立临时输入目录，注册自动清理以避免污染真实数据。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input"
        self.source.mkdir()

    def write_csv(self, samples, name="Cond_1/1rpm/Health/va (1).csv", start=1):
        """生成无表头六列 CSV 和末尾空列，振动列使用明显不同的常数。"""
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = np.column_stack((
            np.arange(start, start + len(samples)),
            np.full((len(samples), 4), 99), samples,
        ))
        np.savetxt(path, rows, delimiter=",", fmt="%.12g", newline=",\n")
        return path

    def test_column_and_first_row(self):
        """验证取第六列而非振动列或末尾空列，且保留第一行和非 1 起始序号。"""
        source = self.write_csv(np.array([0.2, -0.3, 0.4]), start=1000001)
        samples, first, last = load_acoustic(source)
        np.testing.assert_allclose(samples, [0.2, -0.3, 0.4])
        self.assertEqual((first, last), (1000001, 1000003))

    def test_frequency_amplitude_and_antialias(self):
        """验证 38.5 kHz 转 16 kHz 后时长、低频幅值保留和高频抗混叠。"""
        time = np.arange(38500) / 38500
        samples = 2 * np.sin(2 * np.pi * 1000 * time)
        samples += np.sin(2 * np.pi * 12000 * time)
        source = self.write_csv(samples)
        destination = self.root / "out.wav"
        record = convert_file(source, destination, 38500, "FLOAT")
        audio, rate = sf.read(destination)
        self.assertEqual((len(audio), rate), (16000, 16000))
        self.assertEqual(record["gain_applied"], 1)
        spectrum = np.abs(np.fft.rfft(audio)) * 2 / len(audio)
        self.assertEqual(np.argmax(spectrum), 1000)
        self.assertAlmostEqual(spectrum[1000], 2, delta=0.01)
        self.assertLess(spectrum[4000], 0.01)

    def test_pcm_silence_and_rounding(self):
        """验证 PCM 归一化、静音处理及非整数输出长度向上取整。"""
        for samples in (np.zeros(1001), np.linspace(-3, 3, 1001)):
            source = self.write_csv(samples)
            destination = self.root / "pcm.wav"
            convert_file(source, destination, 38500, "PCM_16")
            audio, _ = sf.read(destination)
            self.assertEqual(len(audio), (1001 * 16000 + 38499) // 38500)
            self.assertEqual(sf.info(destination).subtype, "PCM_16")
            expected = 0 if np.all(samples == 0) else 0.99
            self.assertAlmostEqual(np.max(np.abs(audio)), expected, delta=1 / 32768)

    def test_invalid_csv(self):
        """拒绝缺少声音列、序号跳跃和非有限声音数据。"""
        path = self.source / "bad.csv"
        for text in (
            "1,2,3\n2,3,4\n",
            "1,0,0,0,0,0.1,\n3,0,0,0,0,0.2,\n",
            "1,0,0,0,0,nan,\n2,0,0,0,0,0.2,\n",
        ):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_acoustic(path)

    def test_batch_hierarchy_skip_and_failure(self):
        """验证目录保留、同名跨类别不冲突、失败继续、已有输出保护及覆盖。"""
        samples = np.linspace(-0.1, 0.1, 1000)
        self.write_csv(samples)
        self.write_csv(samples, "Cond_2/3rpm/IRC/va (1).csv")
        bad = self.source / "bad.csv"
        bad.write_text("bad\n", encoding="utf-8")
        output = self.root / "output"
        self.assertEqual(run(self.source, output, workers=2), 1)
        self.assertEqual(len(list(output.rglob("*.wav"))), 2)
        bad.unlink()
        wav = output / "Cond_1/1rpm/Health/va (1).wav"
        original = wav.read_bytes()
        self.assertEqual(run(self.source, output), 0)
        self.assertEqual(wav.read_bytes(), original)
        self.assertEqual(run(self.source, output, subtype="PCM_16"), 1)
        self.assertEqual(run(self.source, output, subtype="PCM_16", overwrite=True), 0)
        self.assertEqual(sf.info(wav).subtype, "PCM_16")
        records = [
            json.loads(line) for report in output.glob("conversion_*.jsonl")
            for line in report.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            {item["status"] for item in records}, {"converted", "failed", "skipped"},
        )


if __name__ == "__main__":
    unittest.main()

