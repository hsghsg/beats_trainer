"""ASD 推理系统公共接口，逐窗口产出 Result。"""

import logging
from collections.abc import Iterator
from pathlib import Path

from .audio import AudioWindow, iter_file_windows, iter_websocket_windows, make_window
from .config import load_config
from .models import BackendRunner
from .result import Result, decide

LOGGER = logging.getLogger(__name__)


class ASDInference:
    """从 YAML 初始化模型，并对文件、PCM 服务或内存音频执行异常检测。"""

    def __init__(self, config_path: str | Path):
        """读取唯一配置文件并加载所有启用分类器；配置或模型错误直接抛出。"""
        self.config = load_config(config_path)
        self.runner = BackendRunner(self.config)

    def predict_window(self, window: AudioWindow) -> Result:
        """对十秒 AudioWindow 推理、投票，并记录每个后端及最终标签和阈值。"""
        scores = self.runner.score(window.waveform)
        thresholds = {
            name: value["threshold"]
            for name, value in self.config["classifiers"].items()
            if value["enabled"]
        }
        ensemble = self.config["ensemble"]
        result = decide(
            scores,
            thresholds,
            ensemble["min_anomalous"],
            ensemble["normal_label"],
            ensemble["abnormal_label"],
            window.source,
            window.index,
            window.index * 10.0,
            window.valid_seconds,
        )
        for backend in result.classifiers:
            LOGGER.info(
                "来源=%s 窗口=%d 分类器=%s 标签=%s 分数=%.6f 阈值=%.6f",
                result.source,
                result.window_index,
                backend.name,
                backend.label,
                backend.score,
                backend.threshold,
            )
        LOGGER.info(
            "来源=%s 窗口=%d 最终标签=%s 异常票数=%d 最终阈值=%d",
            result.source,
            result.window_index,
            result.label,
            result.score,
            result.threshold,
        )
        return result

    def predict_array(self, waveform, sample_rate: int = 16000) -> Result:
        """推理恰好十秒的内存波形，支持单声道或 [采样点,声道] 数组并返回 Result。"""
        window = make_window(waveform, sample_rate, "array", 0, "error")
        return self.predict_window(window)

    def run(self) -> Iterator[Result]:
        """按 YAML 选择文件或 WebSocket 输入，逐窗口返回 Result，结束时关闭输入。"""
        config = self.config["input"]
        windows = (
            iter_file_windows(config)
            if config["mode"] == "file"
            else iter_websocket_windows(config)
        )
        try:
            for window in windows:
                yield self.predict_window(window)
        finally:
            windows.close()
