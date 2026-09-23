"""WAV 文件和 WebSocket PCM 的十秒窗口输入。"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LOGGER = logging.getLogger(__name__)
TARGET_RATE = 16000
WINDOW_SECONDS = 10
WINDOW_SAMPLES = TARGET_RATE * WINDOW_SECONDS
PCM_DTYPES = {"s16le": np.dtype("<i2"), "f32le": np.dtype("<f4")}


@dataclass(frozen=True)
class AudioWindow:
    """十秒单声道浮点波形，携带来源、窗口索引及补零前有效时长。"""

    waveform: np.ndarray
    source: str
    index: int
    valid_seconds: float


def make_window(
    samples: np.ndarray, sample_rate: int, source: str, index: int, tail_policy: str
) -> AudioWindow | None:
    """将最多十秒的单/多声道样本混合、重采样为 16 kHz 并按策略处理尾段。

    samples 的形状为 [采样点] 或 [采样点, 声道]。空、非有限、超长或不合法
    输入抛出 ValueError；drop 尾段返回 None；pad 尾段保留原有效时长。
    """
    samples = np.asarray(samples, dtype=np.float32)
    if (
        type(sample_rate) is not int
        or sample_rate <= 0
        or samples.ndim not in (1, 2)
        or samples.size == 0
        or not np.isfinite(samples).all()
    ):
        raise ValueError("音频须为非空有限波形，采样率须为正整数")
    if tail_policy not in {"pad", "drop", "error"}:
        raise ValueError("未知音频尾段策略")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    valid_seconds = len(samples) / sample_rate
    if len(samples) > sample_rate * WINDOW_SECONDS:
        raise ValueError("单次输入超过 10 秒，应先按窗口分割")
    if valid_seconds < WINDOW_SECONDS:
        if tail_policy == "drop":
            LOGGER.info("丢弃不足十秒的尾段：%s，时长 %.3f 秒", source, valid_seconds)
            return None
        if tail_policy == "error":
            raise ValueError(f"音频尾段不足十秒：{source}，{valid_seconds:.3f} 秒")
    if sample_rate != TARGET_RATE:
        divisor = gcd(sample_rate, TARGET_RATE)
        samples = resample_poly(samples, TARGET_RATE // divisor, sample_rate // divisor)
    samples = np.pad(
        samples[:WINDOW_SAMPLES], (0, max(0, WINDOW_SAMPLES - len(samples)))
    )
    return AudioWindow(
        np.ascontiguousarray(samples, dtype=np.float32), source, index, valid_seconds
    )


def iter_file_windows(config: dict) -> Iterator[AudioWindow]:
    """按文件名顺序读取 WAV 文件或目录，逐个产生十秒窗口且不整文件加载。

    支持大小写 WAV 扩展名和可选递归；空目录、空 WAV 或损坏文件报错。
    """
    source = Path(config["path"])
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        candidates = source.rglob("*") if config["recursive"] else source.glob("*")
        paths = sorted(
            p for p in candidates if p.is_file() and p.suffix.lower() == ".wav"
        )
    else:
        raise FileNotFoundError(f"输入路径不存在：{source}")
    if not paths:
        raise ValueError(f"输入目录中没有 WAV 文件：{source}")
    for path in paths:
        if path.suffix.lower() != ".wav":
            raise ValueError(f"文件输入只支持 WAV：{path}")
        with sf.SoundFile(path) as audio:
            if not audio.frames:
                raise ValueError(f"WAV 文件为空：{path}")
            for index, samples in enumerate(
                audio.blocks(
                    blocksize=audio.samplerate * WINDOW_SECONDS,
                    dtype="float32",
                    always_2d=True,
                )
            ):
                window = make_window(
                    samples, audio.samplerate, str(path), index, config["tail_policy"]
                )
                if window is not None:
                    yield window


class PCMBuffer:
    """跨 WebSocket 消息保留字节尾部，按原始采样率和声道数切十秒 PCM。"""

    def __init__(self, sample_rate: int, channels: int, pcm_format: str):
        """指定原始 PCM 采样率、交错声道数和小端格式，初始化字节缓冲区。"""
        if sample_rate <= 0 or channels <= 0 or pcm_format not in PCM_DTYPES:
            raise ValueError("PCM 采样率、声道数或格式无效")
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = PCM_DTYPES[pcm_format]
        self.frame_bytes = channels * self.dtype.itemsize
        self.window_bytes = sample_rate * WINDOW_SECONDS * self.frame_bytes
        self.buffer = bytearray()

    def feed(self, message: bytes) -> Iterator[np.ndarray]:
        """接收二进制 PCM 消息并逐个产出完整窗口，支持跨字节/采样点分包。"""
        if not isinstance(message, bytes):
            raise TypeError("PCM 服务必须返回原始二进制消息，不能是文本或 JSON")
        self.buffer.extend(message)
        while len(self.buffer) >= self.window_bytes:
            chunk = bytes(self.buffer[: self.window_bytes])
            del self.buffer[: self.window_bytes]
            yield self._decode(chunk)

    def _decode(self, chunk: bytes) -> np.ndarray:
        """将完整交错 PCM 帧解码为 float32；16 位整数缩放到 [-1,1)。"""
        samples = np.frombuffer(chunk, dtype=self.dtype).astype(np.float32)
        if self.dtype.kind == "i":
            samples /= 32768.0
        if not np.isfinite(samples).all():
            raise ValueError("PCM 包含 NaN 或无穷值")
        return samples.reshape(-1, self.channels)

    def finish(self) -> np.ndarray | None:
        """连接正常关闭时取出尾段；不完整 PCM 帧报错，避免静默丢失字节。"""
        if len(self.buffer) % self.frame_bytes:
            raise ValueError("PCM 流在采样帧中间结束，字节数与格式不匹配")
        if not self.buffer:
            return None
        chunk = bytes(self.buffer)
        self.buffer.clear()
        return self._decode(chunk)


def iter_websocket_windows(config: dict) -> Iterator[AudioWindow]:
    """连接 PCM 服务，发送可选请求并连续产出窗口；超时和异常断连向上报错。

    仅正常关闭按 tail_policy 处理尾段；不自动重连，避免跨连接拼接音频。
    max_windows 达到上限后主动关闭；零表示不限制窗口数。
    """
    from websockets.exceptions import ConnectionClosedOK
    from websockets.sync.client import connect

    ws = config["websocket"]
    buffer = PCMBuffer(ws["sample_rate"], ws["channels"], ws["format"])
    index = 0
    with connect(
        ws["url"],
        open_timeout=ws["timeout_seconds"],
        close_timeout=ws["timeout_seconds"],
        max_size=ws["max_message_bytes"],
    ) as socket:
        LOGGER.info("已连接 PCM 服务：%s", ws["url"])
        if ws["request"] is not None:
            socket.send(ws["request"])
        while True:
            try:
                message = socket.recv(timeout=ws["timeout_seconds"])
            except ConnectionClosedOK:
                break
            for samples in buffer.feed(message):
                yield make_window(samples, ws["sample_rate"], ws["url"], index, "error")
                index += 1
                if ws["max_windows"] and index >= ws["max_windows"]:
                    return
                if ws["request_each_window"]:
                    socket.send(ws["request"])
        tail = buffer.finish()
        if tail is not None:
            window = make_window(
                tail, ws["sample_rate"], ws["url"], index, config["tail_policy"]
            )
            if window is not None:
                yield window
