"""ASD 输入、配置、异常投票及模型兼容性测试。"""

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from asd.audio import PCMBuffer, iter_file_windows, iter_websocket_windows, make_window
from asd.config import load_config
from asd.frontend import Frontend
from asd.models import BackendRunner, BEATsModel
from asd.result import decide
from asd.vendor.BEATs.BEATs import BEATs, BEATsConfig
from asd.vendor.beats_trainer.classifiers import (
    GMMCosineKNNHybridClassifier,
    LocalDensityKNNClassifier,
    RelativeMahalanobisDistanceClassifier,
)

ASD_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    """返回独立配置副本，供测试修改阈值或输入选项。"""
    return load_config(ASD_ROOT / "infer.yaml")


def test_threshold_equality_and_votes():
    """验证等于阈值计为异常，并且最终阈值记录为所需票数。"""
    result = decide(
        {"a": 0.5, "b": 0.7, "c": 0.8, "d": 0.2},
        {"a": 0.5, "b": 0.8, "c": 0.75, "d": 0.3},
        2,
        "normal",
        "abnormal",
    )
    assert (result.label, result.score, result.threshold) == ("abnormal", 2, 2)
    assert [item.label for item in result.classifiers] == [
        "abnormal",
        "normal",
        "abnormal",
        "normal",
    ]
    assert (
        json.loads(json.dumps(result.to_dict()))["classifiers"][1]["threshold"] == 0.8
    )
    assert decide({"a": 0.4}, {"a": 0.5}, 1, "normal", "abnormal").label == "normal"


@pytest.mark.parametrize(
    "score,threshold,votes",
    [
        (float("nan"), 0.5, 1),
        (0.5, float("inf"), 1),
        (1.1, 0.5, 1),
        (0.5, -0.1, 1),
        (0.5, 0.5, 0),
        (0.5, 0.5, 2),
        (0.5, 0.5, True),
    ],
)
def test_invalid_decision(score, threshold, votes):
    """拒绝非有限分数、越界阈值和不可能的投票数量。"""
    with pytest.raises(ValueError):
        decide({"a": score}, {"a": threshold}, votes, "normal", "abnormal")


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("input", "duration_seconds", 5),
        ("input", "sample_rate", 8000),
        ("input", "tail_policy", "truncate"),
        ("frontend", "method", "unknown"),
        ("frontend", "fbank_std", 0),
        ("frontend", "cwt_time_stride", 0),
        ("ensemble", "min_anomalous", 5),
        ("embedding", "normalize", "true"),
    ],
)
def test_invalid_config(tmp_path, section, key, value):
    """配置错误应在读取时拒绝，不等到加载大模型后才失败。"""
    data = yaml.safe_load((ASD_ROOT / "infer.yaml").read_text(encoding="utf-8-sig"))
    data[section][key] = value
    filename = tmp_path / "infer.yaml"
    filename.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(filename)


def test_config_relative_paths(tmp_path):
    """验证相对路径以配置目录为基准，且所有后端禁用时校验失败。"""
    data = yaml.safe_load((ASD_ROOT / "infer.yaml").read_text(encoding="utf-8-sig"))
    filename = tmp_path / "infer.yaml"
    filename.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(filename)["input"]["path"] == tmp_path / "audio"
    for backend in data["classifiers"].values():
        backend["enabled"] = False
    filename.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(filename)


def test_resampling_and_tail():
    """验证多声道平均、48 kHz 重采样及 pad/drop/error 三种尾段行为。"""
    samples = np.ones((48000, 2), dtype=np.float32)
    samples[:, 1] = -1
    window = make_window(samples, 48000, "test", 0, "pad")
    assert window.waveform.shape == (160000,)
    assert window.valid_seconds == 1
    assert np.max(np.abs(window.waveform)) == 0
    assert make_window(samples, 48000, "test", 0, "drop") is None
    with pytest.raises(ValueError):
        make_window(samples, 48000, "test", 0, "error")
    with pytest.raises(ValueError):
        make_window(np.zeros(160001), 16000, "test", 0, "pad")
    with pytest.raises(ValueError):
        make_window(np.array([np.nan]), 16000, "test", 0, "pad")


def test_file_directory_windows(config, tmp_path):
    """验证递归 WAV、大小写扩展名、长音频切片及有效尾段时间。"""
    (tmp_path / "child").mkdir()
    sf.write(tmp_path / "child" / "a.WAV", np.zeros(21000), 1000, format="WAV")
    config["input"]["path"] = tmp_path
    windows = list(iter_file_windows(config["input"]))
    assert [window.valid_seconds for window in windows] == [10, 10, 1]
    assert [window.index for window in windows] == [0, 1, 2]
    assert all(window.waveform.shape == (160000,) for window in windows)
    config["input"]["recursive"] = False
    with pytest.raises(ValueError):
        list(iter_file_windows(config["input"]))


@pytest.mark.parametrize("pcm_format,dtype", [("s16le", "<i2"), ("f32le", "<f4")])
def test_pcm_arbitrary_fragmentation(pcm_format, dtype):
    """跨任意字节边界拼接 PCM，覆盖多声道和单消息包含多个窗口。"""
    values = np.arange(42, dtype=np.float32).reshape(21, 2) / 100
    wire_values = (
        (values * 32768).astype(dtype)
        if pcm_format == "s16le"
        else values.astype(dtype)
    )
    raw = wire_values.tobytes()
    buffer = PCMBuffer(1, 2, pcm_format)
    decoded = []
    for start in range(0, len(raw), 3):
        decoded.extend(buffer.feed(raw[start : start + 3]))
    assert len(decoded) == 2
    decoded.append(buffer.finish())
    expected = wire_values.astype(np.float32)
    if pcm_format == "s16le":
        expected /= 32768
    np.testing.assert_array_equal(np.concatenate(decoded), expected)
    assert buffer.finish() is None


def test_pcm_bad_packets():
    """文本消息、非有限浮点数和连接尾部不完整帧均应显式报错。"""
    buffer = PCMBuffer(1, 1, "s16le")
    with pytest.raises(TypeError):
        list(buffer.feed("json"))
    list(buffer.feed(b"x"))
    with pytest.raises(ValueError):
        buffer.finish()
    buffer = PCMBuffer(1, 1, "f32le")
    with pytest.raises(ValueError):
        list(buffer.feed(np.full(10, np.nan, dtype="<f4").tobytes()))


def test_frontend_matches_beats(config):
    """log-mel 特征经 BEATs 标准化后须与原生波形预处理数值一致。"""
    torch.manual_seed(7)
    waveform = torch.randn(160000) * 0.01
    front = Frontend(config["frontend"])
    features = front(waveform, 16)
    original = BEATs.preprocess(None, waveform.unsqueeze(0))
    actual = (features - config["frontend"]["fbank_mean"]) / (
        2 * config["frontend"]["fbank_std"]
    )
    torch.testing.assert_close(actual, original)


def test_wavelet_and_token_guard(config):
    """验证 CWT 时频路径、显式抽样和超长序列保护。"""
    config["frontend"]["method"] = "wavelet"
    front = Frontend(config["frontend"])
    with pytest.raises(ValueError, match="token"):
        front(np.zeros(160000, dtype=np.float32), 16)
    config["frontend"]["cwt_time_stride"] = 160
    assert front(np.zeros(160000, dtype=np.float32), 16).shape == (1, 1000, 91)


def make_tiny_checkpoint(path: Path, custom_head: bool = False) -> dict:
    """创建小型真实 BEATs checkpoint，包含可严格加载的骨干和线性或 MLP 头。"""
    torch.manual_seed(8)
    cfg = {
        "encoder_layers": 1,
        "encoder_embed_dim": 16,
        "encoder_ffn_embed_dim": 32,
        "encoder_attention_heads": 2,
        "embed_dim": 16,
        "input_patch_size": 16,
        "conv_pos": 16,
        "conv_pos_groups": 2,
        "relative_position_embedding": True,
        "num_buckets": 32,
        "max_distance": 64,
        "gru_rel_pos": True,
        "deep_norm": True,
    }
    backbone = BEATs(BEATsConfig(cfg))
    head = (
        torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(8, 2),
        )
        if custom_head
        else torch.nn.Linear(16, 2)
    )
    state = {f"backbone.{key}": value for key, value in backbone.state_dict().items()}
    state.update(
        {f"classifier.{key}": value for key, value in head.state_dict().items()}
    )
    params = dict(
        cfg,
        num_classes=2,
        use_custom_head=custom_head,
        hidden_dims=[8],
        activation="relu",
        dropout_rate=0.1,
    )
    torch.save(
        {
            "state_dict": state,
            "hyper_parameters": {"config": {"model": params}, "num_classes": 2},
        },
        path,
    )
    return cfg


@pytest.mark.parametrize("custom_head", [False, True])
def test_native_head_checkpoint(tmp_path, config, custom_head):
    """加载线性/MLP 原生头，确保严格参数重建且拒绝缺失实际骨干权重。"""
    filename = tmp_path / "model.ckpt"
    cfg = make_tiny_checkpoint(filename, custom_head)
    model = BEATsModel(filename, torch.device("cpu"), cfg, need_native=True)
    features = Frontend(config["frontend"])(np.zeros(160000, dtype=np.float32), 16)
    with torch.inference_mode():
        assert model.head(model.embedding(features, config["frontend"])).shape == (1, 2)
    checkpoint = torch.load(filename, weights_only=False)
    del checkpoint["state_dict"]["backbone.patch_embedding.weight"]
    torch.save(checkpoint, filename)
    with pytest.raises((ValueError, KeyError)):
        BEATsModel(filename, torch.device("cpu"), cfg)


def test_pretrained_checkpoint(tmp_path):
    """验证原始 cfg/model 格式的 BEATs 权重可离线加载为 embedding 模型。"""
    filename = tmp_path / "model.ckpt"
    cfg = make_tiny_checkpoint(filename)
    checkpoint = torch.load(filename, weights_only=False)
    state = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("backbone.")
    }
    torch.save({"cfg": cfg, "model": state}, filename)
    assert BEATsModel(filename, torch.device("cpu"), {}).head is None
    with pytest.raises(ValueError, match="Lightning"):
        BEATsModel(filename, torch.device("cpu"), {}, need_native=True)


@pytest.mark.parametrize("method", ["log-mel-energy", "wavelet"])
def test_four_backends_and_label_order(tmp_path, config, method):
    """用真实小型 BEATs 和三个训练后端跑完整推理，校验异常类按标签定位。"""
    config["frontend"].update(method=method, cwt_time_stride=160)
    checkpoint = tmp_path / "tiny.ckpt"
    cfg = make_tiny_checkpoint(checkpoint)
    config["embedding"].update(
        checkpoint=checkpoint, device="cpu", backbone_overrides=cfg
    )
    config["classifiers"]["beats-native"].update(
        checkpoint=checkpoint, labels=["normal", "abnormal"]
    )
    rng = np.random.default_rng(8)
    features = rng.normal(size=(24, 16))
    labels = np.asarray(["normal"] * 12 + ["abnormal"] * 12)
    classifiers = {
        "local-density-knn": LocalDensityKNNClassifier(k=3, density_k=3),
        "relative-mahalanobis": RelativeMahalanobisDistanceClassifier(),
        "gmm-cosine-knn": GMMCosineKNNHybridClassifier(n_components=1, knn_k=3),
    }
    for name, classifier in classifiers.items():
        path = tmp_path / f"{name}.pkl"
        classifier.fit(features, labels).save(path)
        config["classifiers"][name]["path"] = path
    runner = BackendRunner(config)
    assert len(runner.models) == 1
    waveform = np.zeros(160000, dtype=np.float32)
    scores = runner.score(waveform)
    assert set(scores) == set(config["classifiers"])
    with torch.inference_mode():
        model = runner.models[checkpoint]
        pooled = model.embedding(runner.frontend(waveform, 16), config["frontend"])
        expected = torch.softmax(model.head(pooled), dim=-1)[0, 1].item()
    assert scores["beats-native"] == pytest.approx(expected)
    config["classifiers"]["relative-mahalanobis"]["path"] = config["classifiers"][
        "local-density-knn"
    ]["path"]
    with pytest.raises(TypeError):
        BackendRunner(config)


def test_websocket_local_server(config):
    """启动本地 PCM 服务，验证路径、请求文本、奇数字节分包及正常关闭尾段。"""
    from websockets.sync.server import serve

    requests = []

    def handler(connection):
        """记录客户端路径与首条请求，再发送十秒窗口加一秒尾段并正常关闭。"""
        requests.append((connection.request.path, connection.recv()))
        payload = np.ones(11000, dtype="<i2").tobytes()
        connection.send(payload[:3])
        connection.send(payload[3:])

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        config["input"]["websocket"].update(
            url=f"ws://127.0.0.1:{port}/pcmData", request="pcmData", sample_rate=1000
        )
        try:
            windows = list(iter_websocket_windows(config["input"]))
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert requests == [("/pcmData", "pcmData")]
    assert [item.valid_seconds for item in windows] == [10, 1]


def test_standalone_copy(tmp_path):
    """仅复制 ASD 源码到另一目录，运行真实微型模型推理，证明无需父项目。"""
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    target = deployment / "asd"
    shutil.copytree(
        ASD_ROOT,
        target,
        ignore=shutil.ignore_patterns(
            "models", ".test-deps", "tests", "__pycache__", "output", ".pytest_cache"
        ),
    )
    checkpoint = deployment / "tiny.ckpt"
    cfg = make_tiny_checkpoint(checkpoint)
    audio = deployment / "input.wav"
    sf.write(audio, np.zeros(160000), 16000)
    config = yaml.safe_load((target / "infer.yaml").read_text(encoding="utf-8-sig"))
    config["input"]["path"] = str(audio)
    config["embedding"].update(
        checkpoint=str(checkpoint), device="cpu", backbone_overrides=cfg
    )
    for name, backend in config["classifiers"].items():
        backend["enabled"] = name == "beats-native"
    config["classifiers"]["beats-native"]["checkpoint"] = str(checkpoint)
    config["ensemble"]["min_anomalous"] = 1
    config["output"].update(jsonl_path=None, log_path=None)
    (target / "infer.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(target / "infer.py")],
        cwd=deployment,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    result = json.loads(completed.stdout)
    assert result["threshold"] == 1
    assert len(result["classifiers"]) == 1


@pytest.mark.parametrize("failure", ["text", "disconnect", "timeout"])
def test_websocket_failures(config, failure):
    """文本响应、异常关闭和接收超时必须传播失败，不输出伪造的正常结果。"""
    from websockets.exceptions import ConnectionClosedError
    from websockets.sync.server import serve

    finish = threading.Event()

    def handler(connection):
        """按测试参数发送非法响应、异常关闭或等待客户端接收超时。"""
        if failure == "text":
            connection.send("not pcm")
        elif failure == "disconnect":
            connection.close(code=1011, reason="test failure")
        else:
            finish.wait(timeout=5)

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config["input"]["websocket"].update(
            url=f"ws://127.0.0.1:{server.socket.getsockname()[1]}/pcmData",
            timeout_seconds=0.5,
        )
        expected = {
            "text": TypeError,
            "disconnect": ConnectionClosedError,
            "timeout": TimeoutError,
        }[failure]
        try:
            with pytest.raises(expected):
                list(iter_websocket_windows(config["input"]))
        finally:
            finish.set()
            server.shutdown()
            thread.join(timeout=5)


def test_websocket_repeated_request_limit(config):
    """验证按窗口发送请求，以及达到 max_windows 后主动关闭而不发送多余请求。"""
    from websockets.sync.server import serve

    requests = []

    def handler(connection):
        """每次请求返回十秒 PCM，总共服务两个窗口。"""
        for _ in range(2):
            requests.append(connection.recv())
            connection.send(np.zeros(10000, dtype="<i2").tobytes())

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config["input"]["websocket"].update(
            url=f"ws://127.0.0.1:{server.socket.getsockname()[1]}/pcmData",
            sample_rate=1000,
            request="read",
            request_each_window=True,
            max_windows=2,
        )
        try:
            windows = list(iter_websocket_windows(config["input"]))
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert len(windows) == 2
    assert requests == ["read", "read"]
