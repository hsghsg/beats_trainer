"""转换工厂配置、导出数值与 ATC 进程控制测试。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from transfer_factory.atc import build_atc_command, cann_environment, compile_om
from transfer_factory.config import load_config
from transfer_factory.convert import main
from transfer_factory.factory import TransferFactory
from transfer_factory.onnx_export import check_onnx_contract, export_onnx

ROOT = Path(__file__).resolve().parents[1]


class TinyNetwork(torch.nn.Module):
    """可解析预期输出的小型网络，用于 ONNX Runtime 数值验证。"""

    def forward(self, features):
        """计算固定仿射变换，返回与输入同 shape 的输出张量。"""
        return features * 2.0 + 1.0


@pytest.fixture
def config(tmp_path):
    """建立与测试目录隔离的 TorchScript 配置，避免触碰真实模型和产物。"""
    data = load_config(ROOT / "transfer.yaml")
    model = tmp_path / "tiny.pt"
    torch.jit.trace(TinyNetwork(), torch.zeros(1, 4)).save(str(model))
    data["model"].update(kind="torchscript", checkpoint=model)
    data["input"].update(shape=[1, 4], sample_npy=None)
    data["onnx"].update(path=tmp_path / "tiny.onnx", output_names=["output"])
    data["atc"].update(output=tmp_path / "compiled", env_script=None, devlib_path=None)
    data["runtime"].update(
        report_path=tmp_path / "report.json", log_path=tmp_path / "convert.log"
    )
    return data


def write_yaml(path: Path, config: dict) -> None:
    """将包含 Path 对象的测试配置写为标准 YAML 字符串路径。"""
    serializable = json.loads(json.dumps(config, default=str))
    path.write_text(yaml.safe_dump(serializable), encoding="utf-8")


def test_factory_and_real_onnx(config):
    """通过工厂真实导出 TorchScript，校验结构及两组数值对照结果。"""
    result = TransferFactory().convert("pt", "onnx", config)
    assert result["verified"] and len(result["checks"]) == 2
    assert all(item["max_abs_error"] == 0 for item in result["checks"])
    with pytest.raises(ValueError):
        TransferFactory().create("pt", "om")
    with pytest.raises(FileExistsError):
        export_onnx(config)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("input", "shape", [0, 4]),
        ("input", "shape", [-1, 4]),
        ("input", "dtype", "float16"),
        ("onnx", "output_names", ["output", "output"]),
        ("onnx", "atol", float("nan")),
        ("runtime", "overwrite", "false"),
    ],
)
def test_invalid_config(config, tmp_path, section, key, value):
    """拒绝不支持的动态尺寸、精度类型、重名输出及无效阈值。"""
    config[section][key] = value
    path = tmp_path / "invalid.yaml"
    write_yaml(path, config)
    with pytest.raises(ValueError):
        load_config(path)


def test_config_relative_and_collision(tmp_path):
    """相对路径基于配置目录，输出不得覆盖源 checkpoint 或配置文件。"""
    raw = yaml.safe_load((ROOT / "transfer.yaml").read_text(encoding="utf-8-sig"))
    path = tmp_path / "transfer.yaml"
    write_yaml(path, raw)
    assert load_config(path)["onnx"]["path"] == tmp_path / "outputs/beats.onnx"
    raw["onnx"]["path"] = raw["model"]["checkpoint"]
    write_yaml(path, raw)
    with pytest.raises(ValueError, match="覆盖"):
        load_config(path)


def test_input_contract_mismatch(config):
    """拒绝与配置名称、shape 不同的 ONNX，避免错误 ATC 输入。"""
    export_onnx(config)
    config["input"]["shape"] = [1, 8]
    with pytest.raises(ValueError, match="契约"):
        check_onnx_contract(config["onnx"]["path"], config)


def test_failed_export_preserves_old_model(config, tmp_path):
    """非法真实样本导致导出失败时，旧的成功模型文件必须保持原样。"""
    export_onnx(config)
    original = config["onnx"]["path"].read_bytes()
    filename = tmp_path / "sample.npy"
    np.save(filename, np.zeros((1, 4), dtype=np.float64))
    config["input"]["sample_npy"] = filename
    config["runtime"]["overwrite"] = True
    with pytest.raises(ValueError, match="样本"):
        export_onnx(config)
    assert config["onnx"]["path"].read_bytes() == original


def test_atc_command_ascend310b(config):
    """ATC 命令采用 ONNX 框架标识及用户指定型号，并保持形状为单个参数。"""
    command = build_atc_command(config)
    assert "--framework=5" in command
    assert "--soc_version=Ascend310B1" in command
    assert "--input_shape=features:1,4" in command
    assert "--input_format=ND" in command
    config["atc"]["soc_version"] = None
    with pytest.raises(ValueError, match="soc_version"):
        build_atc_command(config)


@pytest.mark.parametrize(
    "option", ["output", "framework", "model", "soc_version", "mode", "dynamic_dims"]
)
def test_atc_reserved_options(config, option):
    """额外 ATC 参数不能覆盖输出路径、芯片型号或静态转换规则。"""
    config["atc"]["extra_args"] = {option: "changed"}
    with pytest.raises(ValueError):
        build_atc_command(config)


def fake_atc(tmp_path: Path, body: str) -> Path:
    """生成仅用于测试 ATC 进程边界的可执行脚本，绝不作为真实 OM 编译器。"""
    path = tmp_path / "fake atc"
    path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_atc_process_success(config, tmp_path):
    """用测试替身验证参数传递与产物发布，不据此宣称真实 CANN 编译通过。"""
    export_onnx(config)
    executable = fake_atc(
        tmp_path,
        "import sys\nfrom pathlib import Path\n"
        "prefix = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--output='))\n"
        "Path(prefix + '.om').write_bytes(b'test-only')\n",
    )
    config["atc"]["executable"] = str(executable)
    result = compile_om(config)
    assert Path(result["output"]).read_bytes() == b"test-only"
    assert result["soc_version"] == "Ascend310B1"


@pytest.mark.parametrize(
    "body,exception",
    [
        ("import sys\nsys.exit(9)\n", RuntimeError),
        ("pass\n", RuntimeError),
        ("import time\ntime.sleep(30)\n", __import__("subprocess").TimeoutExpired),
    ],
)
def test_atc_failure_preserves_old(config, tmp_path, body, exception):
    """非零退出、未生成文件和超时都不能覆盖已有成功 OM。"""
    export_onnx(config)
    config["atc"].update(executable=str(fake_atc(tmp_path, body)), timeout_seconds=1)
    target = Path(str(config["atc"]["output"]) + ".om")
    target.write_bytes(b"existing-good")
    config["runtime"]["overwrite"] = True
    with pytest.raises(exception):
        compile_om(config)
    assert target.read_bytes() == b"existing-good"
    assert not list(tmp_path.glob(".atc-*"))


def test_source_environment_with_spaces(tmp_path):
    """CANN 环境脚本路径含空格时仍可安全加载，且不修改父进程环境。"""
    script = tmp_path / "set env.sh"
    script.write_text(
        "export TRANSFER_FACTORY_TEST_VALUE='with spaces'\n", encoding="utf-8"
    )
    assert cann_environment(script)["TRANSFER_FACTORY_TEST_VALUE"] == "with spaces"
    with pytest.raises(FileNotFoundError):
        cann_environment(tmp_path / "missing.sh")


def test_cli_failure_report(config, tmp_path, monkeypatch):
    """all 的 OM 阶段失败时，报告保留 ONNX 成功记录且退出码非零。"""
    config["atc"]["env_script"] = tmp_path / "missing-cann.sh"
    path = tmp_path / "config.yaml"
    write_yaml(path, config)
    monkeypatch.setattr(sys, "argv", ["convert.py", "all", "--config", str(path)])
    assert main() == 1
    report = json.loads(config["runtime"]["report_path"].read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["stages"][0]["status"] == "success"
    assert config["onnx"]["path"].is_file()


@pytest.mark.parametrize("output_mode", ["embedding", "probabilities", "both"])
def test_beats_export_matches_torch(config, tmp_path, output_mode):
    """导出真实小型 BEATs .pt，验证相对位置注意力与标准化的 ORT 数值一致性。"""
    from asd.vendor.BEATs.BEATs import BEATs, BEATsConfig

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
    model = BEATs(BEATsConfig(cfg))
    checkpoint = tmp_path / "beats.pt"
    torch.save({"cfg": cfg, "model": model.state_dict()}, checkpoint)
    config["model"].update(kind="beats", checkpoint=checkpoint, output_mode="embedding")
    config["input"]["shape"] = [1, 32, 32]
    config["onnx"]["output_names"] = ["embedding"]
    if output_mode != "embedding":
        head = torch.nn.Linear(16, 2)
        state = {f"backbone.{key}": value for key, value in model.state_dict().items()}
        state.update(
            {f"classifier.{key}": value for key, value in head.state_dict().items()}
        )
        torch.save(
            {
                "state_dict": state,
                "hyper_parameters": {"config": {"model": cfg}, "num_classes": 2},
            },
            checkpoint,
        )
        config["model"].update(output_mode=output_mode, backbone_overrides=cfg)
        config["onnx"]["output_names"] = (
            ["probabilities"]
            if output_mode == "probabilities"
            else ["embedding", "probabilities"]
        )
    result = export_onnx(config)
    assert result["verified"]
    assert max(item["max_abs_error"] for item in result["checks"]) < 1e-4


def test_python_factory_state_dict(config):
    """自定义网络通过可导入构造器加载嵌套 state_dict 并完成真实 ONNX 导出。"""
    torch.save({"weights": TinyNetwork().state_dict()}, config["model"]["checkpoint"])
    config["model"].update(
        kind="python_factory",
        factory="transfer_factory.tests.test_transfer:TinyNetwork",
        factory_kwargs={},
        state_dict_key="weights",
    )
    result = export_onnx(config)
    assert result["verified"]
    assert result["model_kind"] == "python_factory"


def test_missing_pytorch_source_does_not_block_om(config, tmp_path):
    """单独 ONNX→OM 不需要原始 PT 文件，两个转换阶段可以独立部署。"""
    export_onnx(config)
    config["model"]["checkpoint"] = tmp_path / "missing.pt"
    config["atc"]["executable"] = str(
        fake_atc(
            tmp_path,
            "import sys\nfrom pathlib import Path\n"
            "prefix = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--output='))\n"
            "Path(prefix + '.om').write_bytes(b'test-only')\n",
        )
    )
    assert compile_om(config)["status"] == "success"


@pytest.mark.parametrize("with_script", [False, True])
def test_offline_compiler_environment(tmp_path, monkeypatch, with_script):
    """无 NPU 编译库和当前 Python 优先用于子进程，父环境不变且拒绝缺失库目录。"""
    import os

    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    script = None
    if with_script:
        script = tmp_path / "set_env.sh"
        script.write_text("export LD_LIBRARY_PATH=/cann/lib\n", encoding="utf-8")
    environment = cann_environment(script, tmp_path)
    suffix = "/cann/lib" if with_script else "/existing/lib"
    assert environment["LD_LIBRARY_PATH"] == str(tmp_path) + os.pathsep + suffix
    assert environment["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert os.environ["LD_LIBRARY_PATH"] == "/existing/lib"
    with pytest.raises(FileNotFoundError, match="离线编译"):
        cann_environment(script, tmp_path / "missing")
