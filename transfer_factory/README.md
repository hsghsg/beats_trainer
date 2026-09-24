# Transfer Factory 模型转换工厂

在 Ubuntu WSL 中执行 **PyTorch → ONNX → 昇腾 OM**，默认目标型号为用户指定的
`Ascend310B1`。子项目位于仓库 `transfer_factory/`，复用相邻 `asd/` 的 BEATs
离线加载器，不修改训练代码或 ASD 推理流程。

## 安装与启动

Windows PowerShell 中执行（使用 Ubuntu-22.04）：

```powershell
wsl -d Ubuntu-22.04 --cd C:\hsg\beats_trainer --exec bash transfer_factory/setup_wsl.sh
.\transfer_factory\run_wsl.ps1 -Action doctor
.\transfer_factory\run_wsl.ps1 -Action pt-to-onnx
.\transfer_factory\run_wsl.ps1 -Action atc-command
.\transfer_factory\run_wsl.ps1 -Action onnx-to-om
# 两阶段连续执行：-Action all
```

也可在 Ubuntu 内运行：

```bash
cd /mnt/c/hsg/beats_trainer
bash transfer_factory/setup_wsl.sh
source transfer_factory/.venv-wsl/bin/activate
python -m transfer_factory doctor
python -m transfer_factory pt-to-onnx
python -m transfer_factory atc-command
python -m transfer_factory onnx-to-om
```

需要 Python 3.10+。若 `python3 -m venv` 提示缺少 ensurepip，请先安装 Ubuntu 的
`python3-venv` 包。安装脚本仅创建子项目下的独立虚拟环境，使用 CPU 版
PyTorch/torchaudio 2.5.1；它不安装 CANN、驱动，也不修改系统 Python。

`run_wsl.ps1` 支持 `-Distribution` 和 `-Config`。所有模型、输入、产物与日志路径
均相对于 YAML 文件所在目录解析。WSL 中使用 `/mnt/c/...`，不能写 `C:\...`。

## 集中配置

编辑带中文行后注释的 `transfer.yaml`。默认读取 `../asd/models/beats.ckpt`，
输入为 `[1,998,128]`，输出 `embedding` 与 `probabilities`。这些源权重被 Git 忽略，
其他机器需自行复制源文件。

常用字段：

| 字段 | 含义 |
| --- | --- |
| model.kind | beats、torchscript 或 python_factory |
| model.checkpoint | 本地源模型 `.pt` 或 `.ckpt` |
| model.output_mode | BEATs 的 embedding、probabilities 或 both |
| input.name / shape | 单输入名称和固定尺寸，本版不接受动态 shape |
| input.sample_npy | 可选真实 float32 验证输入，否则生成可复现随机输入 |
| onnx.output_names | 输出名称，顺序及数量须与模型实际返回值一致 |
| onnx.verify | 默认开启 ONNX Runtime 与 PyTorch 数值对照 |
| atc.soc_version | 默认 Ascend310B1，须匹配目标设备与 CANN 支持范围 |
| atc.env_script | 已安装 CANN Toolkit 的 set_env.sh 路径 |
| atc.devlib_path | 无 NPU 的 WSL 使用 CANN x86_64-linux/devlib；有设备时可设 null |
| atc.output | OM 输出前缀，不带 `.om` 扩展名 |
| runtime.overwrite | 默认 false；重跑时须更换输出路径或显式允许覆盖 |

第一次导出后再执行 `all` 会因 ONNX 已存在而停止；可直接执行 `onnx-to-om`。
目标 `.onnx/.om` 只有在本阶段完成校验后才发布，失败不覆盖已有成功产物。

## BEATs 输入与 ASD 的衔接

**导出的模型输入是未标准化的时频特征，不是原始十秒 PCM。** Kaldi fbank 和
NumPy FFT 小波前处理在 CPU/ASD 中执行，不将其当成可直接移植到 ATC 的网络算子。
BEATs 内部执行一次 `(features - mean) / (2 * std)`，前处理输出不能再次标准化。

十秒、16 kHz 单声道、128 个 mel bin、25 ms 窗长、10 ms 帧移对应 `[1,998,128]`。
如使用小波，须按真实小波输出修改 `input.shape` 并使用匹配训练流程的权重。
导出时的固定尺寸和推理输入必须完全一致。

可在仓库根目录运行以下 Python 代码，使用 ASD 的实际前处理生成验证输入：

```python
from pathlib import Path
import numpy as np
from asd.audio import iter_file_windows
from asd.config import load_config
from asd.frontend import Frontend

config = load_config("asd/infer.yaml")
window = next(iter_file_windows(config["input"]))
features = Frontend(config["frontend"])(window.waveform, patch_size=16)
Path("transfer_factory/outputs").mkdir(exist_ok=True)
np.save("transfer_factory/outputs/sample_features.npy", features.numpy())
```

随后设置 `input.sample_npy: outputs/sample_features.npy`。

`embedding` 输出默认 L2 归一化，可继续交给 ASD 的三个 Python 分类器；
原生分类头使用未归一化的平均 embedding，输出 softmax 概率。此转换不包含
三个 `.pkl` 分类器，也不把 ASD 的投票、阈值和日志逻辑编译进 OM。
`model.labels` 用于记录原生头的列顺序，必须与训练标签对应。

原始预训练 `.pt` 不含项目的原生分类头，需改为：

```yaml
model:
  checkpoint: ../checkpoints/BEATs_iter3_plus_AS2M.pt
  output_mode: embedding
onnx:
  output_names: [embedding]
```

以上是需修改的字段，其他配置保持完整。`.pt` 后缀不能唯一确定网络结构：
`torch.jit.save` 保存的模型选 `torchscript`；普通 `state_dict` 选 `python_factory`，
以 `model.factory: your_module:build_model` 和 `factory_kwargs` 声明网络构造函数，
并按需要填写 `state_dict_key`。构造函数返回 `torch.nn.Module`，加载采用严格匹配。
仅加载可信模型与自定义 Python 模块。

## ONNX → OM（CANN ATC）

按昇腾官方流程，ONNX 使用 `--framework=5`，指定输入 shape 和目标 SoC。
默认配置对应命令的形式为：

```bash
source /home/robo/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/home/robo/Ascend/cann/x86_64-linux/devlib:$LD_LIBRARY_PATH
atc --model=/path/to/beats.onnx \
    --framework=5 \
    --output=/path/to/beats \
    --input_shape="features:1,998,128" \
    --input_format=ND \
    --soc_version=Ascend310B1
```

在 Ubuntu WSL 中准备与系统架构及目标 Ascend310B1 匹配的 CANN Toolkit，按该版本
安装指南补齐 ATC 的 Python 依赖，并将 `atc.env_script` 指向安装环境脚本。
本机验证环境使用 CANN 9.1.1，安装在 `/home/robo/Ascend`。
8.5 及之后版本还必须安装与 Toolkit 版本、主机架构及目标芯片匹配的 **ops 算子包**。
仅 `atc` 能启动不代表具备完整的模型编译环境；缺少 ops 时可能报告
`--host_env_os ... support list of {}`。

在 Ubuntu 内补齐依赖（算子包需从华为官方渠道获取）：

```bash
bash Ascend-cann-310b-ops_9.1.1_linux-x86_64.run --install --install-path=/home/robo/Ascend --type=toolkit
transfer_factory/.venv-wsl/bin/python -m pip install -r transfer_factory/requirements-atc.txt
```

无 NPU 的 WSL 使用配置中的 `atc.devlib_path` 加载离线编译库，无需安装硬件驱动。
实际编译由工厂初始化 CANN 环境，并优先使用当前 Python 虚拟环境；无需在 PowerShell 中 source。
如果已经激活 CANN，可将 `env_script` 设置为 null。

`doctor` 检查包、环境脚本及 ATC；`atc-command` 只预览，不宣称生成模型。
若目标设备报告了更具体的芯片型号，请按目标设备的 `npu-smi info` 和该 CANN
版本文档调整 `soc_version`。工具不会自动把不支持的型号替换成其他芯片。

转换会检查 ATC 退出码、实际生成的非空 `.om` 文件，保存 `.atc.log`，并在超时或
中断时终止编译进程组。ONNX 结构/数值检查通过仍不能保证所有算子受目标 CANN
版本支持；若 ATC 报不支持的算子或精度模式，应查看编译日志并按对应版本适配。
在 WSL 编译不等于已在目标昇腾设备完成推理验证，部署后仍需对照结果。

参考资料：

- [CANN 9.1 ATC 环境准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/atctool/atlasatc_16_0002.html)，说明 ops 算子包和无设备时的 devlib 配置。

- 用户提供的[ONNX 转 OM 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/quickstart/quickstart_18_0010.html)。本次读取原地址仅返回站点导航。
- 可读取的[官方 ONNX 转 OM 快速入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/quickstart/quickstart/quickstart_18_0010.html)，说明 ATC 参数与型号查询。
- [CANN 8.2.RC1.alpha002 ATC 快速入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/devaids/atctool/atlasatc_16_0003.html)，说明 ONNX 转换和编译结果。

## 输出与测试

成功导出会生成 ONNX 或 OM；`outputs/conversion_report.json` 记录本次命令状态、
输入契约、输出路径、SHA-256、数值误差和错误信息。每次命令会更新该报告，
`transfer.log` 追加记录日志。`all` 中 OM 失败仍保留成功 ONNX 及对应阶段记录。

```bash
source transfer_factory/.venv-wsl/bin/activate
python -m pip install -r transfer_factory/requirements-dev.txt
python -m pytest transfer_factory/tests -q
```

测试中 ATC 替身只验证进程调用、错误处理和文件发布，不代表真实 OM 编译。
实际环境和模型验证情况记录在 `VALIDATION.md`。
