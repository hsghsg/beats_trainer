# 验证记录

最新验证日期：2026-09-24。目标设备由用户确认为 `Ascend310B1`。

## 已完成

- Ubuntu-22.04 / WSL2 x86_64 / Python 3.10.12 下建立独立 `.venv-wsl`。
- PyTorch/torchaudio 2.5.1+cpu，ONNX 1.17.0，ONNX Runtime 1.20.1。
- 30 项测试通过（2026-09-24）：配置校验、TorchScript、自定义 state_dict、BEATs 三种输出模式、
  ONNX Runtime 数值一致性、静态输入契约、ATC 命令参数、超时与失败产物保护、无 NPU 的 devlib 配置与编译器 Python 环境。
- Ruff 检查通过；Ubuntu 安装脚本 `bash -n` 通过；新增函数具备功能注释。
- Windows `run_wsl.ps1` 已验证环境检查、ONNX 导出、ATC 命令预览及缺少 CANN 时的失败报告。

## 实际模型转换

1. `asd/models/beats.ckpt` → `outputs/beats.onnx`。
   - 静态输入：float32 `features[1,998,128]`。
   - 输出：`embedding`、`probabilities`。
   - 两组随机输入对照中，embedding 最大绝对差 `2.91e-7`；概率最大绝对差 `3.79e-10`。
   - 原始导出报告：`outputs/beats_export_report.json`。
2. 使用实际十秒音频 `asd/audio/sample.wav` 的 ASD log-mel 前处理验证上述 ONNX。
   - embedding 最大绝对差 `1.27e-7`，概率最大绝对差 `4.08e-10`。
   - 输入示例：`outputs/sample_features.npy`。
   - 对照报告：`outputs/real_audio_validation.json`。
3. `checkpoints/BEATs_iter3_plus_AS2M.pt` → `outputs/pretrained_embedding.onnx`。
   - 输出：`embedding`。
   - 真实特征及随机特征均通过对照，最大绝对差 `1.23e-6`。
   - 报告：`outputs/pretrained_export_report.json`。

产物和虚拟环境均被子项目 `.gitignore` 排除，不提交大文件。
BEATs 原实现导出时有固定尺寸断言的 TracerWarning 及 weight_norm 弃用提示；
本版仅支持配置的静态 shape，已通过该 shape 下的多输入数值对照。

## 真实 ONNX → OM 验证（2026-09-24）

**编译成功**：Ubuntu-22.04 / WSL2 x86_64，CANN Toolkit 9.1.1 + 310B ops 9.1.1，
安装路径 `/home/robo/Ascend/cann`（实际目录 `cann-9.1.1`）。无 NPU、未安装驱动。

本次先修复缺少 ops 引起的 `--host_env_os ... support list of {}`，补装官方
`Ascend-cann-310b-ops_9.1.1_linux-x86_64.run`；再补齐 `requirements-atc.txt`
中的 Python 依赖。配置更新为 `Ascend310B1`，并添加无设备环境使用的 devlib 路径。

- 输入文件：`outputs/beats.onnx`，无需修改已验证的 ONNX 图。
- 输出文件：`outputs/beats.om`，193,073,818 字节（约 184.13 MiB）。
- OM SHA-256：`df46ea2bc4826de8a065afbd29671558cd0d877a181b60002222e31fe082f4b1`。
- ONNX SHA-256：`8dd27ab9d255fd87036ac54a0cb599e539cb93833b91eca1b8087802895f3cb9`。
- ATC 退出码 0，日志明确返回 `ATC run success`。
- 使用 `atc --mode=6 --om=...` 读取成功，确认目标 `Ascend310B1`。
  模型元信息中的 ATC 组件版本为 `9.1.0`；安装包版本为 `9.1.1`，分别如实记录。
- 使用 `atc --mode=1 --om=... --json=...` 导出真实 OM 结构，校验以下契约通过：

| 方向 | 张量 | 数据类型 | 形状 |
| --- | --- | --- | --- |
| 输入 | features | float32 | [1, 998, 128] |
| 输出 0 | embedding | float32 | [1, 768] |
| 输出 1 | probabilities | float32 | [1, 2] |

编译保持 CANN 默认精度模式；OM 内部存在 float16 转换，输入输出为 float32
不代表全部算子采用 float32。ATC 有一条 W11001 提示：
`/encoder/layers.0/self_attn/Abs` 未命中高优先级算子信息库，可能影响性能，未阻断编译。

证据文件：`outputs/beats_om_report.json`、`outputs/beats.atc.log`、
`outputs/beats.om_info.log`、`outputs/beats.om.json`、`outputs/om_validation.json`。
其中 `beats_om_report.json` 是本次成功报告的归档，避免后续命令覆盖通用报告。

激活 CANN 后的全量 `pip check` 还报告 MindStudio/profiler 等附加工具的依赖缺失；
这些工具未参与本次 ATC 编译，本次未安装其依赖，也未验证这些附加工具。

## 尚未完成的设备验证

本次已完成真实离线编译与 OM 结构检查，未在 Ascend310B1 硬件上执行推理。
OM 与 PyTorch/ONNX 的数值误差、ASD 判定一致性、内存峰值及推理延迟需上板验证。
现有 ONNX Runtime 对照结果不能作为 OM 数值验证结果。

参考：[CANN 9.1 ATC 环境准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/atctool/atlasatc_16_0002.html)。
