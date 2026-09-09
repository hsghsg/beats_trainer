# 泵与电机异常声音构造

已使用同一对真实设备录音生成 6 段融合音频，另保存 2 段处理后的单源参考。
输出目录为 `artifacts/abnormal_audio/demo/`，全部为 10 秒、16 kHz、单声道、PCM 16-bit。
音频只是两种设备录音的合成样本，不表示真实泵电机系统发生了耦合故障。

## 素材来源

| 设备 | 数据来源与标签 | 本次选择 |
| --- | --- | --- |
| 泵 | [MIMII](https://zenodo.org/records/3384388)，`abnormal` | 项目已有 `data_ready/train/abnormal/pump-id_00-abnormal-00000000.wav` |
| 电机 | [IDMT-ISA-Electric-Engine](https://zenodo.org/records/7551261)，`engine2_broken` | 官方 ZIP 内 `train/engine2_broken/pure.wav`，下载后命名为 `motor_broken_original.wav` |

泵录音由项目已有训练数据提供，文件头为 16 kHz、8 通道、PCM 16-bit；本次选首通道。
本地目录未记录原下载包的背景噪声 SNR 版本，因此不将它标为 -6、0 或 +6 dB 版本。
`abnormal` 表示数据集异常标签，单文件未提供具体故障机理标签。

电机由 Fraunhofer IDMT 录制，通过供电电压和负载变化模拟 `good`、`heavyload`、`broken` 三种声学工况；
本次明确选择 `broken`，不将重载直接等同于损坏。
选中原始文件实测为 44.1 kHz、单声道、PCM 24-bit、约 372.912 秒；本次取开头 10 秒。
下载器仅获取约 45 MiB 的选中成员及说明 PDF，不下载整个约 1.5 GB 压缩包。

原始电机素材、说明和来源记录位于 `artifacts/abnormal_audio/sources/`。
`sources.json` 保存官方地址、ZIP 内路径、CRC32、SHA-256 和标签依据；ZIP 解压会检查成员 CRC。
融合清单 `demo/manifest.json` 另记录两路原文件的绝对路径和 SHA-256。

## 融合方式

令 `p` 为泵波形，`m` 为电机波形。四种方法均使用同一对截取信号，默认不单独归一化输入。

| 方法 | 运算与用途 | 输出 |
| --- | --- | --- |
| 加权叠加 | `0.5*p + 0.5*m`，保留原录音的幅度差异 | `weighted.wav` |
| 信噪比叠加 | `p + g*m`，控制两设备相对强弱 | `snr_-6dB.wav`、`snr_+0dB.wav`、`snr_+6dB.wav` |
| 等功率交叉渐变 | `cos(theta)*p + sin(theta)*m`，10 秒内从泵过渡至电机 | `crossfade.wav` |
| 间歇叠加 | 泵持续，电机每 2 秒开启 1 秒；开关处 20 ms 余弦淡变 | `intermittent.wav` |

SNR 在这里定义为**泵分量相对于电机分量的功率比**，与 MIMII 原录音的背景噪声 SNR 不同：

```text
g = RMS(p) / (RMS(m) * 10**(SNR_dB/20))
SNR_dB = 20 * log10(RMS(p) / RMS(g*m))
```

因此 +6 dB 为泵更强，-6 dB 为电机更强。间歇模式默认开启区间为 0 dB，
由于关闭区间仍有泵声，整段功率比不等于 0 dB，本次约 +3.54 dB。
交叉渐变的等功率指包络权重平方和为 1，不保证不同能量或相关信号的混合响度恒定。
加权叠加的 50% 权重也不代表等响度：本次电机录音原始 RMS 更大。

所有结果先去直流并用 SciPy 多相滤波重采样。默认只在超出 0.95 峰值时整体衰减，
不硬削波、不逐路改变最终配比。参考音频也采用相同峰值保护。
不循环短录音；未指定时长时取两路公共剩余时长，最多 10 秒；指定时长不足则报错。

## 使用方法

以下命令从项目根目录运行。独立脚本不需要 BEATs 权重、GPU 或 PyTorch。

```powershell
.venv\Scripts\python.exe -m pip install -r scripts/requirements_audio_fusion.txt
.venv\Scripts\python.exe -X utf8 scripts/download_abnormal_motor.py
```

当前素材已下载，无须重新执行下载命令。下载器和融合器均拒绝覆盖已有结果；
再次试验时用新的 `--output-dir`。

```powershell
.venv\Scripts\python.exe -X utf8 scripts/fuse_abnormal_audio.py --pump data_ready/train/abnormal/pump-id_00-abnormal-00000000.wav --motor artifacts/abnormal_audio/sources/motor_broken_original.wav --output-dir artifacts/abnormal_audio/experiment_02 --duration 10
```

可替换为自己的两段异常录音。用 `--methods weighted snr` 选择方法；
`--alpha 0.7` 调整泵权重；`--snr-db -12 -6 0 6 12` 扩展相对强度。
`--pump-start`、`--motor-start` 控制截取起点，`--channel 0` 选择首通道，`--channel -1` 平均降混。
`--period`、`--duty-cycle`、`--fade-ms` 控制间歇包络。周期需要能在当前时长内产生关闭区间。

所有结果及参考文件的参数、时长、峰值、RMS、文件校验和记录在 `manifest.json`。
该脚本根据输入角色赋予 `pump_abnormal`、`motor_abnormal` 标签，不自动判断输入是否异常。

## 验证与训练使用

```powershell
.venv\Scripts\python.exe -X utf8 -m unittest test.test_abnormal_audio_fusion -v
```

8 项测试覆盖功率比、加权幅度、渐变端点、间歇开关、异常输入、抗混叠、通道与裁剪、
端到端 PCM 输出及覆盖保护。真实结果也已检查：8 个音频文件各有 160000 个采样点，
两路 SNR 达到 -6、0、+6 dB，所有文件非静音、无削波且校验和互不相同。

用于分类训练时可赋总标签 `abnormal`，同时保留 `synthetic` 标识、设备角色和源文件哈希。
先按原始录音或设备划分 train/val/test，再只在各自划分内部生成增强样本；
同一录音的不同裁剪与混音不能分散到训练集和测试集。本次两路取自训练来源，
结果保留在实验目录，没有自动写入 `data_ready`。真实测试集应保持独立，用于检验泛化效果。

## 引用与授权

- 泵：Harsh Purohit、Ryo Tanabe、Kenji Ichige、Takashi Endo、Yuki Nikaido、Kaori Suefusa、Yohei Kawaguchi，Hitachi，MIMII Dataset，2019，DOI `10.5281/zenodo.3384388`，CC BY-SA 4.0。
- 电机：Sascha Grollmisch、Jakob Abeßer、Judith Liebetrau、Hanna Lukashevich，Fraunhofer IDMT，IDMT-ISA-Electric-Engine，DOI `10.5281/zenodo.7551261`；论文 *Sounding Industry: Challenges and Datasets for Industrial Sound Analysis*，EUSIPCO 2019。
- 电机的 [Zenodo 元数据](https://zenodo.org/api/records/7551261) 标注 CC BY-NC-ND 4.0；[授权条款](https://creativecommons.org/licenses/by-nc-nd/4.0/) 限制商业使用及修改后材料的分发。本次融合为本地实验，不按原授权对外分发混音。商用或发布派生数据需另获相应授权。

音频与派生文件已通过 `artifacts/abnormal_audio/.gitignore` 排除，不随代码提交。
