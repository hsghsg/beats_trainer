# ASD 独立异常声音检测

基于当前项目推理实现，支持 Ubuntu 和 openEuler 上的 Python/CPU 或 CUDA。
只需复制整个 `asd` 文件夹、安装第三方依赖即可运行，无需安装父项目，
无需 Lightning、训练数据目录或联网下载 BEATs 权重。

## 快速运行

使用 Python 3.10 及以上版本，推荐与训练环境一致的 Python 3.12。
先安装与处理器架构及 CUDA 驱动匹配、版本配套的 `torch` 和 `torchaudio`。

```bash
cd asd
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python infer.py --check-config
python infer.py
```

Ubuntu 缺少音频系统库时安装 `sudo apt-get install libsndfile1`；
openEuler 使用 `sudo dnf install libsndfile`。系统自带 Python 过旧时，
先准备独立的 Python 3.10+ 环境。aarch64 需要适配该架构的 PyTorch 包，
上述命令不会自动配置 CUDA 驱动或替代系统 Python。

当前工作区已在 `models/` 放置本项目 epoch02 的推理权重及三个已训练分类器，
并在 `audio/sample.wav` 放置一个十秒示例。它们被 `.gitignore` 排除；
通过 Git 获取代码的部署机仍需自行复制这些文件。
`beats.ckpt` 去除了优化器和训练状态，仅保留骨干、分类头及重建所需配置。
源 checkpoint 为 `pump_pw_7fa13_finetune` version_3 的 epoch02；
三个后端来自 `artifacts/classifiers_finetune_epoch02/`。

仅加载可信来源的 `.ckpt`、`.pt`、`.pkl`，这些格式的反序列化可执行代码。
部署时尽量保持 numpy、scikit-learn、torch、torchaudio 与训练版本一致。

## 配置与输入

业务配置全部位于 `infer.yaml`，每项均有中文行后注释。
所有相对路径以该 YAML 所在目录为基准，与启动目录无关。
也可在父目录执行 `python -m asd --config asd/infer.yaml`。

- 文件：设置 `input.mode: file`，`input.path` 填 WAV 文件或目录；支持 `.WAV`。
- 文件夹：`recursive: true` 递归扫描，按路径排序，长文件逐个十秒窗口处理。
- 尾段：`tail_policy` 可选 `pad`（右侧补零）、`drop` 或 `error`。
  `valid_seconds` 记录补零前时长；补零会参与特征计算，须与训练策略一致。
- 多声道取均值，非 16 kHz 输入通过多相滤波重采样为 16 kHz。
  不进行峰值归一化、自动增益或静音剔除。

WebSocket 模式设置 `input.mode: websocket`，地址格式是
`ws://localhost:8765/pcmData`（支持 `wss://`）。ASD 是客户端，音频服务由外部提供。
服务端通过二进制消息返回无 WAV 头的原始 PCM；支持 `s16le` 和 `f32le`，
多声道按采样帧交错排列。采样率、声道数和格式必须与 YAML 一致。
`f32le` 按浮点音频原值使用，通常应在 [-1, 1] 范围内。

`request: null` 表示连接后直接接收推送；如果服务需要订阅命令，可把原样发送的
文本或 JSON 字符串放入 `request`。`request_each_window: true` 表示每完成一个
窗口后再次发送请求，适合一次请求返回十秒 PCM 的服务。
不假定私有协议，不对文本或 JSON 响应自动提取音频。

消息边界可以落在采样值的任意字节处，也可一次包含多个窗口。
正常关闭才按尾段策略输出剩余音频；超时、异常关闭、残缺采样帧或文本消息会报错。
不自动重连，避免把不连续音频拼成同一个窗口。`max_windows: 0` 持续运行，
正整数用于限制窗口数；按 Ctrl+C 停止并关闭连接。

## 前处理与模型一致性

`log-mel-energy` 复用项目的 Kaldi fbank 参数：16 kHz、128 个 mel bin、
25 ms 窗长、10 ms 帧移，波形先乘 32768。BEATs 内部仅执行一次
`(特征 - fbank_mean) / (2 * fbank_std)`。

`wavelet` 使用复制的 `CWTAmplitudeTransform`，与项目现有 Morlet CWT 幅值算法一致。
`cwt_time_stride: 1` 保留原始时间分辨率，十秒输入通常产生约 5 万个 Transformer
输入 token，因此默认 `max_tokens: 4096` 会拒绝这种超长序列。
可设置例如 `cwt_time_stride: 160` 将时间点数降至 1000；**改变前处理、尺度数、
抽样步长或标准化后，必须使用以同一处理方式训练的 BEATs/后端并重新标定阈值**。
现有附带权重使用 log-mel，不能仅切换为小波就假设仍有相同检测精度。
若必须复现未抽样 CWT，需在确认内存容量后显式增大 token 上限。

四种后端分别为：

| 配置名称 | 评分方式 | 模型输入 |
| --- | --- | --- |
| beats-native | 项目线性/MLP 头的异常类 softmax 概率 | 未归一化的平均 embedding |
| local-density-knn | 局部密度 KNN 的异常类归一化分数 | 平均 embedding，默认 L2 归一化 |
| relative-mahalanobis | 相对马氏距离的异常类归一化分数 | 同上 |
| gmm-cosine-knn | GMM＋余弦 KNN 的异常类归一化分数 | 同上 |

每个后端支持独立 `enabled` 和 `threshold`。异常分数越大越偏向异常，
分数等于阈值也判异常。所有分数范围均为 [0,1]，但不是跨后端可直接比较的
校准概率；请使用验证集标定各自阈值，示例 0.5 不代表最佳工作点。

`beats-native.labels` 必须与训练时输出列顺序完全一致，示例为 `[abnormal, normal]`。
其他三个分类器从 `.pkl` 的 `classes_` 定位异常列，不假设异常列固定为 0 或 1。
系统面向正常/异常二分类，类别不匹配、权重缺失、后端加载失败均报错停止。

embedding 可加载项目 Lightning `.ckpt` 或原始 BEATs `cfg/model` 格式 `.pt`。
原生分类头须使用项目训练的 Lightning `.ckpt`。两条路径使用相同 checkpoint
时共享一次前向；也可配置不同 checkpoint，但三种后端必须匹配 embedding 权重。
Lightning 未保存完整 BEATs 结构参数时，使用项目默认值并从张量恢复维度。
特殊结构可通过 `embedding.backbone_overrides` 显式补充，示例：

```yaml
  backbone_overrides: {deep_norm: true, max_distance: 800} # 与训练架构保持一致
```

## Result 与日志

每个十秒窗口返回一个 `Result`。后端结果是 `ClassifierResult`，包含
`name`、`label`、`score` 和 `threshold`；最终的 `score` 是异常票数，
`threshold` 是要求的票数 `m`，不是对多个概率阈值取平均。
例如四个后端有两个判异常且 `m=2` 时，最终判异常。
`m` 必须为 1 到启用后端数量之间的整数。

```python
from asd.engine import ASDInference

system = ASDInference("asd/infer.yaml")
for result in system.run():
    print(result.label, result.score, result.threshold)
    print(result.to_dict())
# 对内存十秒波形：result = system.predict_array(waveform, sample_rate=16000)
```

`Result` 还包含 `source`、从 0 开始的 `window_index`、`start_seconds` 和
`valid_seconds`。每个后端的标签、分数、阈值和最终投票结论均写入日志。
命令行 stdout 为逐行 JSON，日志写入 stderr 和可选日志文件。
JSONL 按窗口追加并及时 flush，不把无限 PCM 流积累在内存中。
已有结果文件不会覆盖，重复运行会追加新记录。

## 源码与验证

`vendor/BEATs/` 复制项目的网络、Transformer 和底层模块；
`vendor/beats_trainer/` 复制三种后端、配置类和 CWT 算法。
保留源文件版权头和 `vendor/LICENSE`。`compat.py` 将历史 pickle 中的
`beats_trainer.*` / `BEATs.*` 类型映射至本目录副本，不修改全局模块名。

```bash
python -m pip install pytest
# 在 asd 所在父目录执行
python -m pytest asd/tests -q
```

测试覆盖配置、阈值边界、十秒切片、重采样、PCM 分包、WebSocket 本地服务、
原生线性/MLP 头、三个 embedding 后端及脱离父项目后的完整推理。
当前开发机为 Windows，已进行实际权重推理；Ubuntu 22.04 / Python 3.10
已通过配置加载与命令行启动验证。Linux 完整模型推理及 openEuler 部署尚未
实测，代码采用跨平台 Python 库，无 Windows 专属推理调用。详见 `VALIDATION.md`。
