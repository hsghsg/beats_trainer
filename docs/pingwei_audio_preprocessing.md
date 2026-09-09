# 平圩设备音频预处理

脚本为 `scripts/preprocess_pingwei_audio.py`，所有处理参数放在同目录的
`preprocess_pingwei_audio.yaml` 中。只需 Python 及音频处理依赖，不加载模型。

在项目根目录安装依赖并运行：

```powershell
python -m pip install -r scripts/requirements_pingwei_audio.txt
python scripts/preprocess_pingwei_audio.py
```

指定另一份配置：

```powershell
python scripts/preprocess_pingwei_audio.py --config scripts/preprocess_pingwei_audio.yaml
```

## 默认处理规则

- 输入目录：`D:/pingwei-data/2026.8.24-1/#7机闭式水泵C`，不扫描子目录。
- 仅选择扩展名为 `.wav`、主名前缀为 `sync`、主名后缀为 `7fa13` 的文件。
  扩展名不区分大小写，前后缀区分大小写。
- 先将每条录音整体重采样为 16000 Hz，再连续切成无重叠的 10 秒片段。
  每段恰好 160000 帧；原始文件不修改、不删除。
- 多声道默认取算术平均转为单声道，保存为 16 位 PCM WAV，不做音量归一化。
  可配置 `mono: false` 保留声道；需要浮点编码时设为 `wav_subtype: FLOAT`。
  PCM 输出超范围时会给出削波提示。
- 默认丢弃不足 10 秒的尾段；设置 `tail_policy: pad` 可补零至 10 秒。
  原始文件不足 10 秒时同样适用；空文件不产生切片。
- 输出到项目根目录的 `data_ready/train/normal`。配置中的所有相对路径均以
  **YAML 所在目录**为基准，与命令执行目录无关。

## 命名和序号

例如输入：

```text
sync_20260824_163405_013715_7fa13.wav
```

输出首段：

```text
pw-7fa13-7.闭式水泵.C.mid-260824.1634-normal-0001.wav
```

日期和时间从不含扩展名的原文件名按 `_` 分隔后的第 2、3 列读取，将
`20260824 163405` 转成 `260824.1634`。同一源文件的所有片段保留这个采集时间，
不叠加片段偏移。第 2 段序号为 `0002`，依次递增。

每个不同输出时间字段从 `sequence_start` 开始编号。同一分钟有多份录音时，
按照源路径排序接续序号，以避免相同输出文件名。`sequence_width` 为最小位数。

## 预览、重跑和验证

将 YAML 中的 `dry_run` 改为 `true`，运行相同命令即可预览匹配文件及预计片段数，
不创建输出目录。正式处理时改回 `false`。

默认 `overwrite: false`，脚本在写入前检查全部计划文件名；有同名输出即报错，
不会覆盖已有训练数据。确认需要重新生成同名文件后，可设为 `true`。
覆盖模式不会清理旧批次多出的文件；修改输入集合、切片长度或编号规则后，
建议配置一个新的输出目录，避免混入旧片段。

处理出现损坏音频或写入错误时立即停止并返回非零退出码，保留之前已完成的片段，
失败片段会清理。内存按单个原始文件分配，不同时加载整个文件夹。

运行针对本脚本的测试，不加载训练测试依赖：

```powershell
python -m unittest test.test_preprocess_pingwei_audio -v
```
