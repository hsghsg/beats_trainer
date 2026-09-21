# CUMTB 风电变桨轴承 CSV 声音转换

脚本：`scripts/convert_cumtb_acoustic.py`。

## 数据依据

已读取源目录 readme.txt 并检查实际 CSV：

- 文件无表头，第 1 列为连续样本序号，不是秒时间戳。
- 第 2～5 列为四个振动通道，第 6 列为声学信号。
- 行末逗号形成第 7 列空值。脚本只提取第 1、6 列，不会将空列或振动当声音。
- 序号可能从 1000001 等值开始；每个 CSV 独立输出，不拼接。
- 本地共 1155 个 CSV，约 54.2 GB，覆盖 Cond_1、Cond_2、1rpm、3rpm 及各健康/故障类别。

本地 README 写为 **38.5 Hz**，与数据集原始论文不一致。
[原论文的采集说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC12340508/)
明确给出 **38.5 kHz**，因此默认原始采样率为 **38500 Hz**。
第 1 列没有秒时间信息，不能从其步长推导原始采样率。
如后续得到其他可靠采集参数，可通过 `--source-rate` 指定，单位为 Hz。

## 运行

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\convert_cumtb_acoustic.py
```

默认输入：`D:\dataset\CUMTB风电变桨轴承`。

默认输出：项目目录下的 `dataset/CUMTB_acoustic_16k`，
依据脚本位置定位项目，因此从其他工作目录运行也不会改变默认输出位置。

示例映射：

```text
Cond_1/1rpm/Health/vibandacoustic/va (1).csv
→ dataset/CUMTB_acoustic_16k/Cond_1/1rpm/Health/vibandacoustic/va (1).wav
```

独立环境依赖：`python -m pip install -r scripts/requirements_cumtb.txt`。

## 转换策略

- 目标固定为 **16000 Hz、单声道**，采用多相 FIR 抗混叠重采样，比率 32/77。
- 不分段、不去直流、不删除尾部样本；输出样本数向上取整，时长误差小于 1/16000 秒。
- 1000000 个源样本约为 25.974 秒，对应 415585 个输出样本。
- 默认 **32 位浮点 WAV（FLOAT）**，保留原声学数值及不同文件间的幅值关系。
  本地说明未明确声学数值的物理单位，脚本不进行 Pa 或电压换算。
- 添加 `--subtype PCM_16` 可生成常规 16 位 PCM WAV；每个文件独立将峰值
  归一化到 0.99，静音保持为零。这会改变文件间绝对幅值关系。
- 默认四线程，按文件加载，不将 54 GB 数据一次性载入内存。
  可通过 `--workers 1` 降低内存和磁盘并发。

每次运行输出独立的 `conversion_*.jsonl` 清单，每完成一个文件刷新一条记录，
包含路径、状态、首尾源序号、输入输出采样率和样本数、时长及缩放系数。
单个文件失败时继续处理其他文件，最终存在失败则退出码为 1。

默认跳过格式、采样率、声道和编码符合要求的已有 WAV。
跳过时不重新验证源文件内容或原始采样率；更改源数据或采样率后请使用新目录或
`--overwrite`。不符合目标格式的已有 WAV 会报错，也可用该参数明确覆盖。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest test.test_convert_cumtb_acoustic -v
```

覆盖列提取、首行保留、频率和幅值、抗混叠、PCM/静音、长度取整、
异常序号、目录保留、失败继续处理和覆盖策略。

