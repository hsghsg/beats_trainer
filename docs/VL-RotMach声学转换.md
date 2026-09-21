# VL-RotMach 声学数据转换

脚本：`scripts/convert_vl_rotmach_acoustic.py`。

依据数据集 readme.txt，声学数据包含时间和声压（Pa）。实际检查的 5 个 MAT
均为 Test.Lab 导出的 Signal 结构，而非直接的两列矩阵：

- `Signal.x_values.increment`：采样间隔（秒），实测为 1/51200。
- `Signal.x_values.start_value`：起始时间（秒）。
- `Signal.x_values.number_of_values`：样本数，实测为 3072000，即 60 秒。
- `Signal.y_values.values`：单声道声压数组，单位 Pa。

脚本逐文件读取并验证元数据，使用多相 FIR 滤波进行抗混叠重采样。
51.2 kHz → 16 kHz 的比例为 5/16，每个 60 秒文件输出 960000 个样本。
保留文件名和子目录，不裁剪、不分段、不去直流，也不修改源 MAT。

## 运行

在项目根目录使用已有虚拟环境（已具备依赖）：

```powershell
.\.venv\Scripts\python.exe scripts\convert_vl_rotmach_acoustic.py
```

默认输入为 `D:\dataset\韩国科学技术院多模态数据集\VL-RotMach\acoustic`，
默认输出为同级 acoustic_wav_16k 目录。也可指定路径：

```powershell
.\.venv\Scripts\python.exe scripts\convert_vl_rotmach_acoustic.py --input-dir "D:\dataset\韩国科学技术院多模态数据集\VL-RotMach\acoustic" --output-dir "outputs\vl_rotmach_acoustic_16k"
```

独立环境安装依赖：`python -m pip install -r scripts/requirements_vl_rotmach.txt`。

## 幅值与编码

默认输出 **32 位浮点 WAV（FLOAT）**，保留重采样后的 Pa 数值和文件之间的幅值关系。
原始声压可能超过 ±1，不能直接转换为整数 PCM，否则会削波。
浮点文件适合数据分析；播放器播放超过 ±1 的浮点音频时可能削波。

需要常规 **16 位 PCM WAV** 时，添加 `--subtype PCM_16`。脚本将每个文件在
重采样后的峰值归一化至 0.99，静音保持为零；这会改变文件间的绝对声压关系。
JSON 清单中的 gain_applied 保存缩放系数，WAV 数值除以该系数即可近似还原 Pa
（PCM 存在量化误差）。

输出均为 **16 kHz、单声道**。每次运行生成独立的 conversion_*.json，记录
成功、跳过或失败状态；成功记录含采样率、样本数、时长、源时间起点和幅值系数。
WAV 时间轴从零开始，源时间起点保存在清单中。

默认跳过已有 WAV，不校验已有文件是否符合本次参数；更改编码后请使用新输出目录，
或显式添加 `--overwrite`。单文件失败不影响其他文件；存在失败时退出码为 1。
仅支持上述已验证的 MAT 结构；其他结构和 MATLAB v7.3 文件会报错。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest test.test_convert_vl_rotmach_acoustic -v
```

测试涵盖时长和频率保留、高频抗混叠、浮点幅值、PCM 归一化、静音、异常数据、
失败后继续处理及已有输出保护。

