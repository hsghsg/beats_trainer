# MIMII fan / pump 下载脚本说明

脚本：[download_mimii_dataset.py](../scripts/download_mimii_dataset.py)。

来源固定为 [Zenodo MIMII public 1.0](https://zenodo.org/records/3384388)，仅处理 fan 和 pump，不下载 slider 或 valve。压缩包名称、精确大小、MD5 已于 2026-09-09 对照该记录的官方 API 核验并写入脚本，运行时无需额外第三方 Python 库。

## 运行方式

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -u -X utf8 scripts\download_mimii_dataset.py --output-dir 'D:\dataset\MIMII' --machines fan pump --snrs -6 0 6 --workers 2
```

默认目录、机器类型和信噪比与上面命令相同。先查看计划而不下载：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\download_mimii_dataset.py --dry-run
```

只处理某个版本，例如 0 dB pump：

```powershell
.\.venv\Scripts\python.exe -u -X utf8 scripts\download_mimii_dataset.py --machines pump --snrs 0
```

## 下载范围

| 压缩包 | 官方字节数 |
| --- | --- |
| -6_dB_fan.zip | 10,878,096,548 |
| -6_dB_pump.zip | 8,236,951,723 |
| 0_dB_fan.zip | 10,411,902,283 |
| 0_dB_pump.zip | 7,869,431,302 |
| 6_dB_fan.zip | 10,158,673,161 |
| 6_dB_pump.zip | 7,659,077,508 |

六包共 55,214,132,525 字节，约 55.21 GB。若已有完整的 -6_dB_pump.zip，还需下载 46,977,180,802 字节，约 46.98 GB。这里统计的是 ZIP 大小，解压数据还需要额外空间。

## 输出结构

不同信噪比压缩包内部使用相同的机器目录和文件名，因此分别放在信噪比目录，避免覆盖。

```text
D:\dataset\MIMII\
├── -6_dB\
│   ├── fan\id_00|id_02|id_04|id_06\normal|abnormal\*.wav
│   └── pump\id_00|id_02|id_04|id_06\normal|abnormal\*.wav
├── 0_dB\
│   ├── fan\...
│   └── pump\...
├── 6_dB\
│   ├── fan\...
│   └── pump\...
├── .mimii_state\            # 逐包完成清单、文件大小和 CRC32；失败时保留暂存内容
└── mimii_download.log      # 中文进度日志
```

上图的竖线表示多个并列目录，不是实际目录名。下载中的文件位于根目录，使用 ZIP 名加 .part 后缀；校验成功后改为 .zip。全部成功后，六个目标 ZIP 都会被删除，音频、完成清单和日志保留。

当前项目的 MIMII 预处理脚本按“机器类型／机器 ID／normal 或 abnormal”读取目录，因此接入时可将某个信噪比目录（例如 D:\dataset\MIMII\-6_dB）作为输入根目录。三个信噪比版本的关联样本不应不加区分地随机混入训练和测试集。

## 校验和失败恢复

1. 优先处理目标目录已有的 ZIP，先检查大小及官方 MD5，避免重复下载。
2. 下载支持 HTTP Range 断点续传；若服务端忽略 Range，则重写 .part 并重新下载，防止错误拼接。
3. 网络连接失败时退避重试，默认每包最多 12 次；每隔约 30 秒记录下载进度与速度。
4. MD5 通过后，在独立暂存目录解压。拒绝目录跳转、绝对路径、重复文件路径和符号链接成员。
5. 全部解压文件再次通过大小和 CRC32 检查后，发布最终目录，写入完成清单，随后只删除该包对应的 ZIP。
6. 重复执行会先依据完成清单重新检查已解压文件。检查通过则跳过下载；失败会保留现有内容并报错，不覆盖冲突文件。
7. 中断或失败后，重新运行相同命令即可续传下载；未发布的暂存解压内容会重新解压。MD5 错误、错误的完成清单或最终目录内容冲突需要先检查日志并处理原因。

同一输出目录只运行一个脚本进程。进程内部通过 --workers 控制并发，允许 1、2、3，默认 2。执行期间不要手动修改正在下载、校验或解压的文件，也不要删除 .mimii_state。

脚本返回码：0 表示全部完成，1 表示至少一包失败，130 表示收到中断。某包失败时会继续处理队列中的其他包，并在最终日志列出失败项。

查看实时日志：

```powershell
Get-Content -LiteralPath 'D:\dataset\MIMII\mimii_download.log' -Tail 20 -Wait
```

## 验证

只使用标准库的测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest test.test_download_mimii_dataset -v
```

覆盖成功后删除 ZIP、保留无关文件、重复执行跳过下载、MD5 失败保护、解压目标冲突保护、信噪比目录隔离、不安全 ZIP 路径，以及服务器接受或忽略 Range 的两种情况。

