# 分类器后端评估脚本使用说明

本文对应 [scripts/evaluate_classifier_backends.py](../scripts/evaluate_classifier_backends.py)，用于在同一测试集上比较 BEATs 原生分类头与三个特征分类器，输出指标、逐样本预测、ROC 曲线和标记 EER 阈值的 FAR/FRR 曲线。

## 1. 快速开始

在已安装项目依赖的 Python 环境中，进入项目根目录。以下为 **PowerShell** 示例，多行命令末尾使用反引号续行，反引号后不要添加空格。Linux/macOS 请换用实际项目路径，将续行符改为反斜杠，并将权重路径直接写入参数，或按所用 Shell 的语法定义变量。

```powershell
cd C:\hsg\beats_trainer

$ckpt = "logs\pump_pw_7fa13_finetune\version_3\checkpoints\last.ckpt"

# 首次训练并保存三个后端；以后更换测试集时无需重复执行。
python scripts/train_mimii_embedding_classifier.py `
  --data-dir data_ready `
  --model-path "$ckpt" `
  --classifier all `
  --device auto `
  --output-dir artifacts/classifiers_finetune_epoch02

python scripts/evaluate_classifier_backends.py `
  --data-dir data_ready `
  --classifier-dir artifacts/classifiers_finetune_epoch02 `
  --native-checkpoint "$ckpt" `
  --device auto `
  --extract-batch-size 4 `
  --native-batch-size 4 `
  --output-dir artifacts/evaluation_finetune_epoch02
```

示例使用项目中已有的全量微调检查点；评估其他实验时，将 `$ckpt` 替换为对应的 `.ckpt`。
脚本不会自动选择最新实验或验证指标最好的文件，需要显式指定要比较的检查点。

运行完成后，首先查看输出目录中的 `metrics.md` 和 `roc_curves_combined.svg`。
如果只想确认环境和命令行参数，可以运行：

```powershell
python scripts/evaluate_classifier_backends.py --help
```

首次使用时可在项目根目录执行 `python -m pip install -e .` 安装项目依赖；Python 和 PyTorch/CUDA 环境配置参考 [安装说明](installation.md)。
本脚本使用命令行参数，**不会读取 `train_beats.yaml`**。

## 2. 四种后端及执行流程

| 输出中的 backend | 分类方式 | 本次运行的行为 |
| --- | --- | --- |
| `beats-native-checkpoint` | 检查点中保存的原生分类头，可为线性头或训练时配置的多层头 | 加载已训练参数，直接在测试集上推理 |
| `local-density-knn` | 局部密度加权 KNN | 加载已拟合的 `.pkl`，直接评估测试集 |
| `relative-mahalanobis` | 相对马氏距离分类器 | 加载 `.pkl` 中已估计的统计量，直接评估测试集 |
| `gmm-cosine-knn` | GMM 与余弦 KNN 混合分类器 | 加载已拟合的 `.pkl`，直接评估测试集 |

脚本按以下顺序执行：

1. 收集测试集音频和类别，确定正类标签；无需训练目录和训练特征缓存。
2. 从 `--classifier-dir` 加载所选后端的 `.pkl`，检查分类器类型、拟合状态及类别。模型缺失时直接报错并提示训练命令，不会自动训练。
3. 使用 `--embedding-model-path` 指定的 BEATs 模型提取测试特征；未指定时复用 `--native-checkpoint`。特征采用均值池化，必须与训练分类器时使用的模型和特征提取设置一致。
4. 未设置 `--skip-native` 且原生检查点存在时，评估检查点中保存的分类头。
5. 使用已加载的后端直接预测测试集，计算指标并输出 ROC、FAR/FRR 曲线。

评估过程不训练 BEATs，也不调用后端分类器的 `fit()`。训练由 [train_mimii_embedding_classifier.py](../scripts/train_mimii_embedding_classifier.py) 完成：`--classifier all` 一次训练三个后端并共享训练特征；仍可指定单个后端，默认是 `local-density-knn`。

训练脚本在每个分类器拟合完成后立即保存 `<backend>.pkl`，再执行后续评估；即使后续验证集读取失败，模型文件仍保留。训练只使用 `train`；`val` 和 `test` 目录可选，存在时仅用于指标评估。以前该训练脚本保存的 `.pkl` 也可直接加载。

`--device` 控制 BEATs 特征提取和原生分类头推理设备；三个特征分类器使用 NumPy/scikit-learn 在 CPU 上计算。

## 3. 数据和检查点要求

### 数据目录

```text
data_ready/
├── train/
│   ├── abnormal/
│   │   └── *.wav
│   └── normal/
│       └── *.wav
└── test/
    ├── abnormal/
    │   └── *.wav
    └── normal/
        └── *.wav
```

- 当前指标流程用于**二分类**。训练集和测试集都应包含同样的两个类别；测试集需要同时包含正、负样本才能计算有效 AUC/pAUC。
- 类别由文件夹名称确定，默认正类为 `abnormal`；其他命名需设置 `--positive-label`。
- 评估脚本只读取 `test` 或 `--test-dir`，不需要 `train`、`val`。上图中的 `train` 只供训练脚本使用。
- 支持 `.wav`、`.mp3`、`.flac`、`.m4a`，扩展名匹配不区分大小写。
- 只扫描类别目录的直接子文件，不递归读取更深层目录。
- 原生推理和特征提取按 16 kHz 单声道读取音频。原生推理会将同批次音频补齐至最长长度；较长录音可先切片，并减小批量。
- 原生分类头评估的类别及顺序应与训练时一致。当前脚本按类别名排序映射输出列，不支持在评估阶段重命名类别或重映射类别编号。
- 对比不同检查点或分类器时，保持训练/测试划分一致；原生模型的训练数据也应与当前划分相匹配。

### 检查点参数的区别

| 参数 | 可用文件 | 用途 |
| --- | --- | --- |
| `--native-checkpoint` | 本项目训练生成的 Lightning `.ckpt` | 重建 BEATs 及其分类头，加载完整训练后权重 |
| `--embedding-model-path` | 原始 BEATs `.pt` 或兼容的 Lightning `.ckpt` | 提取三个特征分类器所需的音频特征 |
| 未设置 `--embedding-model-path` | 使用 `--native-checkpoint` 的值 | 让原生分类头与三个特征分类器使用相同来源的主干权重 |

原生 `.ckpt` 需要包含 `state_dict` 和 `hyper_parameters.config`，以及可确定的类别数。
原始 BEATs `.pt` 不能直接作为 `--native-checkpoint`。
原生模型重建时可能还会读取配置中记录的预训练权重路径，因此移动实验文件后也需确认该路径可用。

如果 `--embedding-model-path` 与原生检查点使用不同来源的主干权重，评估结果会同时反映特征来源与分类器的影响；比较分类头时可直接保持默认复用关系。

所有命令行中的相对路径均以**项目根目录**为基准；也支持绝对路径。这与训练 YAML 的“相对路径以 YAML 所在目录为基准”规则不同。

## 4. 完整命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-dir` | `data_ready` | 测试集所在的数据根目录，默认读取其 `test` 子目录 |
| `--output-dir` | `artifacts/classifier_backend_evaluation` | 指标、预测、曲线和默认特征缓存的输出目录 |
| `--test-dir` | 未指定，使用 `data-dir/test` | 新测试集目录，直接包含类别子目录；指定后只重算测试集特征 |
| `--classifier-dir` | `artifacts/mimii_embedding_classifier` | 训练脚本输出的分类器目录，包含 `<backend>.pkl` |
| `--classifiers` | 全部三个后端 | 空格分隔的后端名称；只训练过一个后端时，可仅指定该名称 |
| `--native-checkpoint` | `logs/pump_pw_7fa13_finetune/version_3/checkpoints/last.ckpt` | 原生分类头的训练检查点；不会自动切换到新实验 |
| `--embedding-model-path` | 未指定，复用原生检查点 | 特征提取模型路径 |
| `--device` | `cuda` | `auto`：CUDA 可用则使用 CUDA，否则使用 CPU；也可指定 `cpu`、`cuda`、`cuda:0` 等 |
| `--extract-batch-size` | `16` | 特征提取批量，正整数；显存不足时减小 |
| `--native-batch-size` | `16` | 原生分类头推理批量，正整数；显存不足时减小 |
| `--positive-label` | `abnormal` | 正类名称，用于 Recall、F1、AUC/pAUC 和正类分数输出 |
| `--pauc-max-fpr` | `0.1` | 标准化 pAUC 的最大假阳性率，范围为 `(0, 1]` |
| `--max-files-per-split` | 未限制 | 评估时只限制测试集，取排序后的前 N 个文件；不是每类 N 个，也不是随机或分层抽样 |
| `--recompute-features`、`--recompute-test-features` | 关闭 | 仅重新提取测试特征，不训练或修改分类器 |
| `--skip-native` | 关闭 | 跳过原生分类头，仍提取测试特征并评估所选已保存后端 |
| `-h`、`--help` | — | 显示帮助并退出，不启动评估 |

评估使用复数参数 `--classifiers`，训练使用单数参数 `--classifier`。评估没有 `--config`、`--dry-run` 或“仅评估原生分类头”选项。

后端参数由训练脚本确定并随 `.pkl` 保存，评估时不会重新构造或覆盖参数。训练默认值如下：

| 后端 | 当前参数 |
| --- | --- |
| 局部密度 KNN | `k=15`、`density_k=20` |
| 相对马氏距离 | `covariance_type="diag"`、`regularization=1e-4` |
| GMM + 余弦 KNN | `n_components=4`、`knn_k=15`、`alpha=0.6`、`random_state=42` |

调参后应在训练脚本中重新拟合并保存模型，评估脚本再通过 `--classifier-dir` 加载对应目录。

## 5. 常用运行方式

以下示例均要求已用相同 BEATs 模型训练并保存后端。默认读取 `artifacts/mimii_embedding_classifier`；模型位于其他目录时追加 `--classifier-dir`。仅有一个模型时追加例如 `--classifiers local-density-knn`。

### 只评估三个特征分类器

以下命令使用原始预训练模型提取特征，无需已有的分类头训练检查点：

```powershell
python scripts/evaluate_classifier_backends.py `
  --data-dir data_ready `
  --skip-native `
  --embedding-model-path checkpoints/BEATs_iter3_plus_AS2M.pt `
  --device auto `
  --extract-batch-size 4 `
  --output-dir artifacts/evaluation_pretrained_embeddings
```

也可将 `--embedding-model-path` 换成某次训练的 `.ckpt`，检查该主干提取的特征在三个分类器上的效果。

### 使用 CPU 或降低批量

以下命令使用脚本默认的 baseline 检查点；若要评估其他实验，请追加 `--native-checkpoint`：

```powershell
python scripts/evaluate_classifier_backends.py `
  --device cpu `
  --extract-batch-size 2 `
  --native-batch-size 2 `
  --output-dir artifacts/evaluation_baseline_cpu
```

GPU 显存不足时，可以保留 `--device cuda` 并将两个批量逐步降至 `2` 或 `1`。这两个参数不会限制 KNN/GMM 使用的 CPU 内存。

### 修改正类和 pAUC 范围

如果数据类别为 `fault`、`normal`，且希望只关注 FPR 不超过 5% 的范围：

```powershell
python scripts/evaluate_classifier_backends.py `
  --data-dir data_fault_ready `
  --skip-native `
  --embedding-model-path checkpoints/BEATs_iter3_plus_AS2M.pt `
  --positive-label fault `
  --pauc-max-fpr 0.05 `
  --output-dir artifacts/evaluation_fault_fpr005
```

`data_fault_ready` 为自备数据目录。该示例只评估特征分类器；已保存后端也必须按 `fault`、`normal` 类别训练；若加入原生分类头，同样需要按这两个类别训练得到的检查点。

### 小规模试跑

可先准备一个同时包含两类测试音频的小数据目录，再通过 `--test-dir` 指定。如果使用 `--max-files-per-split N`，需要确认测试集中排序后的前 N 个样本仍包含两类。

例如，某个划分有 100 个 `abnormal` 和 100 个 `normal`，设置 `--max-files-per-split 64` 会只选到前一类，不能得到有效的二分类评估。正式比较时应取消该限制，使用完整且固定的测试集。

## 6. 输出文件和指标

默认输出结构如下；自定义 `--output-dir` 时，根目录相应变化：

```text
artifacts/classifier_backend_evaluation/
├── metrics.csv
├── metrics.md
├── metrics.json
├── test_predictions.csv
├── roc_curves_combined.svg
├── roc_curves/
│   ├── beats-native-checkpoint.svg
│   ├── local-density-knn.svg
│   ├── relative-mahalanobis.svg
│   └── gmm-cosine-knn.svg
├── threshold_curves/
│   └── <backend>.svg
└── features/
    └── test_features_all.npz
```

| 文件 | 内容 |
| --- | --- |
| `metrics.csv` | 各后端的 `backend, accuracy, recall, f1, AUC, pAUC, EER, EER_threshold`，数值保留六位小数 |
| `metrics.md` | 指标表格，以及正类标签、pAUC 最大 FPR |
| `metrics.json` | 正类、pAUC 范围及各后端完整结果，包含预测、正类分数、FPR、TPR、ROC 阈值数组，以及 `error_thresholds`、`far`、`frr`、`EER`、`EER_threshold` |
| `test_predictions.csv` | 每条测试音频的 `path`、真实 `label`，以及各后端的 `<backend>_prediction` 和 `<backend>_score` |
| `roc_curves_combined.svg` | 所有参与评估的后端的 ROC 对比图 |
| `roc_curves/<backend>.svg` | 单个后端的 ROC 曲线，两轴刻度为 0.0～1.0、间隔 0.1 |
| `threshold_curves/<backend>.svg` | 阈值–FAR/FRR 曲线，以紫色交点和辅助线标记插值 EER 及阈值 |
| `features/*.npz` | 特征矩阵 `features`、类别 `labels` 和音频路径 `paths` |

`--skip-native` 不会生成本次运行的原生分类头结果。使用 `--max-files-per-split N` 后，测试缓存文件名改为 `test_features_maxN.npz`。分类器训练输出目录中的 `<backend>.pkl` 与评估结果目录独立。

| 指标 | 含义 |
| --- | --- |
| Accuracy | 测试集中预测正确的样本比例 |
| Recall | 指定正类的召回率 |
| F1 | 指定正类的精确率与召回率的调和平均，不是宏平均 F1 |
| AUC | 基于正类分数计算的完整 ROC 曲线下面积 |
| EER | FAR 与 FRR 相等处的线性插值错误率 |
| EER_threshold | 插值 EER 交点对应的正类分数阈值 |
| pAUC | FPR 从 0 到 `--pauc-max-fpr` 范围内的**标准化**局部 AUC |

pAUC 使用 `roc_auc_score(..., max_fpr=...)` 的标准化计算，随机区分水平对应约 0.5，理想区分为 1；最大 FPR 为 1 时等于完整 AUC。比较 pAUC 时应使用相同的最大 FPR。

Accuracy、Recall、F1 来自各后端默认的类别预测规则。`--pauc-max-fpr` 只影响 pAUC 的积分范围，**不会调整分类阈值，也不表示默认预测的 FPR 已达到该值**。
ROC/AUC 使用原生头的正类 softmax 概率，或特征分类器返回的正类概率/归一化分数。

## 7. 已保存分类器和重复评估

更换测试集时，只需加载原来保存的分类器，无需原训练音频、训练特征缓存或重新拟合。示例沿用快速开始中训练好的三个后端：

```powershell
python scripts/evaluate_classifier_backends.py `
  --test-dir data_new/test `
  --classifier-dir artifacts/classifiers_finetune_epoch02 `
  --native-checkpoint "$ckpt" `
  --device auto `
  --output-dir artifacts/evaluation_new_test
```

`$ckpt` 必须与训练时的模型一致。若后端使用的 BEATs 主干与原生分类头不同，显式传入训练时的 `--embedding-model-path`。

- `--test-dir` 同时作用于原生推理和后端评估，并强制更新测试特征，防止复用原测试集缓存。
- 默认测试缓存位于输出目录的 `features/test_features_all.npz`，或限制样本数时的 `test_features_maxN.npz`。在原路径替换音频后应追加 `--recompute-test-features`。
- 分类器加载会校验类型、拟合状态和类别；不会校验训练时的 BEATs 权重或预处理设置，必须由调用方保持一致。
- 更换训练数据、BEATs 模型或后端参数后，应重新运行训练脚本；更换特征模型时，训练端加 `--recompute-features`，评估端也更新测试特征。
- 评估端不再提供 `--train-feature-cache` 和 `--recompute-train-features`；原命令中的这两个选项应移除，改为指定 `--classifier-dir`。
- 即使命中特征缓存，当前脚本仍初始化 BEATs 提取器，因此模型文件和运行依赖仍然必需。
- 同一输出目录中的指标、预测和同名曲线会被覆盖；旧的、不参与本次运行的曲线不会自动删除。

### FAR/FRR 与 EER 口径

以“正类分数 **大于等于** 阈值”作为判为正类的规则，FAR=FPR（负类误判为正类的比例），FRR=1−TPR（正类误判为负类的比例）。默认正类为 `abnormal`，此时 FAR 表示正常样本误报率，FRR 表示异常样本漏报率。

曲线保留每个不同分数对应的阈值，并包含高于最高分的全拒绝端点。EER 和对应阈值通过相邻阈值的 FAR/FRR 线性插值计算，图中注明 `linear interpolation`。对于重复分数或离散样本，插值交点是估计值，实际按单个阈值分类不一定恰好满足 FAR=FRR。该值描述当前测试集，不会改变原有预测标签或 Accuracy/Recall/F1 的计算规则。

## 8. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `Model not found at ...` | 检查模型路径是否存在。即使设置 `--skip-native`，未显式指定特征模型时仍会尝试使用默认原生检查点；可补充有效的 `--embedding-model-path` |
| 原生检查点不存在但没有自动跳过 | 特征提取发生在原生评估之前，默认又复用原生检查点，因此可能先在特征模型加载阶段失败；需提供有效特征模型，才能继续只评估特征后端 |
| `checkpoint 中缺少 hyper_parameters.config` | 原生模型需要本项目保存的完整 Lightning `.ckpt`，不能传原始 `.pt` 或只有权重字典的文件 |
| 加载模型出现 `size mismatch` | 检查模型配置与权重结构是否匹配。当前特征提取器从 Lightning 配置重建模型时将 `embed_dim` 固定为 512；从零训练修改过该值时需先适配特征提取器 |
| 提示只支持二分类或无法定位正类 | 确认目录中只有目标的两个类别，且 `--positive-label` 与文件夹名称一致 |
| AUC/pAUC 报错或出现 NaN | 检查测试集是否同时含两类，特别是设置样本上限后；也应确认训练集同时含两类 |
| 更换模型后结果没有变化 | 检查是否复用了旧特征缓存，用新模型重新训练后端，并追加 `--recompute-test-features` 更新测试特征 |
| CUDA 不可用或显存不足 | 无 CUDA 时改用 `--device cpu`；显存不足时减小两个批量，或先对长音频切片 |
| 找不到音频 | 检查 `train/类别/音频`、`test/类别/音频` 的层级、扩展名和路径，类别目录下更深层的文件不会被扫描 |
| 找不到分类器 `.pkl` | 先运行训练脚本的 `--classifier all`，或用 `--classifiers` 只选择已有模型；检查 `--classifier-dir` 是否指向训练输出目录 |
| 分类器类别与测试集不一致 | 使用相同类别名称训练后端，或选择与测试集匹配的模型目录 |
| 只有三行指标或出现旧 ROC 图 | 检查是否跳过了原生分类头；确认使用独立输出目录，旧 ROC 文件不会自动清理 |

## 9. 相关入口

- [统一训练说明](training.md)：分类头训练、全量微调、从零训练。
- [统一训练配置](../scripts/train_beats.yaml)：训练模式和模型参数。
- [特征分类器训练脚本](../scripts/train_mimii_embedding_classifier.py)：单独选择分类器、调参并保存 `.pkl`。
