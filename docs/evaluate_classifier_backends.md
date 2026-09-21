# 分类器后端评估脚本使用说明

本文对应 [scripts/evaluate_classifier_backends.py](../scripts/evaluate_classifier_backends.py)，用于在同一测试集上比较 BEATs 原生分类头与三个特征分类器，输出指标、逐样本预测、ROC 曲线和标记 EER 阈值的 FAR/FRR 曲线。

## 1. 快速开始

在已安装项目依赖的 Python 环境中，进入项目根目录。以下为 **PowerShell** 示例，多行命令末尾使用反引号续行，反引号后不要添加空格。Linux/macOS 请换用实际项目路径，将续行符改为反斜杠，并将权重路径直接写入参数，或按所用 Shell 的语法定义变量。

```powershell
cd C:\hsg\beats_trainer

$ckpt = "logs/mimii_pump_finetune/version_0/checkpoints/mimii_pump_finetune-epoch=02-val_accuracy=0.989.ckpt"

python scripts/evaluate_classifier_backends.py `
  --data-dir data_ready `
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
| `local-density-knn` | 局部密度加权 KNN | 使用训练集特征拟合后，评估测试集 |
| `relative-mahalanobis` | 相对马氏距离分类器 | 使用训练集特征估计统计量后，评估测试集 |
| `gmm-cosine-knn` | GMM 与余弦 KNN 混合分类器 | 使用训练集特征拟合后，评估测试集 |

脚本按以下顺序执行：

1. 收集 `train`、`test` 中的音频和类别，确定正类标签。
2. 使用 `--embedding-model-path` 指定的 BEATs 模型提取训练集和测试集特征；未指定时复用 `--native-checkpoint`。特征采用均值池化，已有缓存可直接读取。
3. 未设置 `--skip-native` 且原生检查点存在时，评估检查点中保存的分类头。
4. 分别拟合三个特征分类器，在测试集上计算指标并保存结果。

BEATs 主干在评估过程中不进行梯度训练；三个特征分类器每次运行都会重新拟合。
本脚本不读取已有分类器 `.pkl`，也不保存拟合后的 `.pkl`。如需保存分类器，使用 [train_mimii_embedding_classifier.py](../scripts/train_mimii_embedding_classifier.py)。

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
- 只读取 `train` 和 `test`，不使用 `val`。本脚本输出的指标全部来自测试集。
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
| `--data-dir` | `data_ready` | 包含 `train`、`test` 的数据根目录 |
| `--output-dir` | `artifacts/classifier_backend_evaluation` | 指标、预测、曲线和默认特征缓存的输出目录 |
| `--test-dir` | 未指定，使用 `data-dir/test` | 新测试集目录，直接包含类别子目录；指定后只重算测试集特征 |
| `--train-feature-cache` | 未指定，使用当前输出目录的训练缓存 | 指向先前的训练特征 NPZ，支持更换输出目录时复用；文件必须存在，除非同时显式要求重算训练特征 |
| `--native-checkpoint` | `logs/pump_pw_7fa13_finetune/version_3/checkpoints/last.ckpt` | 原生分类头的训练检查点；不会自动切换到新实验 |
| `--embedding-model-path` | 未指定，复用原生检查点 | 特征提取模型路径 |
| `--device` | `cuda` | `auto`：CUDA 可用则使用 CUDA，否则使用 CPU；也可指定 `cpu`、`cuda`、`cuda:0` 等 |
| `--extract-batch-size` | `16` | 特征提取批量，正整数；显存不足时减小 |
| `--native-batch-size` | `16` | 原生分类头推理批量，正整数；显存不足时减小 |
| `--positive-label` | `abnormal` | 正类名称，用于 Recall、F1、AUC/pAUC 和正类分数输出 |
| `--pauc-max-fpr` | `0.1` | 标准化 pAUC 的最大假阳性率，范围为 `(0, 1]` |
| `--max-files-per-split` | 未限制 | 每个划分只取排序后的前 N 个文件，N 为正整数；不是每类 N 个，也不是随机或分层抽样 |
| `--recompute-features`、`--recompute-test-features` | 关闭 | 仅重新提取测试特征；旧参数现在不再强制重算训练集 |
| `--recompute-train-features` | 关闭 | 显式重新提取训练特征，覆盖选定的训练缓存 |
| `--skip-native` | 关闭 | 跳过原生分类头，仍提取特征并拟合、评估全部三个特征分类器 |
| `-h`、`--help` | — | 显示帮助并退出，不启动评估 |

当前没有 `--classifier`、`--config`、`--dry-run` 或“仅评估原生分类头”的命令行选项。

三个特征分类器的参数固定在脚本的 `build_embedding_classifiers()` 中：

| 后端 | 当前参数 |
| --- | --- |
| 局部密度 KNN | `k=15`、`density_k=20` |
| 相对马氏距离 | `covariance_type="diag"`、`regularization=1e-4` |
| GMM + 余弦 KNN | `n_components=4`、`knn_k=15`、`alpha=0.6`、`random_state=42` |

这些参数暂不支持通过本脚本的命令行修改。需要单独选择特征分类器或调参时，可使用 `train_mimii_embedding_classifier.py --classifier ...`，其指标和输出格式与本脚本不同。

## 5. 常用运行方式

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

`data_fault_ready` 为自备数据目录。该示例只评估特征分类器；若加入原生分类头，应使用按 `fault`、`normal` 类别训练得到的检查点。

### 小规模试跑

可先准备一个同时包含两类训练和测试音频的小数据目录，再通过 `--data-dir` 指定。如果使用 `--max-files-per-split N`，需要确认排序后的前 N 个样本在两个划分中都包含两类。

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
    ├── train_features_all.npz
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

`--skip-native` 不会生成本次运行的原生分类头结果。使用 `--max-files-per-split N` 后，缓存文件名改为 `train_features_maxN.npz` 和 `test_features_maxN.npz`，其中 `N` 替换为实际数值。

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

## 7. 特征缓存和重复评估

训练集和测试集使用独立的重算开关。已有训练缓存默认直接复用；没有训练缓存时才提取训练特征。`--recompute-features`（别名 `--recompute-test-features`）现在**只重算测试集**，训练集必须显式使用 `--recompute-train-features` 才强制重算。

### 更换测试集，复用原训练特征

新测试目录直接包含 `abnormal/`、`normal/` 等类别子目录，`--data-dir` 仍指向原训练集所在的数据根目录。`--test-dir` 同时作用于原生分类器推理和特征分类器评估，并自动重算测试特征，防止读取原测试集缓存。

同一输出目录下运行：

```powershell
python scripts/evaluate_classifier_backends.py `
  --data-dir data_ready `
  --test-dir data_new/test `
  --output-dir artifacts/classifier_backend_evaluation_fresh
```

若要将新测试集结果保存到新目录，显式指定之前的训练缓存：

```powershell
python scripts/evaluate_classifier_backends.py `
  --data-dir data_ready `
  --test-dir data_new/test `
  --train-feature-cache artifacts/classifier_backend_evaluation_fresh/features/train_features_all.npz `
  --output-dir artifacts/evaluation_new_test
```

以上命令复用默认模型；如果原缓存使用了其他模型，必须补充与原运行一致的 `--native-checkpoint` 或 `--embedding-model-path`。`--max-files-per-split` 也应与原训练缓存一致。

- 同一模型和训练数据下重复评估，不需要重算训练特征。缓存只包含音频特征；三个分类器每次重新拟合，原生分类头也会重新推理。
- 默认缓存仍位于输出目录的 `features/` 中，名称区分 `train/test` 和 `all/maxN`；不会校验模型、音频内容或预处理变化。指定训练缓存时，需保证模型、训练样本、标签与提取设置一致。
- 在原路径替换测试音频时，追加 `--recompute-test-features`；通过 `--test-dir` 指定测试集时已自动重算。
- 更换特征模型或训练数据后，用 `--recompute-train-features --recompute-test-features` 同时更新两份特征。若同时指定 `--train-feature-cache`，将覆盖该路径的训练缓存。
- 即使命中缓存，脚本仍会初始化特征提取器，因此有效模型文件和运行依赖仍然必需。
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
| 更换模型后结果没有变化 | 检查是否复用了旧特征缓存，更换输出目录且不引用旧缓存，或同时追加 `--recompute-train-features --recompute-test-features` |
| CUDA 不可用或显存不足 | 无 CUDA 时改用 `--device cpu`；显存不足时减小两个批量，或先对长音频切片 |
| 找不到音频 | 检查 `train/类别/音频`、`test/类别/音频` 的层级、扩展名和路径，类别目录下更深层的文件不会被扫描 |
| 只有三行指标或出现旧 ROC 图 | 检查是否跳过了原生分类头；确认使用独立输出目录，旧 ROC 文件不会自动清理 |

## 9. 相关入口

- [统一训练说明](training.md)：分类头训练、全量微调、从零训练。
- [统一训练配置](../scripts/train_beats.yaml)：训练模式和模型参数。
- [特征分类器训练脚本](../scripts/train_mimii_embedding_classifier.py)：单独选择分类器、调参并保存 `.pkl`。
