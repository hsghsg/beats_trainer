# Training with BEATs

Complete guide for training and fine-tuning BEATs models on custom datasets.

## YAML 驱动的 MIMII 三模式训练

在项目根目录运行：

```bash
python scripts/train_beats.py
```

默认读取 `scripts/train_beats.yaml`。通常只需修改 YAML 中的 `mode`，
无需修改 Python 脚本：

| mode | 权重初始化 | 更新范围 | 默认学习率 / 最大轮数 / 批量 |
| --- | --- | --- | --- |
| `head` | 加载预训练权重 | 仅分类头 | 1.0e-4 / 30 / 16 |
| `finetune` | 加载预训练权重 | BEATs 主干及分类头 | 5.0e-5 / 50 / 4 |
| `scratch` | 随机初始化 | BEATs 主干及分类头 | 1.0e-3 / 100 / 4 |

YAML 中公共的 `data`、`model`、`training` 参数会被所选 `modes` 分支中的
同名参数覆盖；学习率建议保持 `1.0e-4` 这样的 YAML 数值写法。
模式自动控制冻结和随机初始化开关，类别数由数据推断。
实验名称会追加模式后缀，便于分别查看日志和检查点。

也可以临时覆盖训练模式或指定其他配置文件：

```bash
python scripts/train_beats.py --mode head
python scripts/train_beats.py --mode finetune
python scripts/train_beats.py --mode scratch
python scripts/train_beats.py --config scripts/train_beats.yaml --mode finetune --dry-run
```

`--dry-run` 校验参数并打印合并后的配置，不加载模型、不检查数据文件内容，也不启动训练。
配置文件内的所有相对路径以 **YAML 所在目录** 为基准。
默认数据目录为项目的 `data_ready/train`、`data_ready/val`、`data_ready/test`，
每个目录下按类别存放音频。

`training.gpus: 0` 强制使用 CPU，正整数指定 GPU 数量。
`model.model_path` 用于加载原始 BEATs 预训练权重，从零模式会忽略此参数；
`resume_from_checkpoint` 用于恢复 Lightning 的完整训练状态。
恢复训练需保持模式、模型结构、类别及数据划分一致；切换模式启动新实验时，
应将恢复路径设为 `null`。

`test_after_training: true` 会在训练结束后使用最佳可用检查点评估测试集
（未保存最佳检查点时回退到最后检查点）；设为 `false` 时可将 `data.test_dir` 留空。
训练参数和从零训练的模型结构均已在 YAML 中提供中文注释。


## Quick Start

### Fine-tuning (Recommended)

```python
from beats_trainer import BEATsTrainer

# Train from directory structure
trainer = BEATsTrainer.from_directory("/path/to/dataset")
results = trainer.train()
```

### ESC-50 Example (Auto-Download & Train)

```python
from beats_trainer import BEATsTrainer, Config

config = Config()
config.model.freeze_backbone = False  # Fine-tune entire model
config.training.max_epochs = 50
config.training.learning_rate = 5e-5

# Auto-download, organize, and train
trainer = BEATsTrainer.from_esc50(data_dir="./datasets", config=config)
trainer.train()
```

## Data Formats

### Directory Structure (Recommended)

```
your_dataset/
├── class1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── audio3.mp3
├── class2/
│   ├── audio4.wav
│   └── audio5.flac
└── class3/
    ├── audio6.wav
    └── audio7.wav
```

```python
trainer = BEATsTrainer.from_directory("your_dataset")
results = trainer.train()
```

### Pre-Split Datasets

```
dataset/
├── train/
│   ├── class1/
│   └── class2/
├── val/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

```python
trainer = BEATsTrainer.from_split_directories(
    train_data_dir="dataset/train",
    val_data_dir="dataset/val",
    test_data_dir="dataset/test"
)
trainer.train()
```

### CSV Metadata

**dataset.csv:**
```csv
filename,category
audio/sample1.wav,bird
audio/sample2.mp3,dog
audio/sample3.flac,cat
```

```python
trainer = BEATsTrainer.from_csv(
    csv_path="dataset.csv",
    audio_column="filename",
    label_column="category"
)
trainer.train()
```

### Split CSV Files

```python
trainer = BEATsTrainer.from_split_csvs(
    train_csv="train.csv",
    val_csv="val.csv",
    test_csv="test.csv",
    audio_column="filename",
    label_column="category"
)
trainer.train()
```

## Audio Clip Duration Control

BEATs trainer supports both variable-length audio (automatic padding) and fixed-length audio (manual duration control). This gives you precise control over memory usage and training consistency.

### Variable Length (Default)

```python
from beats_trainer import Config

config = Config()
config.data.clip_duration = None  # Default: variable length
config.data.batch_size = 16

trainer = BEATsTrainer.from_directory("dataset", config=config)
# Each batch will be padded to the longest clip in that batch
```

**Characteristics:**
- Preserves original audio length
- Dynamic padding per batch
- Good for consistent-length datasets
- May use more memory

### Fixed Duration Clips

Force all audio clips to be exactly the same length:

```python
config = Config()
config.data.clip_duration = 1.0    # All clips will be exactly 1 second
config.data.sample_rate = 16000    # = 16,000 samples per clip
config.data.batch_size = 32        # Can use larger batches with consistent size

trainer = BEATsTrainer.from_directory("dataset", config=config)
```

**What happens:**
- **Longer audio**: Truncated to target duration (takes from beginning)
- **Shorter audio**: Zero-padded to target duration
- **Exact length**: No modification needed

### Common Duration Configurations

```python
from beats_trainer import Config

# Quick prototyping - very short clips
config = Config()
config.data.clip_duration = 0.5
config.data.batch_size = 64

# Standard short clips
config = Config()
config.data.clip_duration = 1.0
config.data.batch_size = 32

# Medium clips for detailed analysis
config = Config()
config.data.clip_duration = 3.0
config.data.batch_size = 16

# Long clips for complex audio
config = Config()
config.data.clip_duration = 10.0
config.data.batch_size = 8
```

### Memory and Performance

| Clip Duration | Samples (16kHz) | Memory/Clip | Batch Size | Use Case |
|---------------|-----------------|-------------|------------|-----------|
| 0.5s          | 8,000          | ~32 KB     | 64-128     | Quick testing |
| 1.0s          | 16,000         | ~64 KB     | 32-64      | Short sounds |
| 2.0s          | 32,000         | ~128 KB    | 16-32      | Standard clips |
| 5.0s          | 80,000         | ~320 KB    | 8-16       | Long analysis |
| 10.0s         | 160,000        | ~640 KB    | 4-8        | Very long clips |

### Choosing the Right Duration

**Use shorter clips (0.5-2s) for:**
- Quick prototyping
- Simple classification tasks
- Limited compute resources
- Bird calls, alarms, brief sounds

**Use longer clips (3-10s) for:**
- Complex temporal patterns
- Music analysis
- Speech recognition
- Environmental monitoring
- Detailed acoustic analysis

**Use variable length when:**
- Your dataset has consistent clip lengths
- You want to preserve all original audio
- Memory usage is not a concern

### Sample Rate Considerations

Different sample rates affect the number of samples per clip:

```python
from beats_trainer import Config

# High quality, short duration
config = Config()
config.data.sample_rate = 22050    # High quality
config.data.clip_duration = 1.0    # = 22,050 samples
config.data.batch_size = 16

# Standard quality, longer duration
config = Config()
config.data.sample_rate = 16000    # Standard
config.data.clip_duration = 2.0    # = 32,000 samples
config.data.batch_size = 16

# Lower quality, very long duration
config = Config()
config.data.sample_rate = 8000     # Lower quality
config.data.clip_duration = 4.0    # = 32,000 samples
config.data.batch_size = 16
```

All three configurations above use the same memory per clip (~128KB) but provide different quality/duration trade-offs.

## Training Strategies

### Classification Head Only (Fast)

```python
from beats_trainer import Config

config = Config()
config.model.freeze_backbone = True  # Only train classifier
config.model.fine_tune_backbone = False

trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

### Full Fine-tuning (Best Performance)

```python
config = Config()
config.model.freeze_backbone = False  # Train entire model
config.model.fine_tune_backbone = True
config.training.learning_rate = 5e-5  # Lower learning rate

trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

### Training From Scratch

```python
from beats_trainer import Config

config = Config()

# Model settings for training from scratch
config.model.train_from_scratch = True    # Key parameter!
config.model.encoder_layers = 12          # Transformer layers
config.model.encoder_embed_dim = 768      # Hidden dimension
config.model.encoder_attention_heads = 12 # Attention heads
config.model.fine_tune_backbone = True
config.model.freeze_backbone = False

# Training settings for scratch training
config.training.learning_rate = 1e-3      # Higher LR for scratch training
config.training.max_epochs = 100          # More epochs needed
config.training.patience = 15             # More patience

trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

## Advanced Configuration

### Complete Configuration Example

Here's how to configure every aspect of training with a full config:

```python
from beats_trainer import Config

# Complete configuration with all parameters
config = Config(
    # Experiment settings
    experiment_name="my_audio_classifier",
    seed=42
)

# Configure data settings
config.data.batch_size = 32
config.data.num_workers = 4
config.data.sample_rate = 16000
config.data.clip_duration = None        # None = variable length, or set to seconds (e.g., 1.0)

# Data splits (only used with from_directory or from_csv)
config.data.train_split = 0.7           # 70% for training
config.data.val_split = 0.2             # 20% for validation
config.data.test_split = 0.1            # 10% for testing

# Model configuration
config.model.model_path = None           # Use None for auto-download
config.model.num_classes = None          # Set automatically from data
config.model.dropout_rate = 0.1          # Dropout rate for classifier

# Training strategy
config.model.freeze_backbone = False     # True = freeze BEATs, False = fine-tune
config.model.fine_tune_backbone = True   # Enable backbone training
config.model.train_from_scratch = False  # True = no pre-trained weights

# For training from scratch only
config.model.encoder_layers = 12
config.model.encoder_embed_dim = 768
config.model.encoder_attention_heads = 12
config.model.input_patch_size = 16

# Training configuration
config.training.learning_rate = 5e-5     # Lower for fine-tuning, higher for scratch
config.training.optimizer = "adamw"      # "adamw", "adam", or "sgd"
config.training.weight_decay = 1e-4
config.training.scheduler = "cosine"     # "cosine", "step", or "plateau"
config.training.max_epochs = 50
config.training.patience = 10            # Early stopping patience

# Hardware settings
config.training.gpus = 1                 # Number of GPUs (0 for CPU)
config.training.precision = 32           # 16 for mixed precision, 32 for full
config.training.deterministic = False    # True for reproducible results (slower)

# Use the complete configuration
trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

### Configuration for Different Scenarios

#### Small Dataset (< 1000 samples)

```python
config = Config()

config.data.batch_size = 16            # Smaller batches
config.data.train_split = 0.8          # More training data
config.data.val_split = 0.2
config.data.test_split = 0.0           # No test split for small data

config.model.freeze_backbone = True    # Only train classifier head
config.model.dropout_rate = 0.3        # More dropout to prevent overfitting

config.training.learning_rate = 1e-4   # Higher learning rate
config.training.max_epochs = 100       # More epochs
config.training.patience = 20          # More patience
config.training.weight_decay = 1e-3    # More regularization
```

#### Large Dataset (> 10,000 samples)

```python
config = Config()

config.data.batch_size = 64            # Larger batches
config.data.num_workers = 8            # More data loading workers

config.model.freeze_backbone = False   # Fine-tune entire model
config.model.dropout_rate = 0.1        # Less dropout needed

config.training.learning_rate = 1e-5   # Lower learning rate
config.training.max_epochs = 30        # Fewer epochs needed
config.training.patience = 5           # Less patience needed
config.training.weight_decay = 1e-5    # Less regularization
config.training.precision = 16         # Mixed precision for speed
```

#### GPU Optimization

```python
config = Config()

config.data.batch_size = 64            # Larger batch for GPU
config.data.num_workers = 8            # Match CPU cores

config.training.precision = 16         # Mixed precision training
config.training.gpus = 1               # Use GPU
config.training.deterministic = False  # Faster training
        drop_last=True            # Consistent batch sizes
    ),
    training=TrainingConfig(
        precision=16,             # Mixed precision training
        gpus=1,                   # Use GPU
config.training.precision = 16         # Mixed precision training
config.training.gpus = 1               # Use GPU
config.training.deterministic = False  # Faster training
```

### YAML Configuration

You can also save configurations as YAML files:

**config.yaml:**
```yaml
experiment_name: "audio_classifier_v1"
seed: 42

data:
  batch_size: 32
  sample_rate: 16000
  clip_duration: 1.0
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1

model:
  model_path: null
  freeze_backbone: false
  dropout_rate: 0.1

training:
  learning_rate: 5e-5
  optimizer: "adamw"
  max_epochs: 50
  patience: 10
  scheduler: "cosine"
  patience: 10
  scheduler: "cosine"
```

Then load it:

```python
config = Config.from_yaml("config.yaml")
trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

### Quick Parameter Changes

```python
# Start with default config and modify specific parameters
config = Config()

# Change just a few key parameters
config.training.learning_rate = 1e-4
config.training.max_epochs = 100
config.data.batch_size = 16
config.model.freeze_backbone = False

trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

## Model Testing and Evaluation

### Test Trained Model

```python
# Test the model
test_results = trainer.test()
print(f"Test accuracy: {test_results['test_accuracy']:.3f}")
```

### Make Predictions

```python
# Predict on new audio files
predictions = trainer.predict(["new_audio1.wav", "new_audio2.wav"])
```

### Extract Features with Trained Model

```python
# Get feature extractor from trained model
feature_extractor = trainer.get_feature_extractor()
features = feature_extractor.extract_from_files(["audio1.wav", "audio2.wav"])
```
