"""PyTorch Lightning data module for BEATs training."""

import librosa
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Callable

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

from ..core.config import DataConfig


def collate_audio_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for audio data with optional fixed duration.

    Args:
        batch: List of (audio_tensor, padding_mask, label) tuples

    Returns:
        Tuple of (padded_audio, padding_mask, labels) tensors
    """
    # Separate the components
    audio_tensors, padding_masks, labels = zip(*batch)

    # Check if all audio tensors have the same length (fixed duration mode)
    lengths = [audio.shape[0] for audio in audio_tensors]
    all_same_length = len(set(lengths)) == 1

    if all_same_length:
        # Fixed duration mode - all clips are already the same length
        audio_batch = torch.stack(audio_tensors)
        padding_batch = torch.stack(padding_masks)
        label_batch = torch.tensor(labels, dtype=torch.long)
    else:
        # Variable duration mode - need to pad to max length
        max_length = max(lengths)

        # Pad all audio tensors to the same length
        padded_audio = []
        batch_padding_masks = []

        for audio, old_mask in zip(audio_tensors, padding_masks):
            current_length = audio.shape[0]

            if current_length < max_length:
                # 按时间维度补齐，支持 1D 波形与 2D 特征图
                padding_needed = max_length - current_length
                if audio.dim() == 1:
                    padded = torch.nn.functional.pad(
                        audio, (0, padding_needed), value=0.0
                    )
                elif audio.dim() == 2:
                    padded = torch.nn.functional.pad(
                        audio, (0, 0, 0, padding_needed), value=0.0
                    )
                else:
                    raise ValueError("样本张量维度不合法，支持 1D 波形或 2D 时频图")

                # Update padding mask (True = padded/masked, False = real audio)
                new_mask = torch.cat(
                    [
                        old_mask,  # Keep existing mask
                        torch.ones(
                            padding_needed, dtype=torch.bool
                        ),  # Add padding mask
                    ]
                )
            else:
                padded = audio
                new_mask = old_mask

            padded_audio.append(padded)
            batch_padding_masks.append(new_mask)

        # Stack into batches
        audio_batch = torch.stack(padded_audio)
        padding_batch = torch.stack(batch_padding_masks)
        label_batch = torch.tensor(labels, dtype=torch.long)

    return audio_batch, padding_batch, label_batch


class AudioDataset(Dataset):
    """Dataset for loading audio files."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_dir: Path,
        sample_rate: int = 16000,
        clip_duration: Optional[float] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.transform = transform

        # Calculate target length in samples if clip_duration is set
        self.target_length = None
        if self.clip_duration is not None:
            self.target_length = int(self.clip_duration * self.sample_rate)

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.dataframe["category"])
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Load audio
        audio_path = self.data_dir / row["filename"]
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Handle fixed clip duration
        if self.target_length is not None:
            current_length = audio_tensor.shape[0]

            if current_length > self.target_length:
                # Truncate to target length (take from the beginning)
                audio_tensor = audio_tensor[: self.target_length]
                padding_mask = torch.zeros(self.target_length, dtype=torch.bool)

            elif current_length < self.target_length:
                # Pad to target length
                padding_needed = self.target_length - current_length
                audio_tensor = torch.nn.functional.pad(
                    audio_tensor, (0, padding_needed), value=0.0
                )

                # Create padding mask (True = padded, False = real audio)
                padding_mask = torch.cat(
                    [
                        torch.zeros(current_length, dtype=torch.bool),  # Real audio
                        torch.ones(padding_needed, dtype=torch.bool),  # Padded
                    ]
                )
            else:
                # Exact length, no padding needed
                padding_mask = torch.zeros(current_length, dtype=torch.bool)
        else:
            # Variable length - create empty padding mask
            padding_mask = torch.zeros(audio_tensor.shape[0], dtype=torch.bool)

        # Apply transform if any
        if self.transform:
            transformed = self.transform(audio_tensor)

            if not torch.is_tensor(transformed):
                raise TypeError("音频预处理函数必须返回 torch.Tensor")

            if transformed.dim() != 1 and transformed.dim() != 2:
                raise ValueError("预处理输出必须是 1D 波形或 2D 时频图")

            # 原始 padding mask 以变换前长度为准，需按输出时间轴重映射
            target_length = transformed.shape[0]
            source_length = padding_mask.numel()
            if target_length != source_length:
                if target_length == 0:
                    padding_mask = torch.ones(
                        target_length, dtype=torch.bool, device=padding_mask.device
                    )
                elif not padding_mask.any():
                    padding_mask = torch.zeros(
                        target_length, dtype=torch.bool, device=padding_mask.device
                    )
                else:
                    first_padding = int(torch.nonzero(padding_mask, as_tuple=False)[0, 0])
                    valid_length = min(first_padding, source_length)
                    mapped_valid_length = int(
                        round(valid_length * target_length / float(source_length))
                    )
                    mapped_valid_length = max(
                        0, min(mapped_valid_length, target_length)
                    )
                    padding_mask = torch.cat(
                        [
                            torch.zeros(mapped_valid_length, dtype=torch.bool),
                            torch.ones(
                                target_length - mapped_valid_length, dtype=torch.bool
                            ),
                        ]
                    )

            audio_tensor = transformed

        # Encode label
        label = self.label_encoder.transform([row["category"]])[0]

        return audio_tensor, padding_mask, label


class BEATsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: pd.DataFrame,
        data_dir: Path,
        config: DataConfig,
        pre_split: bool = False,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.config = config
        self.pre_split = pre_split
        self.transform = transform

        # For pre-split datasets
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        if self.pre_split:
            # Use pre-provided splits
            self._setup_pre_split()
        else:
            # Perform automatic splitting
            self._setup_auto_split()

        # Store number of classes
        self.num_classes = self.train_dataset.num_classes

    def _setup_pre_split(self):
        """Setup datasets from pre-split dataframes."""
        if self.train_df is not None:
            self.train_dataset = AudioDataset(
                self.train_df,
                self.data_dir,
                self.config.sample_rate,
                clip_duration=self.config.clip_duration,
                transform=self.transform,
            )

        if self.val_df is not None and len(self.val_df) > 0:
            self.val_dataset = AudioDataset(
                self.val_df,
                self.data_dir,
                self.config.sample_rate,
                clip_duration=self.config.clip_duration,
                transform=self.transform,
            )

        if self.test_df is not None and len(self.test_df) > 0:
            self.test_dataset = AudioDataset(
                self.test_df,
                self.data_dir,
                self.config.sample_rate,
                clip_duration=self.config.clip_duration,
                transform=self.transform,
            )

    def _setup_auto_split(self):
        """Setup datasets with automatic splitting."""
        # Shuffle dataset
        dataset_shuffled = self.dataset.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        # Split dataset
        if self.config.test_split > 0:
            train_val, test = train_test_split(
                dataset_shuffled,
                test_size=self.config.test_split,
                random_state=42,
                stratify=dataset_shuffled["category"],
            )
        else:
            train_val = dataset_shuffled
            test = pd.DataFrame()

        if self.config.val_split > 0:
            train, val = train_test_split(
                train_val,
                test_size=self.config.val_split / (1 - self.config.test_split),
                random_state=42,
                stratify=train_val["category"],
            )
        else:
            train = train_val
            val = pd.DataFrame()

        # Create datasets
        self.train_dataset = AudioDataset(
            train,
            self.data_dir,
            self.config.sample_rate,
            clip_duration=self.config.clip_duration,
            transform=self.transform,
        )

        if len(val) > 0:
            self.val_dataset = AudioDataset(
                val,
                self.data_dir,
                self.config.sample_rate,
                clip_duration=self.config.clip_duration,
                transform=self.transform,
            )

        if len(test) > 0:
            self.test_dataset = AudioDataset(
                test,
                self.data_dir,
                self.config.sample_rate,
                clip_duration=self.config.clip_duration,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )


class PreSplitDataModule(pl.LightningDataModule):
    """Data module for pre-split datasets."""

    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        train_data_dir: Optional[str] = None,
        val_data_dir: Optional[str] = None,
        test_data_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: int = 16000,
        clip_duration: Optional[float] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_data_dir = Path(train_data_dir) if train_data_dir else None
        self.val_data_dir = Path(val_data_dir) if val_data_dir else None
        self.test_data_dir = Path(test_data_dir) if test_data_dir else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.transform = transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        # Use provided data directories or try to infer from first file
        if self.train_data_dir:
            train_data_dir = self.train_data_dir
        else:
            # Fallback: extract data directory from the first training file
            first_file = self.train_data.iloc[0]["filename"]
            if "/" in str(first_file):
                # If it's a full path, extract directory
                train_data_dir = Path(first_file).parent.parent
            else:
                # If it's just a filename, assume current directory
                train_data_dir = Path(".")

        # Create datasets with their respective data directories
        self.train_dataset = AudioDataset(
            self.train_data,
            train_data_dir,
            self.sample_rate,
            clip_duration=self.clip_duration,
            transform=self.transform,
        )

        if self.val_data is not None and len(self.val_data) > 0:
            val_data_dir = self.val_data_dir if self.val_data_dir else train_data_dir
            self.val_dataset = AudioDataset(
                self.val_data,
                val_data_dir,
                self.sample_rate,
                clip_duration=self.clip_duration,
                transform=self.transform,
            )

        if self.test_data is not None and len(self.test_data) > 0:
            test_data_dir = self.test_data_dir if self.test_data_dir else train_data_dir
            self.test_dataset = AudioDataset(
                self.test_data,
                test_data_dir,
                self.sample_rate,
                clip_duration=self.clip_duration,
                transform=self.transform,
            )

        # Store number of classes
        self.num_classes = self.train_dataset.num_classes

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )
