"""Test the manual clip duration functionality."""

import pytest
import numpy as np
import tempfile
import sys
from pathlib import Path
import pandas as pd
import soundfile as sf

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beats_trainer.data.module import AudioDataset, collate_audio_batch
from beats_trainer.core.config import DataConfig


def create_test_audio_files(temp_dir: Path, durations: list) -> pd.DataFrame:
    """Create test audio files with different durations."""
    data = []
    
    for i, duration in enumerate(durations):
        # Create audio data
        sample_rate = 16000
        samples = int(duration * sample_rate)
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        
        # Save to file
        filename = f"audio_{i}_{duration}s.wav"
        filepath = temp_dir / filename
        sf.write(filepath, audio, sample_rate)
        
        data.append({
            "filename": filename,
            "category": f"class_{i % 3}",  # 3 classes
            "duration": duration
        })
    
    return pd.DataFrame(data)


def test_clip_duration_padding():
    """Test that clips shorter than target are padded correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create test files with different durations (all shorter than target)
        durations = [0.5, 0.8, 0.9]  # All shorter than 1.0 second
        df = create_test_audio_files(temp_dir, durations)
        
        # Create dataset with 1-second target duration
        dataset = AudioDataset(
            dataframe=df,
            data_dir=temp_dir,
            sample_rate=16000,
            clip_duration=1.0
        )
        
        target_samples = 16000  # 1 second at 16kHz
        
        for i in range(len(dataset)):
            audio, padding_mask, label = dataset[i]
            
            # Check audio length
            assert audio.shape[0] == target_samples, f"Audio {i} should be {target_samples} samples"
            
            # Check padding mask
            assert padding_mask.shape[0] == target_samples, f"Padding mask {i} should be {target_samples} samples"
            
            # Check that some samples are marked as padded
            original_samples = int(durations[i] * 16000)
            expected_real_samples = original_samples
            expected_padded_samples = target_samples - original_samples
            
            real_samples = (~padding_mask).sum().item()
            padded_samples = padding_mask.sum().item()
            
            assert real_samples == expected_real_samples, f"Audio {i}: Wrong number of real samples"
            assert padded_samples == expected_padded_samples, f"Audio {i}: Wrong number of padded samples"


def test_clip_duration_truncation():
    """Test that clips longer than target are truncated correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create test files with different durations (all longer than target)
        durations = [2.0, 3.5, 5.0]  # All longer than 1.0 second
        df = create_test_audio_files(temp_dir, durations)
        
        # Create dataset with 1-second target duration
        dataset = AudioDataset(
            dataframe=df,
            data_dir=temp_dir,
            sample_rate=16000,
            clip_duration=1.0
        )
        
        target_samples = 16000  # 1 second at 16kHz
        
        for i in range(len(dataset)):
            audio, padding_mask, label = dataset[i]
            
            # Check audio length
            assert audio.shape[0] == target_samples, f"Audio {i} should be {target_samples} samples"
            
            # Check padding mask (should be all False since no padding needed)
            assert padding_mask.shape[0] == target_samples, f"Padding mask {i} should be {target_samples} samples"
            assert padding_mask.sum().item() == 0, f"Audio {i}: No samples should be marked as padded"


def test_clip_duration_exact_match():
    """Test that clips exactly matching target duration work correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create test files with exact target duration
        durations = [1.0, 1.0, 1.0]  # Exactly 1.0 second
        df = create_test_audio_files(temp_dir, durations)
        
        # Create dataset with 1-second target duration
        dataset = AudioDataset(
            dataframe=df,
            data_dir=temp_dir,
            sample_rate=16000,
            clip_duration=1.0
        )
        
        target_samples = 16000  # 1 second at 16kHz
        
        for i in range(len(dataset)):
            audio, padding_mask, label = dataset[i]
            
            # Check audio length
            assert audio.shape[0] == target_samples, f"Audio {i} should be {target_samples} samples"
            
            # Check padding mask (should be all False)
            assert padding_mask.shape[0] == target_samples, f"Padding mask {i} should be {target_samples} samples"
            assert padding_mask.sum().item() == 0, f"Audio {i}: No samples should be marked as padded"


def test_collate_fixed_duration():
    """Test collate function with fixed duration clips."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create test files with mixed durations
        durations = [0.5, 2.0, 1.0]  
        df = create_test_audio_files(temp_dir, durations)
        
        # Create dataset with 1-second target duration
        dataset = AudioDataset(
            dataframe=df,
            data_dir=temp_dir,
            sample_rate=16000,
            clip_duration=1.0
        )
        
        # Create a batch
        batch = [dataset[i] for i in range(len(dataset))]
        
        # Collate batch
        audio_batch, padding_batch, label_batch = collate_audio_batch(batch)
        
        # Check batch dimensions
        assert audio_batch.shape == (3, 16000), "Audio batch should be (batch_size, samples)"
        assert padding_batch.shape == (3, 16000), "Padding batch should be (batch_size, samples)" 
        assert label_batch.shape == (3,), "Label batch should be (batch_size,)"
        
        # Check that all clips are the same length (no additional padding needed)
        assert audio_batch.shape[1] == 16000, "All clips should be exactly 16000 samples"


def test_variable_vs_fixed_duration():
    """Compare variable duration vs fixed duration behavior."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create test files with very different durations
        durations = [0.3, 1.7, 0.9, 2.8]  
        df = create_test_audio_files(temp_dir, durations)
        
        # Variable duration dataset
        dataset_variable = AudioDataset(
            dataframe=df,
            data_dir=temp_dir,
            sample_rate=16000,
            clip_duration=None  # Variable length
        )
        
        # Fixed duration dataset  
        dataset_fixed = AudioDataset(
            dataframe=df,
            data_dir=temp_dir,
            sample_rate=16000,
            clip_duration=2.0  # Fixed 2 seconds
        )
        
        # Check variable length dataset
        variable_lengths = []
        for i in range(len(dataset_variable)):
            audio, _, _ = dataset_variable[i]
            variable_lengths.append(audio.shape[0])
        
        # Variable lengths should be different
        assert len(set(variable_lengths)) > 1, "Variable dataset should have different lengths"
        
        # Check fixed length dataset
        fixed_lengths = []
        for i in range(len(dataset_fixed)):
            audio, _, _ = dataset_fixed[i]
            fixed_lengths.append(audio.shape[0])
        
        # Fixed lengths should all be the same
        assert len(set(fixed_lengths)) == 1, "Fixed dataset should have same length for all clips"
        assert fixed_lengths[0] == 32000, "Fixed clips should be 32000 samples (2 seconds at 16kHz)"


def test_different_sample_rates():
    """Test clip duration with different sample rates."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        durations = [0.8, 1.5]  
        df = create_test_audio_files(temp_dir, durations)
        
        # Test with different sample rates
        test_configs = [
            {"sample_rate": 8000, "clip_duration": 1.0, "expected_samples": 8000},
            {"sample_rate": 16000, "clip_duration": 1.0, "expected_samples": 16000},
            {"sample_rate": 22050, "clip_duration": 1.0, "expected_samples": 22050},
        ]
        
        for config in test_configs:
            dataset = AudioDataset(
                dataframe=df,
                data_dir=temp_dir,
                sample_rate=config["sample_rate"],
                clip_duration=config["clip_duration"]
            )
            
            for i in range(len(dataset)):
                audio, padding_mask, label = dataset[i]
                
                assert audio.shape[0] == config["expected_samples"], \
                    f"Audio should be {config['expected_samples']} samples at {config['sample_rate']}Hz"
                assert padding_mask.shape[0] == config["expected_samples"], \
                    f"Padding mask should be {config['expected_samples']} samples"


if __name__ == "__main__":
    # Run tests
    test_clip_duration_padding()
    test_clip_duration_truncation()
    test_clip_duration_exact_match()
    test_collate_fixed_duration()
    test_variable_vs_fixed_duration()
    test_different_sample_rates()
    
    print("✅ All clip duration tests passed!")