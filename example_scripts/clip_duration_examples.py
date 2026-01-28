"""
Example demonstrating manual clip duration control.

This script shows how to set a fixed duration for all audio clips,
with automatic padding or truncation to achieve the target length.
"""

import sys
from pathlib import Path

# Add src to path so we can import beats_trainer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beats_trainer import BEATsTrainer, Config
from beats_trainer.core.config import DataConfig, ModelConfig, TrainingConfig


def example_1_second_clips():
    """Example: Force all clips to be exactly 1 second long."""
    
    print("=== Example: 1-second clips ===")
    
    # Create config with 1-second clip duration
    config = Config(
        data=DataConfig(
            sample_rate=16000,
            clip_duration=1.0,  # Force all clips to be 1 second
            batch_size=16,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1
        ),
        model=ModelConfig(
            freeze_backbone=True
        ),
        training=TrainingConfig(
            max_epochs=10,
            learning_rate=1e-4
        )
    )
    
    # Create trainer with fixed clip duration
    trainer = BEATsTrainer.from_directory(
        data_dir="/path/to/your/dataset",
        config=config
    )
    
    print(f"Sample rate: {config.data.sample_rate} Hz")
    print(f"Clip duration: {config.data.clip_duration} seconds")
    print(f"Target samples per clip: {int(config.data.clip_duration * config.data.sample_rate)}")
    
    # All audio will be processed to exactly 16,000 samples (1 second at 16kHz)
    # - Files longer than 1 second will be truncated
    # - Files shorter than 1 second will be zero-padded
    
    # Train the model
    results = trainer.train()
    return results


def example_short_clips():
    """Example: Very short clips for quick prototyping."""
    
    print("=== Example: 0.5-second clips for quick prototyping ===")
    
    config = Config(
        data=DataConfig(
            clip_duration=0.5,  # Half-second clips
            batch_size=32,      # Can use larger batches with shorter clips
            sample_rate=16000
        ),
        training=TrainingConfig(
            max_epochs=20
        )
    )
    
    trainer = BEATsTrainer.from_directory(
        data_dir="/path/to/your/dataset",
        config=config
    )
    
    print(f"Using {config.data.clip_duration}s clips ({int(config.data.clip_duration * 16000)} samples)")
    print("This is great for:")
    print("- Quick prototyping")
    print("- Testing with limited compute")
    print("- Datasets with naturally short sounds")
    
    return trainer


def example_long_clips():
    """Example: Longer clips for detailed analysis."""
    
    print("=== Example: 5-second clips for detailed analysis ===")
    
    config = Config(
        data=DataConfig(
            clip_duration=5.0,   # 5-second clips
            batch_size=8,        # Smaller batch due to longer clips
            sample_rate=16000
        ),
        training=TrainingConfig(
            max_epochs=15
        )
    )
    
    trainer = BEATsTrainer.from_directory(
        data_dir="/path/to/your/dataset", 
        config=config
    )
    
    print(f"Using {config.data.clip_duration}s clips ({int(config.data.clip_duration * 16000)} samples)")
    print("This is great for:")
    print("- Complex audio with temporal patterns")
    print("- Music classification")
    print("- Speech recognition")
    print("- Environmental sound analysis")
    
    return trainer


def example_variable_vs_fixed():
    """Compare variable length vs fixed length training."""
    
    print("=== Comparison: Variable vs Fixed Length ===")
    
    # Variable length config (default)
    variable_config = Config(
        data=DataConfig(
            clip_duration=None,  # Variable length (default)
            batch_size=16
        )
    )
    
    # Fixed length config
    fixed_config = Config(
        data=DataConfig(
            clip_duration=2.0,   # Fixed 2-second clips
            batch_size=16
        )
    )
    
    print("Variable Length Mode:")
    print("- Preserves original audio length")
    print("- Dynamic padding in batches")
    print("- More memory overhead")
    print("- Better for datasets with consistent lengths")
    
    print("\nFixed Length Mode:")
    print("- All clips exactly the same length")
    print("- Consistent memory usage")
    print("- May truncate important audio")
    print("- Better for mixed-length datasets")
    print("- Faster training (no dynamic padding)")
    
    return variable_config, fixed_config


def example_with_different_sample_rates():
    """Example: Different combinations of sample rates and clip durations."""
    
    print("=== Example: Different sample rates and durations ===")
    
    configs = [
        {
            "name": "High-res, short clips",
            "sample_rate": 22050,
            "clip_duration": 1.0,
            "samples": 22050
        },
        {
            "name": "Standard, medium clips", 
            "sample_rate": 16000,
            "clip_duration": 2.0,
            "samples": 32000
        },
        {
            "name": "Low-res, long clips",
            "sample_rate": 8000,
            "clip_duration": 4.0,
            "samples": 32000
        }
    ]
    
    for cfg in configs:
        print(f"\n{cfg['name']}:")
        print(f"  Sample rate: {cfg['sample_rate']} Hz")
        print(f"  Clip duration: {cfg['clip_duration']} seconds")
        print(f"  Samples per clip: {cfg['samples']}")
        print(f"  Memory per clip: ~{cfg['samples'] * 4 / 1024:.1f} KB")


def main():
    """Demonstrate different clip duration configurations."""
    
    print("BEATs Trainer - Manual Clip Duration Examples")
    print("=" * 50)
    
    # Show different configurations
    example_variable_vs_fixed()
    print()
    example_with_different_sample_rates()
    
    # Uncomment to run actual training examples:
    # example_1_second_clips()
    # example_short_clips() 
    # example_long_clips()
    
    print("\n" + "=" * 50)
    print("Key Benefits of Manual Clip Duration:")
    print("✓ Consistent memory usage")
    print("✓ Predictable training time")
    print("✓ Better batch efficiency")
    print("✓ Handles mixed-length datasets")
    print("✓ Easy to tune for your hardware")


if __name__ == "__main__":
    main()