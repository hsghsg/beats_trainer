"""
Example script for long-form audio prediction.

This script demonstrates how to use the LongFormPredictor to process
long audio files and get timestamped predictions.
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
import logging

# Add src to path so we can import beats_trainer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beats_trainer import BEATsTrainer, LongFormPredictor, create_long_form_predictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_file():
    """Example: Process a single long audio file."""
    
    # Load a trained model (adjust path as needed)
    trainer = BEATsTrainer.from_checkpoint("path/to/your/checkpoint.ckpt")
    
    # Create long-form predictor
    predictor = create_long_form_predictor(
        trainer=trainer,
        chunk_length=10.0,  # 10-second chunks
        overlap=2.0,        # 2-second overlap
        confidence_threshold=0.6  # Only report predictions above 60% confidence
    )
    
    # Process a long audio file
    audio_file = "path/to/long_audio.wav"
    output_file = "predictions.csv"
    
    results = predictor.predict_on_file(
        audio_path=audio_file,
        output_path=output_file
    )
    
    print(f"Found {len(results)} detections")
    print(results.head())


def example_batch_processing():
    """Example: Process multiple audio files."""
    
    # Load trained model
    trainer = BEATsTrainer.from_checkpoint("path/to/your/checkpoint.ckpt")
    
    # Create predictor with different settings
    predictor = create_long_form_predictor(
        trainer=trainer,
        chunk_length=5.0,   # Shorter chunks for more granular detection
        overlap=1.0,        # 1-second overlap
        confidence_threshold=0.5
    )
    
    # Process all audio files in a directory
    input_directory = "path/to/audio/files/"
    output_directory = "predictions_output/"
    
    results = predictor.predict_on_directory(
        input_dir=input_directory,
        output_dir=output_directory
    )
    
    # Print summary
    for filename, df in results.items():
        print(f"{filename}: {len(df)} detections")


def example_custom_analysis():
    """Example: Custom analysis of results."""
    
    # Load model and create predictor
    trainer = BEATsTrainer.from_checkpoint("path/to/your/checkpoint.ckpt")
    predictor = create_long_form_predictor(trainer=trainer)
    
    # Process file
    results = predictor.predict_on_file("path/to/audio.wav")
    
    if len(results) > 0:
        # Analyze results
        print("=== Detection Summary ===")
        print(f"Total detections: {len(results)}")
        
        # Class distribution
        class_counts = results['class_name'].value_counts()
        print(f"\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        
        # Confidence statistics
        print(f"\nConfidence statistics:")
        print(f"  Mean: {results['confidence'].mean():.3f}")
        print(f"  Max: {results['confidence'].max():.3f}")
        print(f"  Min: {results['confidence'].min():.3f}")
        
        # High-confidence detections
        high_conf = results[results['confidence'] > 0.8]
        print(f"\nHigh-confidence detections (>80%): {len(high_conf)}")
        
        # Time coverage
        total_time = results['end_time'].max() - results['start_time'].min()
        detection_time = (results['end_time'] - results['start_time']).sum()
        coverage = detection_time / total_time * 100
        print(f"Time coverage: {coverage:.1f}%")


def main():
    """CLI for long-form audio prediction."""
    parser = argparse.ArgumentParser(description="Process long audio files with BEATs")
    parser.add_argument("checkpoint", help="Path to trained model checkpoint")
    parser.add_argument("audio", help="Path to audio file or directory")
    parser.add_argument("-o", "--output", help="Output path for predictions")
    parser.add_argument("--chunk-length", type=float, default=10.0, 
                       help="Chunk length in seconds (default: 10.0)")
    parser.add_argument("--overlap", type=float, default=0.0,
                       help="Overlap between chunks in seconds (default: 0.0)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--format", choices=['csv', 'txt'], default='csv',
                       help="Output format (default: csv)")
    
    args = parser.parse_args()
    
    try:
        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        trainer = BEATsTrainer.from_checkpoint(args.checkpoint)
        
        # Create predictor
        predictor = create_long_form_predictor(
            trainer=trainer,
            chunk_length=args.chunk_length,
            overlap=args.overlap,
            confidence_threshold=args.threshold
        )
        
        # Check if input is file or directory
        audio_path = Path(args.audio)
        
        if audio_path.is_file():
            # Process single file
            logger.info(f"Processing file: {audio_path}")
            
            if args.output:
                output_path = Path(args.output)
                if args.format == 'txt':
                    output_path = output_path.with_suffix('.txt')
                else:
                    output_path = output_path.with_suffix('.csv')
            else:
                output_path = audio_path.parent / f"{audio_path.stem}_predictions.{args.format}"
            
            results = predictor.predict_on_file(
                audio_path=audio_path,
                output_path=output_path
            )
            
            print(f"Processed {audio_path.name}")
            print(f"Found {len(results)} detections")
            print(f"Results saved to {output_path}")
            
        elif audio_path.is_dir():
            # Process directory
            logger.info(f"Processing directory: {audio_path}")
            
            output_dir = Path(args.output) if args.output else audio_path / "predictions"
            
            results = predictor.predict_on_directory(
                input_dir=audio_path,
                output_dir=output_dir
            )
            
            total_detections = sum(len(df) for df in results.values())
            print(f"Processed {len(results)} files")
            print(f"Total detections: {total_detections}")
            print(f"Results saved to {output_dir}")
            
        else:
            print(f"Error: {audio_path} is not a valid file or directory")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # example_single_file()
    # example_batch_processing()  
    # example_custom_analysis()
    
    # Or run the CLI
    main()