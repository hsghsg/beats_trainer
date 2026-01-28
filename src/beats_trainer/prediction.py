"""
Long-form audio prediction module for processing full audio files.

This module provides functionality to process long audio files (minutes to hours)
by splitting them into chunks, predicting on each chunk, and returning
timestamped predictions.
"""

import numpy as np
import pandas as pd
import torch
import librosa
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LongFormPredictor:
    """
    Predict on long audio files by chunking and processing each segment.

    This class handles the complexity of processing long audio files that are
    too large to process as a single input, common in real-world scenarios
    like surveillance, monitoring, or continuous audio analysis.
    """

    def __init__(
        self,
        model,
        chunk_length: float = 10.0,
        overlap: float = 0.0,
        confidence_threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        """
        Initialize the long-form predictor.

        Args:
            model: Trained BEATsLightningModule or similar model
            chunk_length: Length of each chunk in seconds (default: 10.0)
            overlap: Overlap between chunks in seconds (default: 0.0)
            confidence_threshold: Minimum confidence to report a detection
            sample_rate: Target sample rate for processing
        """
        self.model = model
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate

        # Set model to evaluation mode
        self.model.eval()

        # Get class names if available
        if hasattr(model, "class_names"):
            self.class_names = model.class_names
        elif hasattr(model, "config") and hasattr(model.config.data, "classes"):
            self.class_names = model.config.data.classes
        else:
            # Fallback to numeric class names
            num_classes = model.num_classes if hasattr(model, "num_classes") else 10
            self.class_names = [f"Class_{i}" for i in range(num_classes)]

        logger.info(
            f"Initialized LongFormPredictor with {len(self.class_names)} classes"
        )
        logger.info(f"Chunk length: {chunk_length}s, Overlap: {overlap}s")

    def load_and_preprocess_audio(
        self, audio_path: Union[str, Path]
    ) -> Tuple[np.ndarray, float]:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_array, duration_in_seconds)
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(audio) / sr

        logger.info(
            f"Loaded audio: {audio_path.name}, Duration: {duration:.2f}s, Sample rate: {sr}"
        )

        return audio, duration

    def create_chunks(
        self, audio: np.ndarray, duration: float
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio into overlapping chunks.

        Args:
            audio: Audio array
            duration: Total duration in seconds

        Returns:
            List of (chunk_audio, start_time, end_time) tuples
        """
        chunks = []

        chunk_samples = int(self.chunk_length * self.sample_rate)
        overlap_samples = int(self.overlap * self.sample_rate)
        step_samples = chunk_samples - overlap_samples

        start_sample = 0

        while start_sample < len(audio):
            end_sample = min(start_sample + chunk_samples, len(audio))

            # Extract chunk
            chunk = audio[start_sample:end_sample]

            # Pad if chunk is too short
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), "constant")

            # Calculate time stamps
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate

            chunks.append((chunk, start_time, end_time))

            # Move to next chunk
            start_sample += step_samples

            # Break if we've covered the whole file
            if end_sample >= len(audio):
                break

        logger.info(f"Created {len(chunks)} chunks of {self.chunk_length}s each")
        return chunks

    def predict_chunk(self, chunk: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predict on a single audio chunk.

        Args:
            chunk: Audio chunk as numpy array

        Returns:
            Tuple of (predicted_class_idx, confidence, all_probabilities)
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0)

            # Move to same device as model
            if next(self.model.parameters()).is_cuda:
                chunk_tensor = chunk_tensor.cuda()

            # Get prediction
            if hasattr(self.model, "predict_step"):
                # Use predict_step if available (PyTorch Lightning)
                outputs = self.model.predict_step(chunk_tensor, 0)
                logits = (
                    outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
                )
            else:
                # Direct forward pass
                logits = self.model(chunk_tensor)

            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=-1)

            # Get prediction
            predicted_class = torch.argmax(probabilities, dim=-1).cpu().item()
            confidence = probabilities.max().cpu().item()
            all_probs = probabilities.squeeze().cpu().numpy()

            return predicted_class, confidence, all_probs

    def predict_on_file(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        return_dataframe: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Predict on a full audio file and return timestamped results.

        Args:
            audio_path: Path to audio file
            output_path: Path to save results (optional)
            return_dataframe: Whether to return results as DataFrame

        Returns:
            DataFrame with columns: start_time, end_time, predicted_class,
            confidence, class_name (if return_dataframe=True)
        """
        audio_path = Path(audio_path)

        # Load audio
        audio, duration = self.load_and_preprocess_audio(audio_path)

        # Create chunks
        chunks = self.create_chunks(audio, duration)

        # Process each chunk
        results = []

        logger.info(f"Processing {len(chunks)} chunks...")

        for i, (chunk, start_time, end_time) in enumerate(chunks):
            # Predict on chunk
            predicted_class, confidence, all_probs = self.predict_chunk(chunk)

            # Only include predictions above threshold
            if confidence >= self.confidence_threshold:
                class_name = self.class_names[predicted_class]

                result = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "class_name": class_name,
                }

                # Add all class probabilities
                for j, prob in enumerate(all_probs):
                    result[f"prob_{self.class_names[j]}"] = prob

                results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks")

        logger.info(
            f"Found {len(results)} detections above threshold {self.confidence_threshold}"
        )

        if not results:
            logger.warning("No detections above confidence threshold found")
            if return_dataframe:
                return pd.DataFrame()
            return None

        # Create DataFrame
        if return_dataframe or output_path:
            df = pd.DataFrame(results)

            # Sort by start time
            df = df.sort_values("start_time").reset_index(drop=True)

            # Save to file if requested
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if output_path.suffix.lower() == ".csv":
                    df.to_csv(output_path, index=False)
                elif output_path.suffix.lower() == ".txt":
                    self._save_as_text(df, output_path)
                else:
                    # Default to CSV
                    output_path = output_path.with_suffix(".csv")
                    df.to_csv(output_path, index=False)

                logger.info(f"Results saved to: {output_path}")

            if return_dataframe:
                return df

        return None

    def _save_as_text(self, df: pd.DataFrame, output_path: Path):
        """Save results as formatted text file."""
        with open(output_path, "w") as f:
            f.write("Audio Prediction Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total detections: {len(df)}\n")
            f.write(f"Confidence threshold: {self.confidence_threshold}\n")
            f.write(f"Chunk length: {self.chunk_length}s\n\n")

            f.write("Detections:\n")
            f.write("-" * 50 + "\n")

            for _, row in df.iterrows():
                f.write(f"Time: {row['start_time']:.2f}-{row['end_time']:.2f}s | ")
                f.write(f"Class: {row['class_name']} | ")
                f.write(f"Confidence: {row['confidence']:.3f}\n")

    def predict_on_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_patterns: List[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Process all audio files in a directory.

        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save results
            file_patterns: List of file patterns to match (default: common audio formats)

        Returns:
            Dictionary mapping filename to results DataFrame
        """
        if file_patterns is None:
            file_patterns = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        audio_files = []
        for pattern in file_patterns:
            audio_files.extend(input_dir.glob(pattern))

        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return {}

        logger.info(f"Found {len(audio_files)} audio files to process")

        results = {}

        for audio_file in audio_files:
            logger.info(f"Processing: {audio_file.name}")

            try:
                output_file = output_dir / f"{audio_file.stem}_predictions.csv"
                df = self.predict_on_file(audio_file, output_file)
                results[audio_file.name] = df

            except Exception as e:
                logger.error(f"Error processing {audio_file.name}: {e}")
                results[audio_file.name] = pd.DataFrame()

        logger.info(f"Batch processing complete. Results saved to {output_dir}")
        return results


def create_long_form_predictor(
    trainer,
    chunk_length: float = 10.0,
    overlap: float = 0.0,
    confidence_threshold: float = 0.5,
) -> LongFormPredictor:
    """
    Create a long-form predictor from a trained BEATsTrainer.

    Args:
        trainer: Trained BEATsTrainer instance
        chunk_length: Length of each chunk in seconds
        overlap: Overlap between chunks in seconds
        confidence_threshold: Minimum confidence for detections

    Returns:
        LongFormPredictor instance
    """
    if not hasattr(trainer, "model") or trainer.model is None:
        raise ValueError("Trainer must have a trained model")

    return LongFormPredictor(
        model=trainer.model,
        chunk_length=chunk_length,
        overlap=overlap,
        confidence_threshold=confidence_threshold,
        sample_rate=trainer.config.data.sample_rate
        if hasattr(trainer, "config")
        else 16000,
    )
