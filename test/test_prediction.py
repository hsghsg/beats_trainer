"""Test the long-form prediction functionality."""

import numpy as np
import tempfile
import sys
from pathlib import Path
import torch
import soundfile as sf

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beats_trainer.prediction import LongFormPredictor, create_long_form_predictor


class MockModel:
    """Mock model for testing."""

    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.training = False

    def eval(self):
        """Set to eval mode."""
        self.training = False

    def parameters(self):
        """Mock parameters method."""
        # Return a generator of dummy parameters that are on CPU
        yield torch.tensor([1.0])

    def __call__(self, x):
        """Mock forward pass."""
        batch_size = x.shape[0]
        # Return random logits
        return torch.randn(batch_size, self.num_classes)

    def predict_step(self, x, batch_idx):
        """Mock predict_step for PyTorch Lightning compatibility."""
        return self(x)


class MockTrainer:
    """Mock trainer for testing."""

    def __init__(self):
        self.model = MockModel()

        # Mock config
        class MockConfig:
            class MockData:
                sample_rate = 16000

            data = MockData()

        self.config = MockConfig()


def create_test_audio(duration=30.0, sample_rate=16000):
    """Create test audio data."""
    samples = int(duration * sample_rate)
    # Create simple sine wave
    t = np.linspace(0, duration, samples)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio


def test_long_form_predictor_initialization():
    """Test that LongFormPredictor initializes correctly."""
    model = MockModel(num_classes=5)

    predictor = LongFormPredictor(
        model=model, chunk_length=10.0, overlap=2.0, confidence_threshold=0.6
    )

    assert predictor.chunk_length == 10.0
    assert predictor.overlap == 2.0
    assert predictor.confidence_threshold == 0.6
    assert len(predictor.class_names) == 5


def test_create_long_form_predictor():
    """Test the convenience function to create predictor from trainer."""
    trainer = MockTrainer()

    predictor = create_long_form_predictor(
        trainer=trainer, chunk_length=5.0, overlap=1.0, confidence_threshold=0.7
    )

    assert predictor.chunk_length == 5.0
    assert predictor.overlap == 1.0
    assert predictor.confidence_threshold == 0.7


def test_create_chunks():
    """Test audio chunking functionality."""
    model = MockModel()
    predictor = LongFormPredictor(model=model, chunk_length=5.0, overlap=1.0)

    # Create 12-second audio
    audio = create_test_audio(duration=12.0)
    chunks = predictor.create_chunks(audio, 12.0)

    # Should create 3 chunks: 0-5s, 4-9s, 8-12s (with 1s overlap)
    assert len(chunks) >= 2  # At least 2 chunks

    # Check chunk format
    chunk_audio, start_time, end_time = chunks[0]
    assert isinstance(chunk_audio, np.ndarray)
    assert isinstance(start_time, float)
    assert isinstance(end_time, float)
    assert start_time < end_time


def test_predict_chunk():
    """Test prediction on a single chunk."""
    model = MockModel(num_classes=3)
    predictor = LongFormPredictor(model=model)

    # Create test chunk
    chunk = create_test_audio(duration=5.0)

    predicted_class, confidence, all_probs = predictor.predict_chunk(chunk)

    # Check output format
    assert isinstance(predicted_class, int)
    assert 0 <= predicted_class < 3
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1
    assert isinstance(all_probs, np.ndarray)
    assert len(all_probs) == 3
    assert np.allclose(all_probs.sum(), 1.0, atol=1e-6)  # Probabilities sum to 1


def test_predict_on_file():
    """Test prediction on a full audio file."""
    model = MockModel(num_classes=4)
    predictor = LongFormPredictor(
        model=model,
        chunk_length=5.0,
        confidence_threshold=0.0,  # Accept all predictions for testing
    )

    # Create test audio file
    audio = create_test_audio(duration=15.0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, 16000)
        temp_audio_path = f.name

    try:
        # Predict on file
        results = predictor.predict_on_file(temp_audio_path)

        # Check results format
        assert len(results) > 0  # Should have some predictions

        required_columns = [
            "start_time",
            "end_time",
            "predicted_class",
            "confidence",
            "class_name",
        ]
        for col in required_columns:
            assert col in results.columns

        # Check data types
        assert results["start_time"].dtype in [np.float64, float]
        assert results["end_time"].dtype in [np.float64, float]
        assert results["predicted_class"].dtype in [np.int64, int]
        assert results["confidence"].dtype in [np.float64, float]

        # Check logical constraints
        assert (results["start_time"] < results["end_time"]).all()
        assert (results["confidence"] >= 0).all()
        assert (results["confidence"] <= 1).all()

    finally:
        # Clean up
        Path(temp_audio_path).unlink()


def test_save_results():
    """Test saving results to different formats."""
    model = MockModel()
    predictor = LongFormPredictor(model=model, confidence_threshold=0.0)

    # Create test audio
    audio = create_test_audio(duration=10.0)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Save audio file
        audio_file = temp_dir / "test.wav"
        sf.write(audio_file, audio, 16000)

        # Test CSV output
        csv_output = temp_dir / "results.csv"
        results = predictor.predict_on_file(audio_file, csv_output)
        assert csv_output.exists()
        assert len(results) > 0

        # Test TXT output
        txt_output = temp_dir / "results.txt"
        predictor.predict_on_file(audio_file, txt_output)
        assert txt_output.exists()

        # Check text file content
        txt_content = txt_output.read_text()
        assert "Audio Prediction Results" in txt_content
        assert "Total detections:" in txt_content


def test_confidence_threshold():
    """Test that confidence threshold filtering works."""
    model = MockModel()

    # Test with high threshold (should get fewer results)
    predictor_strict = LongFormPredictor(model=model, confidence_threshold=0.9)

    # Test with low threshold (should get more results)
    predictor_lenient = LongFormPredictor(model=model, confidence_threshold=0.1)

    audio = create_test_audio(duration=10.0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, 16000)
        temp_path = f.name

    try:
        results_strict = predictor_strict.predict_on_file(temp_path)
        results_lenient = predictor_lenient.predict_on_file(temp_path)

        # Lenient threshold should generally give more results
        # (though it's possible random predictions could violate this occasionally)
        assert len(results_lenient) >= len(results_strict)

        # All results should meet threshold
        if len(results_strict) > 0:
            assert (results_strict["confidence"] >= 0.9).all()
        if len(results_lenient) > 0:
            assert (results_lenient["confidence"] >= 0.1).all()

    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    # Run tests
    test_long_form_predictor_initialization()
    test_create_long_form_predictor()
    test_create_chunks()
    test_predict_chunk()
    test_predict_on_file()
    test_save_results()
    test_confidence_threshold()

    print("✅ All tests passed!")
