# Long-Form Audio Prediction

Process long audio files (minutes to hours) with timestamped predictions using trained BEATs models.

## Overview

The `LongFormPredictor` enables real-world deployment by processing long audio files that exceed typical training clip lengths. It automatically:

- Splits long audio into manageable chunks
- Processes each chunk with your trained model
- Returns timestamped predictions with confidence scores
- Saves results in CSV or text format

## Quick Start

### Basic Usage

```python
from beats_trainer import BEATsTrainer, create_long_form_predictor

# Load trained model
trainer = BEATsTrainer.from_checkpoint("my_model.ckpt")

# Create long-form predictor
predictor = create_long_form_predictor(
    trainer=trainer,
    chunk_length=10.0,      # 10-second chunks
    overlap=2.0,            # 2-second overlap between chunks
    confidence_threshold=0.6 # Only report high-confidence detections
)

# Process a long audio file
results = predictor.predict_on_file(
    audio_path="long_recording.wav",
    output_path="predictions.csv"
)

print(f"Found {len(results)} detections")
```

### Results Format

The output DataFrame contains:

| Column | Description |
|--------|-------------|
| `start_time` | Start time of detection (seconds) |
| `end_time` | End time of detection (seconds) |
| `predicted_class` | Class index |
| `confidence` | Prediction confidence (0-1) |
| `class_name` | Human-readable class name |
| `prob_ClassX` | Individual class probabilities |

## Configuration Options

### Chunk Settings

```python
predictor = create_long_form_predictor(
    trainer=trainer,
    chunk_length=5.0,    # Shorter chunks = more granular detection
    overlap=1.0,         # Overlap helps catch events at boundaries
    confidence_threshold=0.7  # Higher threshold = fewer false positives
)
```

**Chunk Length Guidelines:**
- **5-10 seconds**: Good for short sounds (bird calls, alarms)
- **10-20 seconds**: Balanced for most use cases
- **20+ seconds**: For longer events (speech, music)

**Overlap Benefits:**
- Prevents missing events at chunk boundaries
- Improves detection of events that span multiple chunks
- Typical values: 10-20% of chunk length

### Processing Parameters

```python
predictor = LongFormPredictor(
    model=trainer.model,
    chunk_length=10.0,
    overlap=2.0,
    confidence_threshold=0.5,
    sample_rate=16000  # Must match training data
)
```

## Batch Processing

### Process Directory

```python
# Process all audio files in a directory
results = predictor.predict_on_directory(
    input_dir="recordings/",
    output_dir="predictions/",
    file_patterns=['*.wav', '*.mp3']  # Optional: specify formats
)

# Results is a dict mapping filename -> DataFrame
for filename, detections in results.items():
    print(f"{filename}: {len(detections)} detections")
```

### Custom File Patterns

```python
# Only process specific file types
results = predictor.predict_on_directory(
    input_dir="audio_data/",
    output_dir="results/",
    file_patterns=['*.flac', '*.m4a', '*.ogg']
)
```

## Output Formats

### CSV Format (Default)

```python
# Saves as structured CSV file
results = predictor.predict_on_file(
    audio_path="recording.wav",
    output_path="detections.csv"
)
```

### Text Format

```python
# Saves as human-readable text
results = predictor.predict_on_file(
    audio_path="recording.wav",
    output_path="detections.txt"
)
```

Example text output:
```
Audio Prediction Results
==================================================

Total detections: 15
Confidence threshold: 0.6
Chunk length: 10.0s

Detections:
--------------------------------------------------
Time: 12.50-22.50s | Class: bird_call | Confidence: 0.847
Time: 45.20-55.20s | Class: car_horn | Confidence: 0.923
Time: 67.80-77.80s | Class: dog_bark | Confidence: 0.756
```

## Real-World Examples

### Security Monitoring

```python
# Monitor long surveillance recordings
predictor = create_long_form_predictor(
    trainer=security_trainer,
    chunk_length=15.0,  # 15-second analysis windows
    overlap=3.0,        # 3-second overlap
    confidence_threshold=0.8  # High confidence for alerts
)

# Process overnight recording
alerts = predictor.predict_on_file(
    audio_path="security_cam_overnight.wav",
    output_path="security_alerts.csv"
)

# Filter for specific threats
glass_breaking = alerts[alerts['class_name'] == 'glass_breaking']
print(f"Glass breaking events: {len(glass_breaking)}")
```

### Wildlife Monitoring

```python
# Analyze ecosystem recordings
predictor = create_long_form_predictor(
    trainer=wildlife_trainer,
    chunk_length=5.0,   # Short chunks for brief bird calls
    overlap=1.0,        # Catch calls at boundaries
    confidence_threshold=0.6
)

# Process daily field recording
species = predictor.predict_on_file(
    audio_path="forest_24h.wav",
    output_path="species_detections.csv"
)

# Analyze biodiversity
unique_species = species['class_name'].unique()
print(f"Species detected: {len(unique_species)}")
```

### Quality Control

```python
# Monitor production line audio
predictor = create_long_form_predictor(
    trainer=qc_trainer,
    chunk_length=8.0,
    overlap=2.0,
    confidence_threshold=0.7
)

# Check machinery sounds
defects = predictor.predict_on_file(
    audio_path="production_line.wav",
    output_path="quality_check.csv"
)

# Alert on anomalies
anomalies = defects[defects['class_name'].str.contains('fault|error')]
if len(anomalies) > 0:
    print(f"⚠️ {len(anomalies)} potential issues detected")
```

## Analysis & Post-Processing

### Temporal Analysis

```python
# Load predictions for analysis
results = predictor.predict_on_file("recording.wav")

# Calculate temporal patterns
import matplotlib.pyplot as plt

# Plot detection timeline
plt.figure(figsize=(12, 6))
for _, detection in results.iterrows():
    plt.barh(
        detection['class_name'],
        detection['end_time'] - detection['start_time'],
        left=detection['start_time'],
        alpha=0.7
    )
plt.xlabel('Time (seconds)')
plt.title('Detection Timeline')
plt.show()
```

### Confidence Filtering

```python
# Filter by confidence levels
high_conf = results[results['confidence'] > 0.8]
medium_conf = results[results['confidence'].between(0.6, 0.8)]
low_conf = results[results['confidence'] < 0.6]

print(f"High confidence: {len(high_conf)}")
print(f"Medium confidence: {len(medium_conf)}")
print(f"Low confidence: {len(low_conf)}")
```

### Class-Specific Analysis

```python
# Analyze specific classes
bird_detections = results[results['class_name'].str.contains('bird')]
vehicle_detections = results[results['class_name'].str.contains('car|truck|engine')]

# Time-based grouping
results['hour'] = (results['start_time'] // 3600).astype(int)
hourly_counts = results.groupby(['hour', 'class_name']).size()
print("Detections by hour:")
print(hourly_counts)
```

## Performance Considerations

### Memory Usage

Large audio files are processed in chunks to manage memory:

```python
# For very large files, use smaller chunks
predictor = create_long_form_predictor(
    trainer=trainer,
    chunk_length=5.0,  # Smaller chunks use less memory
    overlap=0.5
)
```

### Processing Speed

- **GPU**: Automatically used if available and model supports it
- **Batch Processing**: Use `predict_on_directory()` for multiple files
- **Chunk Size**: Balance between accuracy and speed

### Disk Space

Output files scale with detection count:
- CSV: ~100-500 bytes per detection
- Text: ~80-200 bytes per detection
- Consider compression for large datasets

## Troubleshooting

### Common Issues

**No detections found:**
```python
# Lower confidence threshold
predictor.confidence_threshold = 0.3

# Check chunk length matches training data
predictor.chunk_length = 10.0  # Match your training clips
```

**Memory errors:**
```python
# Use smaller chunks
predictor = create_long_form_predictor(
    trainer=trainer,
    chunk_length=5.0,  # Reduce from default 10.0
    overlap=1.0
)
```

**Slow processing:**
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Reduce overlap for faster processing
predictor.overlap = 0.0  # No overlap = faster
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.INFO)

# Enable detailed logging
results = predictor.predict_on_file("test.wav")
# Shows: chunk creation, processing progress, detection counts
```

## Command Line Usage

Use the example script for batch processing:

```bash
# Process single file
python example_scripts/long_form_prediction_examples.py \
    model.ckpt audio.wav --output predictions.csv

# Process directory
python example_scripts/long_form_prediction_examples.py \
    model.ckpt audio_dir/ --output predictions/ --threshold 0.7

# Custom settings
python example_scripts/long_form_prediction_examples.py \
    model.ckpt recording.wav \
    --chunk-length 5.0 \
    --overlap 1.0 \
    --threshold 0.8 \
    --format txt
```

## Integration Examples

### Web API

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

# Load model once at startup
predictor = create_long_form_predictor(
    BEATsTrainer.from_checkpoint("production_model.ckpt")
)

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    audio_file = request.files['audio']
    results = predictor.predict_on_file(audio_file)
    return jsonify(results.to_dict('records'))
```

### Monitoring Pipeline

```python
import time
from pathlib import Path

def monitor_directory(watch_dir, output_dir):
    """Continuously monitor directory for new audio files."""
    processed = set()

    while True:
        for audio_file in Path(watch_dir).glob("*.wav"):
            if audio_file not in processed:
                print(f"Processing new file: {audio_file}")
                predictor.predict_on_file(
                    audio_file,
                    output_dir / f"{audio_file.stem}_predictions.csv"
                )
                processed.add(audio_file)

        time.sleep(10)  # Check every 10 seconds
```

This enables real-world deployment of your trained BEATs models on continuous audio streams, recordings, and production environments.
