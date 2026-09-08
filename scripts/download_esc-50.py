import urllib.request
import zipfile
import os
from pathlib import Path

dataset_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
dataset_path = "./dataset/ESC-50-master"

if not os.path.exists(dataset_path):
    print("Downloading ESC-50 dataset...")
    urllib.request.urlretrieve(dataset_url, "esc50.zip")

    print("Extracting dataset...")
    with zipfile.ZipFile("esc50.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    # Clean up zip file
    os.remove("esc50.zip")
    print("Dataset downloaded and extracted!")
else:
    print("ESC-50 dataset already exists!")

# Check dataset structure
audio_dir = Path(dataset_path) / "audio"
meta_file = Path(dataset_path) / "meta" / "esc50.csv"

print(f"Audio files directory: {audio_dir}")
print(f"Metadata file: {meta_file}")
print(f"Number of audio files: {len(list(audio_dir.glob('*.wav')))}")