"""
Configuration module for project paths.
Defines base directory, data directories, and model save directory,
and ensures they exist on import.
"""

from pathlib import Path

# Base directory of the project (one level up from src/)
BASE_DIR: Path = Path(__file__).parent.parent

# Directories for raw data, processed data, and saved models
RAW_DATA_DIR: Path = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR: Path = BASE_DIR / 'data' / 'processed'
MODEL_SAVE_DIR: Path = BASE_DIR / 'outputs' / 'models'

# Automatically create these directories if they don't exist
for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_SAVE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
