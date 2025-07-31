from pathlib import Path
import pandas as pd
import logging
import re
from typing import Tuple, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_labels_from_filename(filename: str) -> Tuple[float, float]:
    """
    Extract Pb and Cd concentrations from a filename.
    Expected patterns like "Pb0.1_Cd0_L1.swp" or "Sample Pb 0.1 – Cd 0 µM.xlsx".
    Returns:
        pb (float), cd (float)
    Raises:
        ValueError: if labels cannot be parsed.
    """
    pattern = r'Pb\s*([0-9]*\.?[0-9]+).*?Cd\s*([0-9]*\.?[0-9]+)'
    match = re.search(pattern, filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse Pb and Cd from filename: {filename}")
    return float(match.group(1)), float(match.group(2))


def parse_swp_file(filepath: Path) -> pd.DataFrame:
    """
    Robustly parse a .swp file into a DataFrame with columns ['X', 'Y'].
    - Skips comments and empty lines.
    - Splits on whitespace and takes the first two numeric tokens per line.
    """
    if not filepath.exists():
        logger.error(f"SWP file not found: {filepath}")
        raise FileNotFoundError(f"{filepath} not found.")

    data: List[Tuple[float, float]] = []
    with filepath.open('r') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = re.split(r'\s+', line)
            nums: List[float] = []
            for tok in tokens:
                try:
                    nums.append(float(tok))
                except ValueError:
                    continue
            if len(nums) >= 2:
                data.append((nums[0], nums[1]))
            else:
                logger.debug(f"Line {line_no} skipped in {filepath.name}: tokens={tokens}")

    if not data:
        logger.error(f"No numeric data parsed from {filepath}")
        raise ValueError(f"No valid (X,Y) pairs found in {filepath}")

    return pd.DataFrame(data, columns=['X', 'Y'])


def load_dataset(directory: Path) -> pd.DataFrame:
    """
    Load all data files in a directory. Supports .swp, .xlsx/.xls, .csv.
    Each file must have Pb and Cd labels in its filename.
    Returns a single DataFrame with columns ['X', 'Y', 'pb', 'cd', 'sample_id'].
    """
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Data folder {directory} does not exist or is not a directory")
        raise FileNotFoundError(f"Folder not found: {directory}")

    frames: List[pd.DataFrame] = []
    for file_path in directory.iterdir():
        suffix = file_path.suffix.lower()
        try:
            if suffix == '.swp':
                df = parse_swp_file(file_path)
            elif suffix in {'.xlsx', '.xls'}:
                df = pd.read_excel(file_path, engine='openpyxl')
            elif suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                logger.warning(f"Skipping unsupported file type: {file_path.name}")
                continue
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue

        try:
            pb, cd = extract_labels_from_filename(file_path.name)
        except ValueError as ve:
            logger.error(ve)
            continue

        df['pb'] = pb
        df['cd'] = cd
        df['sample_id'] = file_path.stem

        frames.append(df)
        logger.info(f"Loaded {len(df)} rows from {file_path.name} (Pb={pb}, Cd={cd})")

    if not frames:
        logger.error(f"No valid data loaded from {directory}")
        raise ValueError(f"No valid data in folder: {directory}")

    return pd.concat(frames, ignore_index=True)
