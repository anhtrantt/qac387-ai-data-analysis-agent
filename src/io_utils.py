#io_utils
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(reports: Path) -> None:
    """Create output folders."""
    # BLANK 1: create the figures folder
    # HINT: (reports / "figures").mkdir(...)
    (reports / "figures").mkdir(parents=True, exist_ok=True)


def read_data(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with basic error handling."""
    # BLANK 2: raise FileNotFoundError if path does not exist
    if not path.exists():
        raise FileNotFoundError("Path doesn't exist")

    # BLANK 3: read the CSV into df
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df
