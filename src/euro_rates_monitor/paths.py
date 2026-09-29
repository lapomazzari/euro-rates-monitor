"""Project directories. Everything is relative to the repository root (or ERM_ROOT)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ERM_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SAMPLE = ROOT / "data" / "sample"
FIGURES = ROOT / "figures"
NOTES = ROOT / "notes"
