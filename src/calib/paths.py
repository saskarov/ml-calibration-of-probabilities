"""Repository paths, resolved relative to this file so nothing is hard-coded."""

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
RESULTS_DIR: Path = ROOT / "results"
FIGURES_DIR: Path = ROOT / "figures"

for _directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
