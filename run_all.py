"""Execute every notebook headlessly, in order, failing loudly on the first error.

Usage (from the repository root):  python run_all.py
"""

import subprocess
import sys

from calib.paths import ROOT

NOTEBOOKS = [
    "01_data_exploration.ipynb",
    "02_main_experiments.ipynb",
    "03_extra_experiments.ipynb",
]


def main() -> None:
    for name in NOTEBOOKS:
        path = ROOT / "notebooks" / name
        if not path.exists():
            sys.exit(f"Missing notebook: {path}")
        print(f"Running {name} ...", flush=True)
        subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=-1",
                str(path),
            ],
            check=True,  # raises CalledProcessError if a notebook fails
            cwd=ROOT,
        )
    print("All notebooks executed successfully.")


if __name__ == "__main__":
    main()
