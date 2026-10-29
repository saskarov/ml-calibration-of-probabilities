# Calibration of Probabilities

Statistical Methods for Machine Learning (Università degli Studi di Milano, A.Y. 2025/26), Assignment 2.

This project evaluates and improves the calibration of Logistic Regression and Random Forest on two real-world binary classification datasets (Spambase and Adult). It compares Platt scaling with isotonic regression, both implemented from scratch in numpy. Reference: Niculescu-Mizil & Caruana, *Predicting Good Probabilities with Supervised Learning*, ICML 2005.

## Reproduction

Requires Python 3.10+. From the repository root:

```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate
# Windows (PowerShell)
python -m venv .venv; .venv\Scripts\Activate.ps1

pip install -r requirements.txt && pip install -e .
python run_all.py
```

`run_all.py` executes the notebooks in `notebooks/` in order. Datasets are downloaded from OpenML into `data/` on first run. All results are written to `results/` and all figures to `figures/`.

Expected runtime: about 35–60 minutes (56 minutes measured on a 12-thread Intel Core i5-12450HX laptop with 16 GB RAM, Windows 11; most of it is the random-forest grid search on Adult). The three notebooks can also be run with `RECOMPUTE = False` in their first cell, which loads the saved CSVs from `results/` and only redraws the figures.

## Tests

```bash
pytest -q
```

## Layout

| path | contents |
|---|---|
| `src/calib/` | library code: data, models, calibrators, metrics, experiments, plotting |
| `notebooks/` | thin notebooks that run the experiments and display the results |
| `tests/` | tests checking the from-scratch implementations against scikit-learn |
| `results/` | CSV / NPZ outputs |
| `figures/` | PDF + PNG figures |
| `report/` | LaTeX source and the final `report.pdf` |
