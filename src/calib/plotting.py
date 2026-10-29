"""Figures and the main table for the report.

Every function reads its inputs from results/, saves figures/<name>.pdf (for LaTeX)
and figures/<name>.png (for the README and notebooks), and returns the figure.
Sizes are in inches for an A4 LaTeX page: full width 6.3, half width 3.1.

Each method has a fixed colour, line style and marker in every figure, so the
curves stay distinguishable in grayscale and for colour-blind readers.
"""

import functools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

from calib.calibration import sigmoid
from calib.metrics import reliability_bins
from calib.paths import FIGURES_DIR, RESULTS_DIR

FULL_WIDTH = 6.3
HALF_WIDTH = 3.1

METHODS = ("uncalibrated", "platt", "isotonic")
METHOD_LABEL = {"uncalibrated": "Uncalibrated", "platt": "Platt", "isotonic": "Isotonic"}
METHOD_COLOR = {"uncalibrated": "#2a78d6", "platt": "#eb6834", "isotonic": "#1baf7a"}
METHOD_LINESTYLE = {"uncalibrated": "-", "platt": "--", "isotonic": "-."}
METHOD_MARKER = {"uncalibrated": "o", "platt": "s", "isotonic": "^"}

MODELS = ("logreg", "rf")
MODEL_LABEL = {"logreg": "Logistic regression", "rf": "Random forest"}
DATASETS = ("spambase", "adult")
DATASET_LABEL = {"spambase": "Spambase", "adult": "Adult"}
METRIC_LABEL = {"accuracy": "Accuracy", "log_loss": "Log-loss", "brier": "Brier score",
                "ece": "ECE", "auc": "ROC AUC"}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8985"  # diagonal and other reference lines

STYLE = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "pdf.fonttype": 42,  # embed TrueType fonts in the PDF
}


def _styled(plot_function):
    """Run a plotting function with the report style (without changing global rcParams)."""
    @functools.wraps(plot_function)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            fig = plot_function(*args, **kwargs)
            # The first figure activates the backend, which may switch interactive mode on
            # (Jupyter's inline backend does). Keep that setting instead of letting
            # rc_context restore the old one, otherwise later figures are not displayed.
            interactive = matplotlib.is_interactive()
        matplotlib.interactive(interactive)
        return fig
    return wrapper


def save_figure(fig: Figure, name: str) -> None:
    """Save figures/<name>.pdf and figures/<name>.png."""
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png")


def _method_style(method: str) -> dict:
    return {"color": METHOD_COLOR[method], "linestyle": METHOD_LINESTYLE[method],
            "marker": METHOD_MARKER[method], "label": METHOD_LABEL[method]}


def _method_handles(methods=METHODS) -> list[Line2D]:
    return [Line2D([], [], **_method_style(m)) for m in methods]


def _log_axis(ax, which: str) -> None:
    """Log scale on the x or y axis, with plain decimal labels (0.3, 0.5, 1, 2) instead of 3x10^-1."""
    if which == "x":
        ax.set_xscale("log")
        axis = ax.xaxis
    else:
        ax.set_yscale("log")
        axis = ax.yaxis
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v))) in (2, 3, 5) else ""
    ))


def load_preds(dataset: str) -> dict[str, np.ndarray]:
    """Seed-0 test predictions and fitted maps saved by experiments.run_main."""
    with np.load(RESULTS_DIR / f"preds_{dataset}_seed0.npz") as f:
        return dict(f)


# ---------------------------------------------------------------- T1


def make_table_t1(decimals: int = 4) -> pd.DataFrame:
    """Table T1: mean ± std over seeds of every metric, per dataset, model and method.

    Reads results/metrics_summary.csv; writes figures/table_t1.tex (a complete
    booktabs tabular, included in the report with \\input) and returns the table.
    """
    summary = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")
    metrics = list(METRIC_LABEL)
    table = pd.DataFrame({
        "Dataset": summary["dataset"].map(DATASET_LABEL),
        "Model": summary["model"].map({"logreg": "LR", "rf": "RF"}),
        "Method": summary["method"].map(METHOD_LABEL),
    })
    for m in metrics:
        table[METRIC_LABEL[m]] = [
            f"{mean:.{decimals}f} ± {std:.{decimals}f}"
            for mean, std in zip(summary[f"{m}_mean"], summary[f"{m}_std"])
        ]

    lines = [r"\begin{tabular}{lll" + "c" * len(metrics) + "}",
             r"\toprule",
             " & ".join(table.columns) + r" \\",
             r"\midrule"]
    previous_group = None
    for _, row in table.iterrows():
        group = (row["Dataset"], row["Model"])
        if previous_group is not None and group != previous_group:
            lines.append(r"\midrule" if group[0] != previous_group[0] else r"\addlinespace")
        cells = [str(v).replace("±", r"$\pm$") for v in row]
        lines.append(" & ".join(cells) + r" \\")
        previous_group = group
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FIGURES_DIR / "table_t1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


# ---------------------------------------------------------------- F1


@_styled
def plot_reliability(dataset: str, n_bins: int = 10) -> Figure:
    """F1: reliability diagrams (seed 0, test set), one column per model.

    Top: observed fraction of positives vs mean predicted probability per bin
    (equal-width bins; empty bins are not drawn), with the diagonal y = x.
    Bottom: histogram of the predicted probabilities (log scale).
    """
    preds = load_preds(dataset)
    y = preds["y_test"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 3.5), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [3, 1.2]})
    hist_edges = np.linspace(0, 1, 21)
    for col, model in enumerate(MODELS):
        ax, ax_hist = axes[0, col], axes[1, col]
        ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1, linestyle=":")
        for method in METHODS:
            p = preds[f"{model}_p_{method}"]
            bins = reliability_bins(y, p, n_bins=n_bins)
            keep = bins.count > 0
            ax.plot(bins.mean_pred[keep], bins.frac_pos[keep], **_method_style(method))
            ax_hist.hist(p, bins=hist_edges, histtype="step", color=METHOD_COLOR[method],
                         linestyle=METHOD_LINESTYLE[method], linewidth=1.2)
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel("Predicted probability")
        if col == 0:
            ax.set_ylabel("Observed fraction of positives")
            ax_hist.set_ylabel("Count")

    handles = _method_handles() + [
        Line2D([], [], color=REFERENCE, linewidth=1, linestyle=":", label="Perfect calibration")
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    save_figure(fig, f"f1_reliability_{dataset}")
    return fig


# ---------------------------------------------------------------- F2


def _binned_frequencies(s: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean score and fraction of positives in n_bins groups of equal size, sorted by score."""
    order = np.argsort(s, kind="stable")
    groups = np.array_split(order, n_bins)
    return (np.array([s[g].mean() for g in groups]), np.array([y[g].mean() for g in groups]))


@_styled
def plot_correction_maps(n_bins: int = 20, quantile_range: tuple[float, float] = (0.05, 0.95)
                         ) -> Figure:
    """F2: the learned maps score -> probability (seed 0), per dataset and model.

    Uncalibrated: sigmoid(s) (for both models, since s is the logit of predict_proba).
    Platt: sigmoid(a s + b). Isotonic: the fitted step function. Grey dots: the
    calibration set in n_bins equal-count groups (mean score, fraction of positives).
    The x-axis covers the 5th to 95th percentile of the calibration scores.
    """
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.0), sharey=True, layout="constrained")
    for row, dataset in enumerate(DATASETS):
        preds = load_preds(dataset)
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            s_cal = preds[f"{model}_s_cal"]
            lo, hi = np.quantile(s_cal, quantile_range)
            grid = np.linspace(lo, hi, 500)

            ax.plot(grid, sigmoid(grid), color=METHOD_COLOR["uncalibrated"],
                    linestyle=METHOD_LINESTYLE["uncalibrated"], label=METHOD_LABEL["uncalibrated"])
            a, b = preds[f"{model}_platt_ab"]
            ax.plot(grid, sigmoid(a * grid + b), color=METHOD_COLOR["platt"],
                    linestyle=METHOD_LINESTYLE["platt"], label=METHOD_LABEL["platt"])

            # Step function: block k's value holds from its min score to the next block's min.
            starts, values = preds[f"{model}_iso_min"], preds[f"{model}_iso_value"]
            idx = np.clip(np.searchsorted(starts, grid, side="right") - 1, 0, len(values) - 1)
            ax.step(grid, values[idx], where="post", color=METHOD_COLOR["isotonic"],
                    linestyle=METHOD_LINESTYLE["isotonic"], label=METHOD_LABEL["isotonic"])

            s_bin, f_bin = _binned_frequencies(s_cal, preds["y_cal"], n_bins)
            inside = (s_bin >= lo) & (s_bin <= hi)
            ax.plot(s_bin[inside], f_bin[inside], linestyle="none", marker="o", markersize=3,
                    color=REFERENCE, label="Calibration set (binned)")

            ax.set_xlim(lo, hi)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{DATASET_LABEL[dataset]}, {MODEL_LABEL[model]}\n"
                         f"Platt a = {a:.2f}, b = {b:.2f}; isotonic {len(values)} blocks")
            ax.set_xlabel("Score s")
            if col == 0:
                ax.set_ylabel("Calibrated probability")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4)
    save_figure(fig, "f2_correction_maps")
    return fig


# ---------------------------------------------------------------- F3


@_styled
def plot_histograms(n_bins: int = 20) -> Figure:
    """F3: histograms of test-set predicted probabilities (seed 0), before and after calibration.

    One row per (dataset, model), one column per method; rows share the y-axis.
    """
    rows = [(d, m) for d in DATASETS for m in MODELS]
    fig, axes = plt.subplots(len(rows), len(METHODS), figsize=(FULL_WIDTH, 4.6), sharex=True,
                             sharey="row", layout="constrained")
    edges = np.linspace(0, 1, n_bins + 1)
    for r, (dataset, model) in enumerate(rows):
        preds = load_preds(dataset)
        for c, method in enumerate(METHODS):
            ax = axes[r, c]
            ax.hist(preds[f"{model}_p_{method}"], bins=edges, color=METHOD_COLOR[method],
                    edgecolor="white", linewidth=0.5)
            if r == 0:
                ax.set_title(METHOD_LABEL[method])
            if c == 0:
                ax.set_ylabel(f"{DATASET_LABEL[dataset]}\n{'LR' if model == 'logreg' else 'RF'}: count")
            if r == len(rows) - 1:
                ax.set_xlabel("Predicted probability")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
    save_figure(fig, "f3_histograms")
    return fig


# ---------------------------------------------------------------- F4


@_styled
def plot_calibration_size(metrics=("log_loss", "brier", "ece"), log_y=("log_loss",)) -> Figure:
    """F4: test metric vs calibration-set size (mean ± std band over seeds), Platt vs isotonic.

    One row per metric, one column per model; the dotted line is the uncalibrated model.
    Reads results/cal_size.csv.
    """
    df = pd.read_csv(RESULTS_DIR / "cal_size.csv")
    dataset = df["dataset"].iloc[0]
    n_seeds = df["seed"].nunique()
    fig, axes = plt.subplots(len(metrics), len(MODELS), figsize=(FULL_WIDTH, 1.5 * len(metrics) + 0.6),
                             sharex=True, squeeze=False, layout="constrained")
    for c, model in enumerate(MODELS):
        sub = df[df["model"] == model]
        for r, metric in enumerate(metrics):
            ax = axes[r, c]
            uncal = sub.loc[sub["method"] == "uncalibrated", metric].mean()
            ax.axhline(uncal, color=METHOD_COLOR["uncalibrated"], linestyle=":", linewidth=1.2)
            for method in ("platt", "isotonic"):
                stats = (sub[sub["method"] == method].groupby("cal_size")[metric]
                         .agg(["mean", "std"]).reset_index())
                ax.fill_between(stats["cal_size"], stats["mean"] - stats["std"],
                                stats["mean"] + stats["std"], color=METHOD_COLOR[method],
                                alpha=0.18, linewidth=0)
                ax.plot(stats["cal_size"], stats["mean"], **_method_style(method))
            ax.set_xscale("log")
            sizes = sorted(sub.loc[sub["method"] != "uncalibrated", "cal_size"].unique())
            ax.set_xticks(sizes, [f"{n // 1000}k" if n >= 1000 else str(n) for n in sizes])
            ax.xaxis.set_minor_locator(NullLocator())
            if metric in log_y:
                _log_axis(ax, "y")
            if r == 0:
                ax.set_title(MODEL_LABEL[model])
            if c == 0:
                ax.set_ylabel(METRIC_LABEL[metric])
            if r == len(metrics) - 1:
                ax.set_xlabel("Calibration-set size")

    handles = _method_handles(("platt", "isotonic")) + [
        Line2D([], [], color=METHOD_COLOR["uncalibrated"], linestyle=":", linewidth=1.2,
               label="Uncalibrated")
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               title=f"{DATASET_LABEL.get(dataset, dataset)}, test set, mean ± std over {n_seeds} seeds")
    save_figure(fig, "f4_calibration_size")
    return fig


# ---------------------------------------------------------------- F5


@_styled
def plot_train_vs_heldout(metrics=("log_loss", "ece"), log_x=("log_loss",)) -> Figure:
    """F5: test metric when the calibrators are fitted on the training set vs a held-out set.

    One row per dataset, one column per metric. Filled markers: fitted on the calibration
    set; hollow markers: fitted on the training set. Mean ± std over seeds.
    Reads results/train_vs_heldout.csv.
    """
    df = pd.read_csv(RESULTS_DIR / "train_vs_heldout.csv")
    n_seeds = df["seed"].nunique()
    stats = (df.groupby(["dataset", "model", "method", "fit_on"])[list(metrics)]
             .agg(["mean", "std"]))
    entries = [(model, method) for model in MODELS for method in METHODS]

    fig, axes = plt.subplots(len(DATASETS), len(metrics), figsize=(FULL_WIDTH, 3.8),
                             sharey=True, squeeze=False, layout="constrained")
    y_pos = np.arange(len(entries))[::-1]
    for r, dataset in enumerate(DATASETS):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            for y_i, (model, method) in zip(y_pos, entries):
                for fit_on, offset in (("cal", 0.15), ("train", -0.15)):
                    key = (dataset, model, method, fit_on)
                    if key not in stats.index:  # uncalibrated rows exist only for fit_on = "cal"
                        continue
                    mean, std = stats.loc[key, (metric, "mean")], stats.loc[key, (metric, "std")]
                    color = METHOD_COLOR[method]
                    ax.errorbar(mean, y_i + (offset if method != "uncalibrated" else 0),
                                xerr=std, color=color, marker=METHOD_MARKER[method],
                                markersize=5, linestyle="none", elinewidth=1, capsize=0,
                                markerfacecolor=color if fit_on == "cal" else "white")
            if metric in log_x:
                _log_axis(ax, "x")
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{'LR' if m == 'logreg' else 'RF'} {METHOD_LABEL[k]}"
                                for m, k in entries])
            ax.axhline(2.5, color=GRID, linewidth=0.8)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{DATASET_LABEL[dataset]}: {METRIC_LABEL[metric]}")

    handles = [
        Line2D([], [], color=INK_SECONDARY, marker="o", linestyle="none", markersize=5,
               label="Fitted on calibration set"),
        Line2D([], [], color=INK_SECONDARY, marker="o", linestyle="none", markersize=5,
               markerfacecolor="white", label="Fitted on training set"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2,
               title=f"Test set, mean ± std over {n_seeds} seeds")
    save_figure(fig, "f5_train_vs_heldout")
    return fig
