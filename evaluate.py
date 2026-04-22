"""
evaluate.py
───────────
Generates and saves 6 evaluation graphs to reports/evaluation/.
Called automatically at the end of train.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for CLI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

REPORTS_DIR = os.path.join("reports", "evaluation")

# ── shared dark theme ──────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
DIM_TEXT  = "#8b949e"
BLUE      = "#58a6ff"
GREEN     = "#3fb950"
ORANGE    = "#f0883e"
RED       = "#f85149"
PURPLE    = "#d2a8ff"
PALETTE   = [BLUE, GREEN, ORANGE, PURPLE]


def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    BORDER,
        "text.color":        TEXT,
        "axes.labelcolor":   TEXT,
        "xtick.color":       DIM_TEXT,
        "ytick.color":       DIM_TEXT,
        "grid.color":        BORDER,
        "grid.alpha":        0.6,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })


def _save(fig, filename: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"      ✓  {path}")
    return path


# ── 1. Confusion Matrix ───────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, model_name: str = "Best Model"):
    _apply_dark_theme()
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        linewidths=0.5, linecolor=BORDER,
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    # label corners
    for (i, j), label in [((0, 0), "TN"), ((0, 1), "FP"), ((1, 0), "FN"), ((1, 1), "TP")]:
        ax.text(j + 0.5, i + 0.78, label, ha="center", color=ORANGE, fontsize=9, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", color=BLUE, pad=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return _save(fig, "confusion_matrix.png")


# ── 2. ROC Curves ─────────────────────────────────────────────────────
def save_roc_curves(models_data: list, X_test, y_test):
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(models_data, PALETTE):
        scores = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=DIM_TEXT, lw=1, label="Random")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", color=BLUE, pad=14)
    ax.legend(framealpha=0.2, loc="lower right")
    ax.grid(True)
    return _save(fig, "roc_curve.png")


# ── 3. Precision-Recall Curves ────────────────────────────────────────
def save_pr_curves(models_data: list, X_test, y_test):
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(models_data, PALETTE):
        scores = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(rec, prec, color=color, lw=2, label=f"{name}  (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", color=BLUE, pad=14)
    ax.legend(framealpha=0.2, loc="upper right")
    ax.grid(True)
    return _save(fig, "pr_curve.png")


# ── 4. Feature Importance ─────────────────────────────────────────────
def save_feature_importance(model, feature_names: list, model_name: str = "Model"):
    if not hasattr(model, "feature_importances_"):
        print("      ⚠  Model has no feature_importances_; skipping.")
        return
    _apply_dark_theme()
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]  # top-15
    idx_rev = idx[::-1]
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(idx_rev)))

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        [feature_names[i] for i in idx_rev],
        importances[idx_rev],
        color=colors,
    )
    for bar, val in zip(bars, importances[idx_rev]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=DIM_TEXT)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {model_name}", color=BLUE, pad=14)
    ax.grid(True, axis="x")
    return _save(fig, "feature_importance.png")


# ── 5. Model Comparison ───────────────────────────────────────────────
def save_model_comparison(results: dict):
    _apply_dark_theme()
    metrics  = ["accuracy", "precision", "recall", "f1"]
    names    = list(results.keys())
    x        = np.arange(len(names))
    w        = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, color) in enumerate(zip(metrics, PALETTE)):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(x + i * w, vals, w, label=metric.capitalize(), color=color, alpha=0.87)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + w / 2, bar.get_height() + 0.007,
                    f"{v:.2f}", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy / Precision / Recall / F1", color=BLUE, pad=14)
    ax.legend(framealpha=0.2)
    ax.grid(True, axis="y")
    return _save(fig, "model_comparison.png")


# ── 6. Class Distribution ─────────────────────────────────────────────
def save_class_distribution(y):
    _apply_dark_theme()
    labels = ["Approved (Good)", "Rejected (Bad)"]
    counts = [int((y == 1).sum()), int((y == 0).sum())]
    colors = [GREEN, RED]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # pie
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": TEXT},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(13); at.set_fontweight("bold")
    ax1.set_title("Class Distribution (Pie)", color=BLUE, pad=14)
    ax1.set_facecolor(PANEL_BG)

    # bar
    bars = ax2.bar(labels, counts, color=colors, alpha=0.85, width=0.35,
                   edgecolor=DARK_BG, linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                 str(cnt), ha="center", fontsize=12, fontweight="bold", color=TEXT)
    ax2.set_ylabel("Number of Records")
    ax2.set_title("Class Distribution (Bar)", color=BLUE, pad=14)
    ax2.grid(True, axis="y")

    return _save(fig, "class_distribution.png")


# ── Master call ───────────────────────────────────────────────────────
def generate_all(best_model, best_model_name: str, all_models: list,
                 X_test, y_test, y_pred, feature_names: list,
                 results: dict, y_full):
    print("\n" + "=" * 60)
    print("  GENERATING EVALUATION GRAPHS")
    print("=" * 60)
    save_confusion_matrix(y_test, y_pred, best_model_name)
    save_roc_curves(all_models, X_test, y_test)
    save_pr_curves(all_models, X_test, y_test)
    save_feature_importance(best_model, feature_names, best_model_name)
    save_model_comparison(results)
    save_class_distribution(y_full)
    print(f"\n  ✅  All graphs → {os.path.abspath(REPORTS_DIR)}")
    print("=" * 60 + "\n")
