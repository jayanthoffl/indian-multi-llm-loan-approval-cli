# -*- coding: utf-8 -*-
"""
main.py
-------
Entry point for the AI Loan Eligibility Predictor.

Options:
  1 — Loan Eligibility Predictor (terminal Q&A)
  2 — Model Evaluation Dashboard (graphs: confusion matrix, ROC, etc.)
"""

import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Force UTF-8 stdout on Windows ─────────────────────────────────────
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
if hasattr(sys.stdin, "buffer") and sys.stdin.encoding != "utf-8":
    sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding="utf-8",
                                  errors="replace", line_buffering=True)

# ── ANSI colours ──────────────────────────────────────────────────────
GREEN  = "\033[92m";  RED    = "\033[91m";  YELLOW = "\033[93m"
BLUE   = "\033[94m";  CYAN   = "\033[96m";  BOLD   = "\033[1m"
DIM    = "\033[2m";   RESET  = "\033[0m"

def c(text, *codes): return "".join(codes) + str(text) + RESET
def hr(char="-", n=58): print(char * n)


# ══════════════════════════════════════════════════════════════════════
#  OPTION 2 — Evaluation Dashboard
# ══════════════════════════════════════════════════════════════════════

GRAPH_DIR = os.path.join("reports", "evaluation")

GRAPHS = [
    ("confusion_matrix.png",  "Confusion Matrix"),
    ("roc_curve.png",         "ROC Curve"),
    ("pr_curve.png",          "Precision-Recall Curve"),
    ("feature_importance.png","Feature Importance"),
    ("model_comparison.png",  "Model Comparison"),
    ("class_distribution.png","Class Distribution"),
]


def show_evaluation_dashboard():
    """Load all saved evaluation PNGs and display as a 2×3 dashboard."""
    # Import matplotlib here with interactive backend (NOT Agg)
    import matplotlib
    matplotlib.use("TkAgg")          # interactive; falls back silently on non-Tk systems
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.gridspec as gridspec

    print()
    hr("=")
    print(c("  MODEL EVALUATION DASHBOARD", BOLD, BLUE))
    hr("=")

    # ── Check all graphs exist ─────────────────────────────────────────
    missing = []
    for fname, title in GRAPHS:
        path = os.path.join(GRAPH_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)

    if missing:
        print(c("\n  [!] Some graphs are missing. Run  python train.py  first.", RED, BOLD))
        for m in missing:
            print(c(f"      - {m}", RED))
        print()
        return

    print(c(f"\n  Loading 6 graphs from:  {os.path.abspath(GRAPH_DIR)}\n", DIM))

    # ── Build sub-menu ────────────────────────────────────────────────
    print(c("  What would you like to view?", YELLOW, BOLD))
    print()
    print(f"      {c('1', CYAN, BOLD)}.  View ALL graphs  (2 x 3 dashboard)")
    for i, (fname, title) in enumerate(GRAPHS, 2):
        print(f"      {c(str(i), CYAN, BOLD)}.  {title}")
    print(f"      {c('0', DIM)}.  Back to main menu")
    print()

    while True:
        try:
            raw = input(f"  {c('>', CYAN, BOLD)} Choose [0-{len(GRAPHS)+1}]: ").strip()
            choice = int(raw)
            if 0 <= choice <= len(GRAPHS) + 1:
                break
            print(c(f"    Enter a number between 0 and {len(GRAPHS)+1}.", RED))
        except ValueError:
            print(c("    Invalid input.", RED))

    if choice == 0:
        return

    # ── Helper: display one PNG ────────────────────────────────────────
    def _show_single(idx: int):
        fname, title = GRAPHS[idx]
        path = os.path.join(GRAPH_DIR, fname)
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0d1117")
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        fig.suptitle(title, color="#58a6ff", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

    # ── All graphs dashboard ───────────────────────────────────────────
    if choice == 1:
        fig = plt.figure(figsize=(20, 12), facecolor="#0d1117")
        fig.suptitle(
            "AI Loan Predictor — Model Evaluation Dashboard",
            color="#58a6ff", fontsize=16, fontweight="bold", y=0.98,
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.08, wspace=0.06)
        for i, (fname, title) in enumerate(GRAPHS):
            path = os.path.join(GRAPH_DIR, fname)
            ax = fig.add_subplot(gs[i // 3, i % 3])
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, color="#8b949e", fontsize=10, pad=4)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    else:
        # Single graph (choice 2–7 → index 0–5)
        _show_single(choice - 2)

    print(c("  Opening graph window... (close the window to return to menu)\n", DIM))
    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  MAIN MENU
# ══════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 58)
    print(c("       🏦  AI LOAN ELIGIBILITY PREDICTOR  🏦", BOLD, BLUE))
    print(c("          India-Specific  |  ML Credit Risk Model", DIM))
    print("=" * 58)
    print(c("  Powered by: Random Forest / Decision Tree / Logistic Reg.", DIM))
    print()


def main_menu():
    print_banner()

    # Check if model exists
    model_ready = os.path.exists(os.path.join("models", "loan_model.pkl"))
    graphs_ready = os.path.exists(os.path.join(GRAPH_DIR, "confusion_matrix.png"))

    status_model  = c("[READY]", GREEN, BOLD) if model_ready  else c("[NOT TRAINED]", RED)
    status_graphs = c("[READY]", GREEN, BOLD) if graphs_ready else c("[NOT TRAINED]", RED)

    print(c("  Status:", BOLD))
    print(f"    Prediction Model  : {status_model}")
    print(f"    Evaluation Graphs : {status_graphs}")
    print()

    if not model_ready:
        print(c("  [!] No trained model found. Run  python train.py  first.", YELLOW))
        print()

    hr()
    print(c("  What would you like to do?\n", YELLOW, BOLD))
    print(f"      {c('1', CYAN, BOLD)}.  Loan Eligibility Predictor")
    print(c("         Answer a few questions and get an instant prediction", DIM))
    print(c("         + India-specific bank recommendations if rejected\n", DIM))
    print(f"      {c('2', CYAN, BOLD)}.  Model Evaluation Dashboard")
    print(c("         View confusion matrix, ROC curve, feature importance,", DIM))
    print(c("         model comparison, precision-recall & class distribution\n", DIM))
    print(f"      {c('0', DIM)}.  Exit")
    print()

    while True:
        try:
            raw = input(f"  {c('>', CYAN, BOLD)} Enter choice [0/1/2]: ").strip()
            choice = int(raw)
            if choice in (0, 1, 2):
                return choice
            print(c("    Please enter 0, 1, or 2.", RED))
        except ValueError:
            print(c("    Invalid input — please enter 0, 1, or 2.", RED))


def run():
    while True:
        try:
            choice = main_menu()

            if choice == 0:
                print(c("\n  Goodbye!\n", DIM))
                sys.exit(0)

            elif choice == 1:
                # ── Import and run the loan predictor ──────────────────
                try:
                    import loan_predictor
                    print()
                    loan_predictor.run()
                except SystemExit:
                    pass  # loan_predictor.run() may call sys.exit on keyboard interrupt
                except Exception as e:
                    print(c(f"\n  [Error] {e}", RED))

            elif choice == 2:
                # ── Show evaluation dashboard ──────────────────────────
                try:
                    show_evaluation_dashboard()
                except ImportError:
                    print(c("\n  [!] matplotlib is required. Run: pip install matplotlib", RED))
                except Exception as e:
                    print(c(f"\n  [Error displaying graphs] {e}", RED))

            # ── Ask to continue ────────────────────────────────────────
            print()
            hr()
            again = input(
                f"  {c('>', CYAN, BOLD)} Return to main menu? [y/n]: "
            ).strip().lower()
            if again not in ("y", "yes", ""):
                print(c("\n  Goodbye!\n", DIM))
                sys.exit(0)

        except KeyboardInterrupt:
            print(c("\n\n  Goodbye!\n", DIM))
            sys.exit(0)


if __name__ == "__main__":
    run()
