# -*- coding: utf-8 -*-
"""
data_cleaning.py
----------------
Loads the German Credit dataset, runs a full data-quality audit,
encodes categorical features, and returns clean arrays ready for ML.
"""

import io
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)


def load_and_clean(verbose: bool = True) -> tuple:
    """
    Load, audit, and clean the German Credit dataset.

    Returns
    -------
    X              : pd.DataFrame  – encoded feature matrix (all 20 features)
    y              : np.ndarray    – binary target (1=good/approved, 0=bad/rejected)
    label_encoders : dict          – {col: LabelEncoder} for categorical columns
    categorical_cols : list        – names of categorical columns
    """

    def _print(msg):
        if verbose:
            print(msg)

    _print("\n" + "=" * 60)
    _print("  DATA LOADING & CLEANING PIPELINE")
    _print("=" * 60)

    # ── 1. Fetch ──────────────────────────────────────────────────────
    _print("\n[1/5] Fetching German Credit dataset from OpenML...")
    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()
    _print(f"      [OK] Loaded {df.shape[0]} records, {df.shape[1]} columns")

    # ── 2. Quality report ─────────────────────────────────────────────
    _print("\n[2/5] Data Quality Report:")
    _print(f"      Shape        : {df.shape}")
    missing = df.isnull().sum().sum()
    _print(f"      Missing vals : {missing}  {'[OK] none' if missing == 0 else '[WARN] needs attention'}")
    dupes = df.duplicated().sum()
    _print(f"      Duplicates   : {dupes}  {'[OK] none' if dupes == 0 else '[WARN] will be dropped'}")

    if dupes > 0:
        df.drop_duplicates(inplace=True)
        _print(f"      → Dropped {dupes} duplicate rows")

    if missing > 0:
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
        _print("      → Imputed: numeric→median, categorical→mode")

    _print("\n      Column summary:")
    _print(f"      {'Column':<35} {'DType':<15} {'Unique'}")
    _print("      " + "-" * 58)
    for col in df.columns:
        _print(f"      {col:<35} {str(df[col].dtype):<15} {df[col].nunique()}")

    # ── 3. Target ─────────────────────────────────────────────────────
    _print("\n[3/5] Encoding target column...")
    y = np.where(df["class"] == "good", 1, 0)
    X = df.drop(columns=["class"]).copy()

    # ── 4. Encode categoricals ────────────────────────────────────────
    _print("\n[4/5] Label-encoding categorical features...")
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    label_encoders: dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    _print(f"      [OK] Encoded {len(categorical_cols)} categorical columns")

    # ── 5. Class distribution ─────────────────────────────────────────
    _print("\n[5/5] Class distribution:")
    good = int((y == 1).sum())
    bad = int((y == 0).sum())
    _print(f"      [OK] Approved (Good) : {good:>4}  ({good / len(y) * 100:.1f}%)")
    _print(f"      [OK] Rejected (Bad)  : {bad:>4}  ({bad  / len(y) * 100:.1f}%)")
    _print(f"      [OK] {X.shape[1]} features ready for modeling")
    _print("=" * 60 + "\n")

    return X, y, label_encoders, categorical_cols


if __name__ == "__main__":
    X, y, le, cat_cols = load_and_clean()
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"Categorical columns: {cat_cols}")
