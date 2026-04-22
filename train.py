# -*- coding: utf-8 -*-
"""
train.py
--------
Full ML pipeline:
  1. Load & clean data          (data_cleaning.py)
  2. Train/test split
  3. Train 3 classifiers        (Decision Tree, Random Forest, Logistic Regression)
  4. Print comparison table
  5. Auto-select best by F1
  6. Save model artifacts
  7. Generate evaluation graphs (evaluate.py)
"""

import io
import sys
import os
import numpy as np
import joblib

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

import data_cleaning
import evaluate

os.makedirs("models", exist_ok=True)

# ── 1. Load & clean ───────────────────────────────────────────────────
X, y, label_encoders, categorical_cols = data_cleaning.load_and_clean()
feature_names = X.columns.tolist()

# ── 2. Train / test split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Scale ──────────────────────────────────────────────────────────
print("=" * 60)
print("  MODEL TRAINING PIPELINE")
print("=" * 60)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. Define models ──────────────────────────────────────────────────
models_config = [
    ("Decision Tree",     DecisionTreeClassifier(max_depth=8, random_state=42)),
    ("Random Forest",     RandomForestClassifier(n_estimators=200, max_depth=10,
                                                  random_state=42, n_jobs=-1)),
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42,
                                                solver="lbfgs")),
]

# ── 5. Train & evaluate ───────────────────────────────────────────────
print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("  " + "─" * 65)

results:  dict = {}
trained_models: list = []

for name, model in models_config:
    model.fit(X_train_sc, y_train)
    y_pred_m = model.predict(X_test_sc)

    acc  = accuracy_score(y_test,  y_pred_m)
    prec = precision_score(y_test, y_pred_m, zero_division=0)
    rec  = recall_score(y_test,    y_pred_m, zero_division=0)
    f1   = f1_score(y_test,        y_pred_m, zero_division=0)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    trained_models.append((name, model))
    print(f"  {name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

# ── 6. Best model ─────────────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]["f1"])
best_model = dict(trained_models)[best_name]
y_pred_best = best_model.predict(X_test_sc)

print("  " + "─" * 65)
print(f"  [BEST] Best Model : {best_name}")
print(f"       F1 Score  : {results[best_name]['f1']:.4f}")
print(f"       Accuracy  : {results[best_name]['accuracy']:.4f}\n")

# Classification report
print("  Detailed Classification Report (Best Model):")
print(classification_report(y_test, y_pred_best,
                             target_names=["Rejected (Bad)", "Approved (Good)"]))

# ── 7. Save artifacts ─────────────────────────────────────────────────
joblib.dump(best_model,    "models/loan_model.pkl")
joblib.dump(scaler,        "models/scaler.pkl")
joblib.dump(label_encoders,"models/label_encoders.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")
joblib.dump(categorical_cols, "models/categorical_cols.pkl")

print("  Saved model artifacts:")
for f in ["loan_model.pkl", "scaler.pkl", "label_encoders.pkl",
          "feature_names.pkl", "categorical_cols.pkl"]:
    print(f"      [OK] models/{f}")

# ── 8. Evaluation graphs ──────────────────────────────────────────────
evaluate.generate_all(
    best_model      = best_model,
    best_model_name = best_name,
    all_models      = trained_models,
    X_test          = X_test_sc,
    y_test          = y_test,
    y_pred          = y_pred_best,
    feature_names   = feature_names,
    results         = results,
    y_full          = y,
)

print("=" * 60)
print("  [DONE] Training complete!")
print("  Run:  python loan_predictor.py")
print("=" * 60)