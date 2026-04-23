# 🏦 AI Loan Eligibility Predictor — India-Specific CLI

> A terminal-based Machine Learning system that predicts loan eligibility using the German Credit dataset, compares multiple classifiers, generates rich evaluation graphs, and — on rejection — recommends real Indian banks & government schemes tailored to the applicant's financial profile.

---

## 📸 Demo

**Approved path:**
```
══════════════════════════════════════════════════════════
       🏦  AI LOAN ELIGIBILITY PREDICTOR  🏦
          India-Specific  |  ML Credit Risk Model
══════════════════════════════════════════════════════════

  ✔  LOAN APPLICATION APPROVED ✔
  --------------------------------------------------
  Requested Amount : ₹60,000
  Duration         : 12 months
  Model Confidence : 73.4%

  Suggested next steps:
      ▸ Apply at HDFC / ICICI / Axis Bank for the best personal loan rates.
      ▸ Check pre-approved offers in your bank's mobile app.
```

**Rejected path:**
```
  ✖  LOAN APPLICATION REJECTED
  ══════════════════════════════════════════════════════════
  ✦ Fintech / NBFC Lenders (Alternative Scoring)
  ·······················································
  ● Bajaj Finserv  —  Personal Loan
    Range : ₹1,00,000 – ₹35,00,000
    Why   : Instant approval, minimal paperwork; 0-cost EMI options.

  ✦ Government Schemes
  ·······················································
  ● PMMY / Mudra Loan  —  Shishu / Kishor / Tarun
    Range : ₹50,000 – ₹10,00,000
    Why   : Zero collateral; for micro-entrepreneurs & self-employed.
```

---

## 📋 Table of Contents

- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Folder Structure](#-folder-structure)
- [ML Models Used](#-ml-models-used)
- [Dataset](#-dataset)
- [Evaluation Graphs](#-evaluation-graphs)
- [India Bank Recommendation Engine](#-india-bank-recommendation-engine)
- [Getting Started](#-getting-started)
- [How to Use](#-how-to-use)
- [Input Fields Explained](#-input-fields-explained)
- [File Reference](#-file-reference)

---

## 🛠 Tech Stack

| Category | Tool / Library |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn 1.3+ |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Model Serialization | joblib |
| Dataset Source | OpenML (`credit-g`, German Credit) |
| Terminal UI | ANSI escape codes (native, no extra deps) |

---

## 🏗 Project Architecture

```
                    ┌─────────────────────────────────┐
                    │         train.py                │
                    │   (Master Pipeline Orchestrator) │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
   ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐
   │ data_cleaning.py│  │   3 ML Models    │  │  evaluate.py   │
   │                 │  │                  │  │                │
   │ • Fetch dataset │  │ • Decision Tree  │  │ • Confusion Mx │
   │ • Quality audit │  │ • Random Forest  │  │ • ROC Curve    │
   │ • Encode cats   │  │ • Logistic Reg.  │  │ • PR Curve     │
   │ • Return X, y   │  │                  │  │ • Feat Import  │
   └────────┬────────┘  │ Auto-select best │  │ • Model Comp   │
            │           │ model by F1 score│  │ • Class Distrib│
            │           └────────┬─────────┘  └───────┬────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │       models/           │
                    │  loan_model.pkl         │
                    │  scaler.pkl             │
                    │  label_encoders.pkl     │
                    │  feature_names.pkl      │
                    │  categorical_cols.pkl   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    loan_predictor.py    │
                    │   (Terminal CLI App)    │
                    │                        │
                    │  • Collect 8 inputs    │
                    │  • Scale & predict     │
                    │  • APPROVED → tips     │
                    │  • REJECTED → banks    │
                    └────────────────────────┘
```

---

## 📁 Folder Structure

```
loan-predictor-cli/
│
├── data_cleaning.py          # Step 1: Data loading, quality audit & encoding
├── train.py                  # Step 2: Full ML pipeline (run this first)
├── evaluate.py               # Step 3: Graph generation (called by train.py)
├── loan_predictor.py         # Step 4: Terminal CLI predictor (run to predict)
├── requirements.txt          # Pinned Python dependencies
│
├── models/                   # Auto-created on first train.py run
│   ├── loan_model.pkl        # Trained best-performing classifier
│   ├── scaler.pkl            # Fitted StandardScaler
│   ├── label_encoders.pkl    # LabelEncoders for categorical features
│   ├── feature_names.pkl     # Ordered list of feature names
│   └── categorical_cols.pkl  # List of categorical column names
│
├── reports/
│   └── evaluation/           # Auto-created on first train.py run
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── pr_curve.png
│       ├── feature_importance.png
│       ├── model_comparison.png
│       └── class_distribution.png
│
└── venv/                     # Virtual environment (not committed to git)
```

---

## 🤖 ML Models Used

Three classifiers are trained and compared on every `train.py` run. The best model (by **F1 score**) is auto-selected and saved.

### 1. Decision Tree Classifier
- **Why**: Highly interpretable; easy to understand decision rules
- **Config**: `max_depth=8`, `random_state=42`
- **Pros**: Fast, explainable, no scaling needed (scaled anyway for consistency)
- **Cons**: Prone to overfitting on small datasets

### 2. Random Forest Classifier ⭐ (usually wins)
- **Why**: Ensemble of 200 trees — reduces variance significantly
- **Config**: `n_estimators=200`, `max_depth=10`, `n_jobs=-1`
- **Pros**: Best accuracy/F1, built-in feature importance, robust to noise
- **Cons**: Less interpretable than a single tree

### 3. Logistic Regression
- **Why**: Probabilistic output; good calibration of confidence scores
- **Config**: `max_iter=1000`, `solver='lbfgs'`
- **Pros**: Fast, interpretable coefficients, well-calibrated probabilities
- **Cons**: Assumes linear decision boundary

### Results on German Credit Dataset (80/20 split)

| Model | Accuracy | Precision | Recall | **F1** |
|---|---|---|---|---|
| Decision Tree | 72.0% | 0.804 | 0.793 | 0.799 |
| **Random Forest** | **77.0%** | **0.787** | **0.921** | **0.849** |
| Logistic Regression | 70.0% | 0.760 | 0.836 | 0.796 |

> Random Forest consistently wins on F1 score and is auto-selected as the prediction model.

---

## 📊 Dataset

**German Credit Dataset** (OpenML: `credit-g`, version 1)

| Property | Value |
|---|---|
| Source | [OpenML credit-g](https://www.openml.org/d/31) |
| Records | 1,000 loan applicants |
| Features | 20 (13 categorical + 7 numerical) |
| Target | `good` (700 records, 70%) / `bad` (300 records, 30%) |
| Original Currency | Deutsche Marks (DM) |

**Key Features Used in Prediction:**

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Applicant age in years |
| `credit_amount` | Numeric | Loan amount (DM) — converted from ₹ input |
| `duration` | Numeric | Loan term in months |
| `checking_status` | Categorical | Checking account balance status |
| `credit_history` | Categorical | Past credit repayment behaviour |
| `savings_status` | Categorical | Savings account / investment size |
| `employment` | Categorical | Employment duration |
| `housing` | Categorical | Housing situation (own/rent/free) |
| `purpose` | Categorical | Reason for taking the loan |

> The model is trained on all 20 features. The CLI collects the 8 most impactful ones and uses sensible defaults for the rest.

---

## 📈 Evaluation Graphs

All graphs are auto-generated into `reports/evaluation/` after every training run.

| Graph | File | What it Shows |
|---|---|---|
| Confusion Matrix | `confusion_matrix.png` | TP / FP / TN / FN breakdown |
| ROC Curve | `roc_curve.png` | AUC-ROC for all 3 models |
| Precision-Recall | `pr_curve.png` | Precision vs Recall tradeoff |
| Feature Importance | `feature_importance.png` | Top 15 features (Random Forest) |
| Model Comparison | `model_comparison.png` | Grouped bar: Acc/Prec/Recall/F1 |
| Class Distribution | `class_distribution.png` | Good vs Bad credit (pie + bar) |

All graphs use a **dark GitHub-style theme** with a polished colour palette.

---

## 🏦 India Bank Recommendation Engine

When a loan is **rejected**, the system runs a rule-based engine that suggests real Indian lenders based on:
- Requested loan amount (₹)
- Loan duration (months)
- Credit history score
- Employment status

### Recommendation Matrix

| Loan Amount | Category | Banks Suggested |
|---|---|---|
| ≤ ₹2,00,000 | Micro / SFB | Bandhan Bank, Ujjivan SFB, Jana SFB |
| ≤ ₹2,00,000 | Govt Schemes | PMMY / Mudra Loan, SIDBI Stand-Up India |
| ₹2L – ₹5L | Fintech / NBFC | Bajaj Finserv, KreditBee, MoneyTap, CASHe, PaySense |
| ₹2L – ₹5L | Secured (bad credit) | Muthoot Finance, Manappuram, IIFL Finance |
| ₹5L – ₹25L | Private Banks | HDFC, ICICI, Axis, Kotak Mahindra |
| ₹5L – ₹25L | Fintech Backup | Bajaj Finserv, KreditBee, MoneyTap |
| > ₹25L | PSU Banks | SBI, PNB, Bank of Baroda |
| > ₹25L (long term) | Housing Finance | LIC HFL, PNB Housing, Indiabulls HF |
| Any (bad credit) | Gold / Asset-Backed | Muthoot Finance, Manappuram, IIFL Finance |

The engine also prints **5 actionable tips** to improve CIBIL score and reapplication chances.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip
- Git (optional, for cloning)
- Windows 10+ / macOS / Linux terminal

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RAG-Loan-predictor.git
cd RAG-Loan-predictor/loan-predictor-cli
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `scikit-learn` — ML models & preprocessing
- `pandas` — data manipulation
- `numpy` — numerical computing
- `matplotlib` + `seaborn` — evaluation graphs
- `joblib` — model serialization
- `scipy` — scikit-learn dependency

### 4. Train the Model

```bash
python train.py
```

This will:
1. Download the German Credit dataset automatically (via OpenML — ~2MB, one-time)
2. Run the full data cleaning pipeline with a quality audit
3. Train 3 ML models and print a comparison table
4. Auto-select the best model by F1 score
5. Save all model artifacts to `models/`
6. Generate 6 evaluation graphs to `reports/evaluation/`

Expected output:
```
============================================================
  DATA LOADING & CLEANING PIPELINE
============================================================
[1/5] Fetching German Credit dataset from OpenML...
      [OK] Loaded 1000 records, 21 columns
...
  Model                       Accuracy  Precision     Recall         F1
  -----------------------------------------------------------------
  Decision Tree                 0.7200     0.8043     0.7929     0.7986
  Random Forest                 0.7700     0.7866     0.9214     0.8487
  Logistic Regression           0.7000     0.7597     0.8357     0.7959

  [BEST] Best Model : Random Forest
         F1 Score   : 0.8487
...
  [DONE] Training complete!
```

### 5. Launch the App

```bash
python main.py
```

You'll see a main menu with two options:

```
==========================================================
       🏦  AI LOAN ELIGIBILITY PREDICTOR  🏦
          India-Specific  |  ML Credit Risk Model
==========================================================

  Status:
    Prediction Model  : [READY]
    Evaluation Graphs : [READY]

  What would you like to do?

      1.  Loan Eligibility Predictor
           Answer a few questions and get an instant prediction
           + India-specific bank recommendations if rejected

      2.  Model Evaluation Dashboard
           View confusion matrix, ROC curve, feature importance,
           model comparison, precision-recall & class distribution

      0.  Exit
```

---

## 🖥 How to Use

Once you run `loan_predictor.py`, the CLI walks you through **8 questions**:

```
▶ Your age (years): 25
▶ Loan amount requested (₹ — e.g. 500000 for 5 lakh): 300000
▶ Loan duration (months — e.g. 24, 36, 60): 36

  Checking account status:
      1.  no checking account
      2.  < 0 DM (overdrawn)
      3.  0 – 200 DM balance
      4.  > 200 DM balance
▶ Enter number [1–4]: 3

  ... (4 more questions)
```

After answering, you instantly get either:
- ✔ **APPROVED** — with confidence score and next steps
- ✖ **REJECTED** — with Indian bank recommendations tailored to your profile

---

## 📝 Input Fields Explained

| # | Question | What to Enter |
|---|---|---|
| 1 | Age | Your age in whole years (e.g. `25`) |
| 2 | Loan Amount (₹) | Total amount needed (e.g. `500000` for ₹5 lakh) |
| 3 | Duration (months) | Repayment period (e.g. `24`, `36`, `60`) |
| 4 | Checking Account | Select the closest match to your bank balance |
| 5 | Credit History | Be honest — this is the most impactful feature |
| 6 | Savings | Approximate total savings/investments |
| 7 | Employment | How long you've been continuously employed |
| 8 | Housing | Whether you own, rent, or live freely |
| 9 | Purpose | Primary reason for the loan |

> **Note on Currency**: The model was trained on German Deutsche Marks (DM). Your ₹ input is automatically converted using a 1 DM ≈ ₹30 factor before being passed to the model. The StandardScaler then normalizes the values relative to the training distribution.

---

## 📂 File Reference

| File | Role | Run Directly? |
|---|---|---|
| `data_cleaning.py` | Loads dataset, runs quality audit, encodes features | `python data_cleaning.py` (for audit only) |
| `train.py` | Full ML pipeline — trains, evaluates, saves model | ✅ `python train.py` |
| `evaluate.py` | Graph generation module | Called by `train.py` automatically |
| `loan_predictor.py` | Terminal CLI predictor | ✅ `python loan_predictor.py` |
| `requirements.txt` | Python dependencies | `pip install -r requirements.txt` |

---

## 🔁 Re-Training

You can re-train at any time. The pipeline is fully reproducible:

```bash
python train.py
```

All model artifacts in `models/` and graphs in `reports/evaluation/` will be overwritten.

---

## ⚠️ Limitations & Notes

- **Dataset origin**: The German Credit dataset reflects 1990s European lending patterns. It is used here as a proxy for demonstrating ML concepts — not as a real Indian credit scoring system.
- **Currency conversion**: The ₹ → DM conversion (÷30) is approximate. Results should be treated as indicative, not definitive.
- **Bank suggestions**: The recommendation engine is rule-based, not live. Always verify current eligibility criteria directly with the lender.
- **CIBIL Score**: This model does not use CIBIL / credit score as a direct input. Credit history (question 5) is the closest proxy.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- [UCI / OpenML German Credit Dataset](https://www.openml.org/d/31) — Hans Hofmann, University of Hamburg
- [scikit-learn](https://scikit-learn.org/) — ML framework
- [Bajaj Finserv](https://www.bajajfinserv.in/), [KreditBee](https://www.kreditbee.in/), [Muthoot Finance](https://www.muthootfinance.com/) and other mentioned institutions for publicly available product information

---

*Built with Python · scikit-learn · pandas · matplotlib · seaborn*
