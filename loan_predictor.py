# -*- coding: utf-8 -*-
"""
loan_predictor.py
-----------------
Terminal-based Loan Eligibility Predictor (India-specific)
  - Loads trained model from models/
  - Collects 8 key inputs from user
  - Predicts APPROVED / REJECTED
  - On rejection: India-specific bank recommendation engine
"""

import io
import joblib
import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ── Force UTF-8 stdout so special chars print correctly on Windows ─────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
if sys.stdin.encoding != "utf-8":
    sys.stdin  = io.TextIOWrapper(sys.stdin.buffer,  encoding="utf-8",
                                  errors="replace", line_buffering=True)

# ── ANSI colour helpers (work on Windows 10+ terminals) ───────────────
GREEN  = "\033[92m";  RED    = "\033[91m";  YELLOW = "\033[93m"
BLUE   = "\033[94m";  CYAN   = "\033[96m";  BOLD   = "\033[1m"
DIM    = "\033[2m";   RESET  = "\033[0m"

def c(text, *codes): return "".join(codes) + str(text) + RESET
def hr(char="-", n=58): print(char * n)


# ══════════════════════════════════════════════════════════════════════
#  INDIA-SPECIFIC BANK RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════

BANKS = {
    # ── Micro / very small loans ──────────────────────────────────────
    "micro": [
        {"name": "Bandhan Bank",
         "product": "Micro Loan",
         "range": "₹10,000 – ₹2,00,000",
         "why": "Specialises in underbanked segments; flexible eligibility criteria."},
        {"name": "Ujjivan Small Finance Bank",
         "product": "Personal Loan",
         "range": "₹50,000 – ₹5,00,000",
         "why": "Low income / self-employed friendly; quick disbursal."},
        {"name": "Jana Small Finance Bank",
         "product": "Micro Finance Loan",
         "range": "₹10,000 – ₹1,50,000",
         "why": "Group-lending model; great for first-time borrowers."},
    ],

    # ── Fintech / NBFC personal loans ─────────────────────────────────
    "fintech": [
        {"name": "Bajaj Finserv",
         "product": "Personal Loan",
         "range": "₹1,00,000 – ₹35,00,000",
         "why": "Instant approval, minimal paperwork; 0-cost EMI options."},
        {"name": "KreditBee",
         "product": "Instant Personal Loan",
         "range": "₹1,000 – ₹4,00,000",
         "why": "Alternative AI-based credit scoring; good for thin-file borrowers."},
        {"name": "MoneyTap",
         "product": "Revolving Credit Line",
         "range": "₹3,000 – ₹5,00,000",
         "why": "Pay interest only on what you draw; flexible repayment."},
        {"name": "CASHe",
         "product": "Short-term Salary Loan",
         "range": "₹1,000 – ₹4,00,000",
         "why": "Uses social score & income patterns; great for salaried millennials."},
        {"name": "PaySense",
         "product": "Personal Loan",
         "range": "₹5,000 – ₹5,00,000",
         "why": "100% online process; approves lower CIBIL profiles."},
    ],

    # ── Mainstream bank personal loans ────────────────────────────────
    "bank_personal": [
        {"name": "HDFC Bank",
         "product": "Personal Loan",
         "range": "₹50,000 – ₹40,00,000",
         "why": "Fastest approval for existing customers; pre-approved offers."},
        {"name": "ICICI Bank",
         "product": "Personal Loan",
         "range": "₹50,000 – ₹50,00,000",
         "why": "iMobile pre-approved offers; co-applicant allowed."},
        {"name": "Axis Bank",
         "product": "Personal Loan",
         "range": "₹50,000 – ₹40,00,000",
         "why": "Competitive rates for salaried professionals."},
        {"name": "Kotak Mahindra Bank",
         "product": "Personal Loan",
         "range": "₹50,000 – ₹25,00,000",
         "why": "Quick online application; accepts lower income brackets."},
    ],

    # ── PSU / large loans ─────────────────────────────────────────────
    "psu_large": [
        {"name": "State Bank of India (SBI)",
         "product": "SBI Xpress Credit",
         "range": "₹25,000 – ₹35,00,000",
         "why": "Lowest interest rates for govt / PSU employees."},
        {"name": "Punjab National Bank (PNB)",
         "product": "Personal Loan",
         "range": "₹25,000 – ₹10,00,000",
         "why": "Flexible for rural applicants; relaxed norms for farmers."},
        {"name": "Bank of Baroda",
         "product": "Baroda Personal Loan",
         "range": "₹1,00,000 – ₹10,00,000",
         "why": "Good for existing account holders; lower processing fee."},
    ],

    # ── Secured / gold loans (bad credit) ────────────────────────────
    "secured": [
        {"name": "Muthoot Finance",
         "product": "Gold Loan",
         "range": "₹1,500 – ₹1,50,00,000",
         "why": "No CIBIL check needed; instant approval against gold."},
        {"name": "Manappuram Finance",
         "product": "Gold Loan",
         "range": "₹5,000 – ₹1,00,00,000",
         "why": "Same-day disbursement; accept low-quality gold ornaments."},
        {"name": "IIFL Finance",
         "product": "Gold / Secured Loan",
         "range": "₹3,000 – ₹50,00,000",
         "why": "Accepts multiple collateral types; doorstep service available."},
    ],

    # ── Govt schemes ─────────────────────────────────────────────────
    "govt": [
        {"name": "PMMY / Mudra Loan (via any bank)",
         "product": "Shishu / Kishor / Tarun",
         "range": "₹50,000 – ₹10,00,000",
         "why": "Zero collateral; for micro-entrepreneurs & self-employed."},
        {"name": "SIDBI Stand-Up India",
         "product": "Business Loan",
         "range": "₹10,00,000 – ₹1,00,00,000",
         "why": "SC/ST/Women entrepreneurs; subsidised rates."},
    ],

    # ── Long-tenure home / housing loans ─────────────────────────────
    "housing": [
        {"name": "LIC Housing Finance",
         "product": "Home Loan",
         "range": "₹5,00,000 – ₹15,00,00,000",
         "why": "Long tenures up to 30 years; accepts slightly lower CIBIL."},
        {"name": "PNB Housing Finance",
         "product": "Home Loan",
         "range": "₹8,00,000 – ₹5,00,00,000",
         "why": "Flexible repayment; step-up EMI for younger borrowers."},
        {"name": "Indiabulls Housing Finance",
         "product": "Home Loan",
         "range": "₹5,00,000 – ₹10,00,00,000",
         "why": "Less stringent income proof; good for self-employed."},
    ],
}


def _bank_block(banks: list, header: str):
    """Print a formatted block of bank recommendations."""
    print(f"\n  {c(header, CYAN, BOLD)}")
    hr("·", 54)
    for b in banks:
        print(f"  {c('●', YELLOW, BOLD)} {c(b['name'], BOLD)}  —  {b['product']}")
        print(f"    {c('Range :', DIM)} {b['range']}")
        print(f"    {c('Why   :', DIM)} {b['why']}")
        print()


def suggest_banks(loan_amount_inr: float, duration_months: float,
                  credit_history_score: int, employment_score: int):
    """
    Rule-based India-specific bank recommendation engine.

    Parameters
    ----------
    loan_amount_inr     : requested loan in ₹
    duration_months     : loan duration
    credit_history_score: 0=critical/bad  1=delayed  2=existing paid  3=all paid
    employment_score    : 0=unemployed  1=<1yr  2=1-4yr  3=4-7yr  4=≥7yr
    """
    print("\n" + "═" * 58)
    print(c("  ✖  LOAN APPLICATION REJECTED", RED, BOLD))
    print("═" * 58)
    print(c("  Your current profile does not meet standard bank\n", DIM))
    print(c("  lending criteria based on our credit risk assessment.\n", DIM))
    print(c("  However, several lenders in India may still be able", YELLOW))
    print(c("  to support you. Here are our recommendations:\n", YELLOW))

    # ── Micro loans ───────────────────────────────────────────────────
    if loan_amount_inr <= 200_000:
        _bank_block(BANKS["micro"], "✦ Micro-Finance & Small Finance Banks")
        _bank_block(BANKS["govt"],  "✦ Government Schemes (No Collateral Required)")

    # ── Fintech range ─────────────────────────────────────────────────
    elif loan_amount_inr <= 500_000:
        _bank_block(BANKS["fintech"], "✦ Fintech / NBFC Lenders (Alternative Scoring)")
        if credit_history_score <= 1:
            _bank_block(BANKS["secured"], "✦ Secured Loans (Gold / Asset-Backed — No CIBIL)")
        _bank_block(BANKS["govt"], "✦ Government Schemes")

    # ── Medium loans ──────────────────────────────────────────────────
    elif loan_amount_inr <= 2_500_000:
        _bank_block(BANKS["bank_personal"], "✦ Mainstream Bank Personal Loans")
        _bank_block(BANKS["fintech"][:3],   "✦ NBFC / Fintech Alternatives")
        if credit_history_score <= 1:
            _bank_block(BANKS["secured"], "✦ Secured Loans (If You Have Gold / Assets)")

    # ── Large / home loans ────────────────────────────────────────────
    else:
        if duration_months >= 60:
            _bank_block(BANKS["housing"],    "✦ Housing Finance Companies (Long-Term)")
        _bank_block(BANKS["psu_large"],      "✦ PSU Banks (Subsidised Rates)")
        _bank_block(BANKS["bank_personal"][:2], "✦ Private Banks")
        if credit_history_score <= 1:
            _bank_block(BANKS["secured"],    "✦ Secured Option (Gold / Property Loan)")

    # ── Universal tips ────────────────────────────────────────────────
    print("─" * 58)
    print(c("  💡  Tips to improve eligibility:", YELLOW, BOLD))
    tips = [
        "Clear existing overdue EMIs before reapplying.",
        "Apply with a creditworthy co-applicant / guarantor.",
        "Reduce the requested loan amount or extend tenure.",
        "Build CIBIL score ≥ 700 (check free on Paisa Bazaar).",
        "Show consistent income for 6+ months (bank statements).",
    ]
    for tip in tips:
        print(f"      {c('▸', CYAN)} {tip}")
    print()


# ══════════════════════════════════════════════════════════════════════
#  CLI INPUT HELPERS
# ══════════════════════════════════════════════════════════════════════

def _ask(prompt: str, cast=float, valid=None, default=None):
    """Prompt user, validate, and cast input."""
    while True:
        try:
            raw = input(f"  {c('▶', CYAN)} {prompt} ").strip()
            if raw == "" and default is not None:
                return default
            val = cast(raw)
            if valid and val not in valid:
                print(c(f"    ✗ Choose from: {valid}", RED))
                continue
            return val
        except (ValueError, TypeError):
            print(c("    ✗ Invalid input — please try again.", RED))


def _ask_choice(prompt: str, options: list):
    """
    Present a numbered menu and return the user's chosen value.
    """
    print(f"\n  {c(prompt, BOLD)}")
    for i, opt in enumerate(options, 1):
        print(f"      {c(str(i), CYAN, BOLD)}.  {opt}")
    idx = _ask(f"Enter number [1–{len(options)}]: ", cast=int,
               valid=list(range(1, len(options) + 1)))
    return options[idx - 1], idx - 1   # (label, 0-based index)


# ══════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════

def run():
    # ── Banner ─────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print(c("       🏦  AI LOAN ELIGIBILITY PREDICTOR  🏦", BOLD, BLUE))
    print(c("          India-Specific  |  ML Credit Risk Model", DIM))
    print("═" * 58)
    print(c("  Powered by: Random Forest / Decision Tree / Logistic Reg.", DIM))
    print()

    # ── Load model artefacts ───────────────────────────────────────────
    try:
        model          = joblib.load("models/loan_model.pkl")
        scaler         = joblib.load("models/scaler.pkl")
        label_encoders = joblib.load("models/label_encoders.pkl")
        feature_names  = joblib.load("models/feature_names.pkl")
        categorical_cols = joblib.load("models/categorical_cols.pkl")
    except FileNotFoundError:
        print(c("  ✗  Model not found. Run  python train.py  first.", RED, BOLD))
        sys.exit(1)

    print(c("  ✓  Model loaded successfully.\n", GREEN))
    hr()
    print(c("  Please answer the following questions:\n", YELLOW))

    try:
        # ── Numeric inputs ─────────────────────────────────────────────
        age = _ask("Your age (years):", float)
        loan_inr = _ask("Loan amount requested (₹ — e.g. 500000 for 5 lakh):", float)
        duration = _ask("Loan duration (months — e.g. 24, 36, 60):", float)

        # Internally scale ₹ to the unit the model was trained on (1 unit ≈ ₹30)
        SCALE_FACTOR = 30.0
        credit_amount_dm = loan_inr / SCALE_FACTOR

        # ── Categorical inputs ─────────────────────────────────────────

        # checking_status
        chk_options = [
            "No checking / savings account linked",
            "Account overdrawn / negative balance",
            "Balance ₹0 – ₹6,000",
            "Balance > ₹6,000",
        ]
        chk_map = ["no checking", "<0", "0<=X<200", ">=200"]
        _, chk_idx = _ask_choice("Checking / bank account status:", chk_options)
        checking_status = chk_map[chk_idx]

        # credit_history
        hist_options = [
            "No credits taken / all paid on time",
            "All existing credits paid on time",
            "Existing credits paid back duly (some delays)",
            "Delayed payments in the past",
            "Critical / other existing credits at other banks",
        ]
        hist_map = ["no credits/all paid", "all paid", "existing paid",
                    "delayed previously", "critical/other existing credit"]
        _, hist_idx = _ask_choice("Credit history:", hist_options)
        credit_history = hist_map[hist_idx]
        credit_history_score = hist_idx   # 0=worst .. 4=best (reversed scale)
        credit_history_score = 4 - hist_idx  # normalise: 0=bad, 4=good

        # savings_status
        sav_options = [
            "No savings / investment account",
            "Savings < ₹3,000",
            "Savings ₹3,000 – ₹15,000",
            "Savings ₹15,000 – ₹30,000",
            "Savings > ₹30,000",
        ]
        sav_map = ["no known savings", "<100", "100<=X<500", "500<=X<1000", ">=1000"]
        _, sav_idx = _ask_choice("Savings / investments:", sav_options)
        savings_status = sav_map[sav_idx]

        # employment
        emp_options = ["Unemployed",
                       "Employed < 1 year",
                       "Employed 1 – 4 years",
                       "Employed 4 – 7 years",
                       "Employed ≥ 7 years"]
        emp_map = ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
        _, emp_idx = _ask_choice("Employment duration:", emp_options)
        employment = emp_map[emp_idx]

        # housing
        hsg_options = ["Own home", "Free (living with family / provided)", "Rented"]
        hsg_map = ["own", "free", "rent"]
        _, hsg_idx = _ask_choice("Housing situation:", hsg_options)
        housing = hsg_map[hsg_idx]

        # purpose
        pur_options = ["New vehicle", "Used vehicle", "Furniture / equipment",
                       "Electronics / TV / Radio", "Domestic appliances",
                       "Home repairs", "Education / training",
                       "Business / self-employment", "Other"]
        pur_map = ["new car", "used car", "furniture/equipment", "radio/tv",
                   "domestic appliances", "repairs", "education", "business", "other"]
        _, pur_idx = _ask_choice("Purpose of loan:", pur_options)
        purpose = pur_map[pur_idx]

        # ── Build feature vector ───────────────────────────────────────
        raw = {
            "checking_status":       checking_status,
            "duration":              duration,
            "credit_history":        credit_history,
            "purpose":               purpose,
            "credit_amount":         credit_amount_dm,
            "savings_status":        savings_status,
            "employment":            employment,
            "installment_commitment": 3,   # median default
            "personal_status":       "male single",  # neutral default
            "other_parties":         "none",
            "residence_since":       2,
            "property_magnitude":    "car",
            "age":                   age,
            "other_payment_plans":   "none",
            "housing":               housing,
            "existing_credits":      1,
            "job":                   "skilled",
            "num_dependents":        1,
            "own_telephone":         "none",
            "foreign_worker":        "yes",
        }

        input_df = pd.DataFrame([raw])

        # Encode categoricals using saved label encoders
        for col in categorical_cols:
            le = label_encoders[col]
            val = str(input_df.at[0, col])
            # Handle unseen labels gracefully
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                # Fall back to the most frequent class (index 0)
                input_df[col] = 0

        # Ensure column order matches training
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)

        # ── Predict ────────────────────────────────────────────────────
        prediction = model.predict(input_scaled)[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)[0]
            confidence = prob[prediction] * 100
        else:
            confidence = None

        # ── Display result ─────────────────────────────────────────────
        print()
        hr("═")

        if prediction == 1:
            print(c("\n  ✔  LOAN APPLICATION APPROVED ✔", GREEN, BOLD))
            hr("─")
            print(f"\n  {c('Requested Amount :', DIM)} ₹{loan_inr:,.0f}")
            print(f"  {c('Duration         :', DIM)} {int(duration)} months")
            if confidence:
                print(f"  {c('Model Confidence :', DIM)} {confidence:.1f}%")
            print()
            print(c("  Your profile matches low-risk credit historical data.", GREEN))
            print("  You are likely eligible for standard bank loans.")
            print()
            print(c("  Suggested next steps:", YELLOW, BOLD))
            steps = [
                "Apply at HDFC / ICICI / Axis Bank for the best personal loan rates.",
                "Check pre-approved offers in your bank's mobile app.",
                "Compare EMIs on BankBazaar or PaisaBazaar before signing.",
                "Ensure all required documents (ITR, salary slips) are ready.",
            ]
            for s in steps:
                print(f"      {c('▸', GREEN)} {s}")
        else:
            suggest_banks(loan_inr, duration,
                          credit_history_score, emp_idx)

        print("\n" + "═" * 58 + "\n")

    except KeyboardInterrupt:
        print(c("\n\n  [Exited] — Goodbye!\n", DIM))
        sys.exit(0)
    except ValueError as e:
        print(c(f"\n  ✗  Input error: {e}", RED))
        sys.exit(1)


if __name__ == "__main__":
    run()