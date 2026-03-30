"""
Task file for Issue #31 – Implement cleaning method
====================================================
This script demonstrates and verifies each cleaning step applied to the
credit card fraud dataset before it enters the Isolation Forest pipeline.

Run from the repo root:
    python offline-detection/tasks/implement_cleaning.py
"""

import sys
from pathlib import Path

# Allow importing from the parent offline-detection directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from main import load_data, clean_data

SEPARATOR = "-" * 60


def report(label: str, df: pd.DataFrame) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(SEPARATOR)
    print(f"  Shape          : {df.shape}")
    print(f"  Duplicate rows : {df.duplicated().sum()}")
    print(f"  Missing values : {df.isna().sum().sum()}")
    if "Amount" in df.columns:
        print(f"  Amount  < 0    : {(df['Amount'] < 0).sum()}")
    if "Time" in df.columns:
        print(f"  Time    < 0    : {(df['Time'] < 0).sum()}")


def run_assertions(original: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    print(f"\n{SEPARATOR}")
    print("  Running assertions …")
    print(SEPARATOR)

    assert cleaned.duplicated().sum() == 0, \
        "FAIL: Duplicate rows still present after cleaning."
    print("  PASS: No duplicate rows.")

    assert cleaned.isna().sum().sum() == 0, \
        "FAIL: Missing values still present after cleaning."
    print("  PASS: No missing values.")

    if "Amount" in cleaned.columns:
        assert (cleaned["Amount"] >= 0).all(), \
            "FAIL: Negative Amount values remain."
        print("  PASS: Amount values are non-negative.")

    if "Time" in cleaned.columns:
        assert (cleaned["Time"] >= 0).all(), \
            "FAIL: Negative Time values remain."
        print("  PASS: Time values are non-negative.")

    assert len(cleaned) <= len(original), \
        "FAIL: Row count increased after cleaning."
    print(f"  PASS: Row count went from {len(original):,} → {len(cleaned):,} "
          f"(removed {len(original) - len(cleaned):,} row(s)).")

    print(f"\n{SEPARATOR}")
    print("  All assertions passed. ✓")
    print(SEPARATOR)


if __name__ == "__main__":
    print("\nLoading data …")
    data, target = load_data()

    report("BEFORE cleaning", data)

    print("\nRunning clean_data() …\n")
    cleaned = clean_data(data)

    report("AFTER cleaning", cleaned)

    run_assertions(data, cleaned)
