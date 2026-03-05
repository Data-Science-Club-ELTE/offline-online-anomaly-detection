

import kagglehub
import pandas as pd
from pathlib import Path

DATASET_ID = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"


def load_creditcard_fraud() -> pd.DataFrame:
    dataset_path = kagglehub.dataset_download(DATASET_ID)
    csv_file = Path(dataset_path) / CSV_NAME
    return pd.read_csv(csv_file)


def print_alessandro_summary(df: pd.DataFrame) -> None:
    """Alessandro: shape (rows, columns)."""
    print("Shape of the dataset:", df.shape)
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])


def print_nazrin_summary(df: pd.DataFrame) -> None:
    """Nazrin: column names, first 5 rows, dataset info."""
    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head().to_string())

    print("\nDataset Info:")
    df.info()


def print_amina_exploration(df: pd.DataFrame) -> None:
    """Amina: data types summary table, statistical summary, class distribution."""
    print("\n--- Data types and nullability ---")
    df.info()

    print("\n--- Dtype summary (column, dtype, non_null, null_count) ---")
    dtype_summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "null_count": df.isna().sum(),
    })
    print(dtype_summary.to_string())

    print("\n--- Numeric summary (count, mean, std, min, quartiles, max) ---")
    print(df.describe().to_string())

    print("\n--- Extended summary (all columns, custom percentiles) ---")
    extended = df.describe(
        include="all",
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    )
    print(extended.to_string())

    print("\n--- Class distribution (0 = normal, 1 = fraud) ---")
    print(df["Class"].value_counts(dropna=False).to_string())
    print("\nProportions:")
    print(df["Class"].value_counts(normalize=True).to_string())


def run_all(df: pd.DataFrame) -> None:
    """Run all summaries: Alessandro, Nazrin, then Amina."""
    print("========== Alessandro: shape ==========")
    print_alessandro_summary(df)

    print("\n========== Nazrin: columns, head, info ==========")
    print_nazrin_summary(df)

    print("\n========== Amina: data exploration ==========")
    print_amina_exploration(df)


if __name__ == "__main__":
    df = load_creditcard_fraud()
    run_all(df)
