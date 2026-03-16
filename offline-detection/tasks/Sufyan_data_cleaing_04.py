import pandas as pd
from data_loading_and_exploration import load_creditcard_fraud


def report_data_quality(df: pd.DataFrame, title: str) -> None:
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    print(f"\n===== {title} =====")
    print("Shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Total missing values:", missing.sum())
    if len(missing_cols) == 0:
        print("Columns with missing values: 0")
    else:
        print(f"Columns with missing values: {len(missing_cols)}")
        for col, count in missing_cols.items():
            print(f"  - {col}: {count}")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    deduped = df.drop_duplicates(keep="first").copy()
    print(f"\nRemoved duplicate rows: {df.shape[0] - deduped.shape[0]}")
    return deduped


def handle_missing_values(df: pd.DataFrame, target_col: str = "Class") -> pd.DataFrame:
    if df.isna().sum().sum() == 0:
        print("\nNo missing values found. No imputation needed.")
        return df

    cleaned = df.dropna(subset=[target_col]).copy()
    print(f"\nDropped rows with missing target '{target_col}': {df.shape[0] - cleaned.shape[0]}")

    # Numeric features → median
    num_cols = [c for c in cleaned.select_dtypes("number").columns if c != target_col]
    cleaned[num_cols] = cleaned[num_cols].fillna(cleaned[num_cols].median())

    # Object features → mode, fallback to "unknown"
    obj_cols = cleaned.select_dtypes(exclude="number").columns
    for col in obj_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode(dropna=True).iloc[0] if not cleaned[col].mode(dropna=True).empty else "unknown")

    print("Filled missing values: numeric → median, categorical → mode.")
    return cleaned


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    report_data_quality(df, "Before Cleaning")
    df = remove_duplicates(df)
    df = handle_missing_values(df, target_col="Class")
    report_data_quality(df, "After Cleaning")
    return df


if __name__ == "__main__":
    data = load_creditcard_fraud()
    cleaned_data = clean_data(data)