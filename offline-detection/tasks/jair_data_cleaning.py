import pandas as pd
from data_loading_and_exploration import load_creditcard_fraud

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows"""
    initial = df.shape[0]
    df_clean = df.drop_duplicates(keep="first").copy()
    print(f"Removed {initial - df_clean.shape[0]} duplicate rows")
    return df_clean

def handle_missing_values(df: pd.DataFrame, target_col="Class") -> pd.DataFrame:
    """Handle missing values"""
    df_clean = df.copy()
    
    # Drop rows with missing target
    if target_col in df_clean.columns:
        df_clean = df_clean.dropna(subset=[target_col])
    
    # Numeric → median
    num_cols = df_clean.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    
    # Categorical → mode, fallback "unknown"
    cat_cols = df_clean.select_dtypes(exclude="number").columns.tolist()
    for c in cat_cols:
        if df_clean[c].isna().any():
            mode = df_clean[c].mode(dropna=True)
            df_clean[c] = df_clean[c].fillna(mode.iloc[0] if not mode.empty else "unknown")
    
    return df_clean

def detect_inconsistent_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and fix inconsistent values"""
    df_clean = df.copy()
    
    # Example: no negative values in Amount, Time should be >=0
    for col in ["Amount", "Time"]:
        if col in df_clean.columns:
            neg_count = (df_clean[col] < 0).sum()
            if neg_count > 0:
                print(f"Correcting {neg_count} negative values in {col} → setting to 0")
                df_clean.loc[df_clean[col] < 0, col] = 0
    
    # Add other inconsistency checks if needed
    return df_clean

def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that do not contribute"""
    df_clean = df.copy()
    # Example: drop ID if exists
    irrelevant = [c for c in df_clean.columns if "id" in c.lower()]
    if irrelevant:
        print(f"Removing irrelevant columns: {irrelevant}")
        df_clean = df_clean.drop(columns=irrelevant)
    return df_clean

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps"""
    df_clean = remove_duplicates(df)
    df_clean = handle_missing_values(df_clean)
    df_clean = detect_inconsistent_values(df_clean)
    df_clean = remove_irrelevant_columns(df_clean)
    return df_clean

def save_clean_dataset(df: pd.DataFrame, filename="creditcard_cleaned.csv") -> None:
    df.to_csv(filename, index=False)
    print(f"Clean dataset saved to {filename}")

def generate_cleaning_report(df_before: pd.DataFrame, df_after: pd.DataFrame, filename="cleaning_report.txt") -> None:
    with open(filename, "w",encoding="utf-8") as f:
        f.write("=== Cleaning Report ===\n")
        f.write(f"Initial shape: {df_before.shape}\n")
        f.write(f"Cleaned shape: {df_after.shape}\n\n")
        
        f.write("Removed duplicates, handled missing values, corrected inconsistent values, removed irrelevant columns.\n")
        f.write("Numeric columns → missing values filled with median.\n")
        f.write("Categorical columns → missing values filled with mode or 'unknown'.\n")
        f.write("Potentially inconsistent values corrected (Amount, Time ≥ 0).\n")
        f.write("Irrelevant columns removed (IDs, etc.)\n")
    print(f"Cleaning report saved to {filename}")

if __name__ == "__main__":
    df = load_creditcard_fraud()
    df_clean = clean_data(df)
    save_clean_dataset(df_clean)
    generate_cleaning_report(df, df_clean)