import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loading_and_exploration import load_creditcard_fraud
from Sufyan_data_cleaing_04 import clean_data

# Load and clean data
df = load_creditcard_fraud()
df = clean_data(df)

# Folder for the plots
output_folder = "dataset_analysis_plots"
os.makedirs(output_folder, exist_ok=True)

# Dataset Structure Report
print("\n===== Dataset Structure Report =====")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nColumn data types and null counts:")
dtype_summary = pd.DataFrame({
    "Column": df.columns,
    "DataType": df.dtypes.astype(str),
    "Non-Null Count": df.notna().sum(),
    "Null Count": df.isna().sum()
})
print(dtype_summary.to_string())
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Numeric descriptive statistics
print("\n===== Numeric Feature Summary =====")
numeric_cols = df.select_dtypes(include='number').columns.tolist()
print(df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T)

# Categorical descriptive statistics
cat_cols = df.select_dtypes(include='object').columns.tolist()
if cat_cols:
    print("\n===== Categorical Feature Summary =====")
    print(df[cat_cols].describe().to_string())

# Plot feature distributions (to detect skew or unusual values)
print("\n===== Saving Feature Distributions =====")
for col in numeric_cols:
    if col != "Class":  
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{col}_distribution.png"))
        plt.close()

# Correlation matrix (to find highly correlated features)
print("\n===== Saving Feature Correlations =====")
corr = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "correlation_matrix.png"))
plt.close()

# Show highly correlated pairs (|corr| > 0.8)
high_corr = [(c1, c2, corr_val) 
             for i, c1 in enumerate(corr.columns) 
             for j, c2 in enumerate(corr.columns) 
             if i < j and abs(corr_val := corr.iloc[i,j]) > 0.8]
print("Highly correlated features (|corr| > 0.8):")
for c1, c2, val in high_corr:
    print(f"{c1} ↔ {c2}: {val:.2f}")

# Class imbalance
print("\n===== Saving Class Distribution =====")
plt.figure(figsize=(4,3))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Normal Transactions")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "class_distribution.png"))
plt.close()

# Identify potentially problematic columns
print("\n===== Potentially Problematic Columns =====")
for col in numeric_cols:
    unique_ratio = df[col].nunique() / df.shape[0]
    if unique_ratio < 0.01:  
        print(f"{col} is almost constant ({df[col].nunique()} unique values)")
    if df[col].nunique() > 1 and (df[col].skew() > 5 or df[col].skew() < -5):  
        print(f"{col} is highly skewed (skew={df[col].skew():.2f})")

print(f"\nAll plots saved in folder: {output_folder}")