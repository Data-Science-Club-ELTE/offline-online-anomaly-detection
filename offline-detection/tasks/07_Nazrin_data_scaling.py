import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loading_and_exploration import load_creditcard_fraud
from Sufyan_data_cleaing_04 import clean_data

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()

    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    print("\nScaling completed for Time and Amount")

    return df


if __name__ == "__main__":
    df = load_creditcard_fraud()
    df = clean_data(df)

    scaled_df = scale_features(df)

    print("\nScaled statistics:")
    print(scaled_df[['Time', 'Amount']].describe())

    print("\nScaled dataset preview:")
    print(scaled_df.head())
    


