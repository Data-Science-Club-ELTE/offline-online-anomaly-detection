import pandas as pd 

def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df 

def main():
    df = load_data()
    print("Dataset loaded successfully!")
    print(df.head())

if __name__ == "__main__":
    main()