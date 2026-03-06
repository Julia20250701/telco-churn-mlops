from pathlib import Path

import pandas as pd


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """
    Load the raw churn dataset from a CSV file.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    raw_path = Path("data/raw/telco_churn.csv")
    df = load_raw_data(raw_path)
    print(df.head())
    print(df.shape)
