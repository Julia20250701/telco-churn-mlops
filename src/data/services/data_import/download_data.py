import shutil
from pathlib import Path

import kagglehub


def download_telco_data() -> Path:
    """
    Download the Telco churn dataset from Kaggle and copy the CSV
    into the project's data/raw directory.
    """
    download_path = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))

    target_dir = Path("data/raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(download_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded path: {download_path}")

    source_file = csv_files[0]
    target_file = target_dir / "telco_churn.csv"

    shutil.copy2(source_file, target_file)

    return target_file


if __name__ == "__main__":
    saved_path = download_telco_data()
    print(f"Dataset copied to: {saved_path}")
