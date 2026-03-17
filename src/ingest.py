import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

import kaggle  # noqa: E402 — must import after env vars are set

COMPETITION = "home-credit-default-risk"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_data(force: bool = False) -> None:
    """Download all competition files from Kaggle and unzip them into data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    already_downloaded = any(RAW_DIR.glob("*.csv"))
    if already_downloaded and not force:
        print(f"Data already present in {RAW_DIR}. Pass force=True to re-download.")
        return

    print(f"Downloading '{COMPETITION}' dataset from Kaggle...")
    kaggle.api.competition_download_files(
        competition=COMPETITION,
        path=RAW_DIR,
        quiet=False,
    )

    for zip_path in RAW_DIR.glob("*.zip"):
        print(f"Unzipping {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    print(f"Done. Files in {RAW_DIR}:")
    for f in sorted(RAW_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:45s} {size_mb:>8.1f} MB")


def list_files() -> list[Path]:
    """Return sorted list of CSV files in data/raw/."""
    return sorted(RAW_DIR.glob("*.csv"))


if __name__ == "__main__":
    download_data()
