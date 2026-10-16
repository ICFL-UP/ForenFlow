"""Download training datasets from Kaggle using kagglehub."""
import os
import shutil
import glob
try:
    import kagglehub
except ImportError:
    kagglehub = None
import config


def download_datasets():
    if kagglehub is None:
        raise ImportError("kagglehub not installed, skipping Kaggle download")
    os.makedirs(config.RAW_DIR, exist_ok=True)

    print("[1/2] Downloading Cybersecurity Intrusion dataset ...")
    path1 = kagglehub.dataset_download(
        "smmmmmmmmmmmm/cybersecurity-intrusion-simulated-network"
    )
    csvs = glob.glob(os.path.join(path1, "**", "*.csv"), recursive=True)
    if csvs:
        dst = os.path.join(config.RAW_DIR, "dataset1_intrusion.csv")
        shutil.copy2(csvs[0], dst)
        print(f"  -> saved to {dst}")
    else:
        raise FileNotFoundError(f"No CSV found in {path1}")

    print("[2/2] Downloading LUFlow dataset ...")
    path2 = kagglehub.dataset_download(
        "mryanm/luflow-network-intrusion-detection-data-set"
    )
    candidates = glob.glob(os.path.join(path2, "**", "*2022*"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(path2, "**", "*.csv"), recursive=True)

    if candidates:
        src = candidates[0]
        if os.path.isdir(src):
            inner = glob.glob(os.path.join(src, "*.csv"))
            if inner:
                src = inner[0]
        dst = os.path.join(config.RAW_DIR, "dataset2_luflow.csv")
        shutil.copy2(src, dst)
        print(f"  -> saved to {dst}")
    else:
        raise FileNotFoundError(f"No CSV found in {path2}")

    print("All datasets downloaded.\n")


if __name__ == "__main__":
    download_datasets()
