"""Dataset loading, 70/10/20 split, scaling before windowing, DataLoader creation."""
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from config import (BATCH_SIZE, FEATURE_COLUMNS, LABEL_COLUMN, MODEL_DIR, PROCESSED_DIR,
                    RANDOM_SEED, SEQUENCE_LENGTH, SEQUENCE_STRIDE, TEST_SPLIT, VAL_SPLIT)


class SequenceDataset(Dataset):
    """Overlapping windows of seq_len consecutive (already scaled) flow vectors."""

    def __init__(self, features, labels, seq_len=SEQUENCE_LENGTH, stride=SEQUENCE_STRIDE):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.seq_len = seq_len
        self.starts = list(range(0, len(features) - seq_len + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len
        return (torch.from_numpy(self.features[start:end]),
                torch.tensor([self.labels[end - 1]], dtype=torch.float32))


def load_dataframe(path=None):
    path = path or PROCESSED_DIR / "combined_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"No processed data at {path}. Run preprocess.py first.")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {path.name}")

    df = df[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with NaN/inf values")
    return df


def split_dataframe(df):
    """Stratified 70/10/20 split."""
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=df[LABEL_COLUMN])
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SPLIT / (1 - TEST_SPLIT), random_state=RANDOM_SEED,
        stratify=train_val_df[LABEL_COLUMN])
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    for name, part in (("Train", train_df), ("Val  ", val_df), ("Test ", test_df)):
        print(f"  {name} attack ratio: {part[LABEL_COLUMN].mean():.3f}")
    return train_df, val_df, test_df


def scale_splits(train_df, val_df, test_df):
    """Fit the scaler on the training rows only, then scale every flow vector once, before
    windowing."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    X_val = scaler.transform(val_df[FEATURE_COLUMNS].values)
    X_test = scaler.transform(test_df[FEATURE_COLUMNS].values)
    return scaler, X_train, X_val, X_test


def make_loader(X, y, seq_len=SEQUENCE_LENGTH, shuffle=False, batch_size=BATCH_SIZE):
    return DataLoader(SequenceDataset(X, y, seq_len=seq_len), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)


def create_dataloaders(batch_size=BATCH_SIZE, scaler_dir=MODEL_DIR):
    train_df, val_df, test_df = split_dataframe(load_dataframe())
    scaler, X_train, X_val, X_test = scale_splits(train_df, val_df, test_df)

    scaler_dir.mkdir(parents=True, exist_ok=True)
    with open(scaler_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved to {scaler_dir / 'scaler.pkl'}")

    y_train, y_val, y_test = (part[LABEL_COLUMN].values for part in (train_df, val_df, test_df))
    train_loader = make_loader(X_train, y_train, shuffle=True, batch_size=batch_size)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size)
    test_loader = make_loader(X_test, y_test, batch_size=batch_size)
    print(f"  Sequences: train={len(train_loader.dataset)} | val={len(val_loader.dataset)} "
          f"| test={len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader, scaler


def create_normal_only_loaders(scaler, batch_size=BATCH_SIZE):
    """Normal-traffic-only loaders: the autoencoder must never see an attack while training."""
    train_df, val_df, _ = split_dataframe(load_dataframe())
    train_normal = train_df[train_df[LABEL_COLUMN] == 0]
    val_normal = val_df[val_df[LABEL_COLUMN] == 0]
    print(f"  Normal-only: train={len(train_normal)} | val={len(val_normal)}")
    return (
        make_loader(scaler.transform(train_normal[FEATURE_COLUMNS].values),
                    train_normal[LABEL_COLUMN].values, shuffle=True, batch_size=batch_size),
        make_loader(scaler.transform(val_normal[FEATURE_COLUMNS].values),
                    val_normal[LABEL_COLUMN].values, batch_size=batch_size),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("F-RIaaS Dataset Loading")
    print("=" * 60)
    train_loader, _, _, _ = create_dataloaders()
    X_batch, y_batch = next(iter(train_loader))
    print(f"\n  Sample batch shape: X={X_batch.shape}, y={y_batch.shape}")
    print(f"  X range: [{X_batch.min():.3f}, {X_batch.max():.3f}]")
    print(f"  y distribution: {y_batch.mean():.3f} attack ratio")
