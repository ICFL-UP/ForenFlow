"""Bring the three training sources onto the 18-feature schema and merge them (Chapter 4, 4.2.2)."""
import numpy as np
import pandas as pd

import config

PROTOCOL_CODES = {"tcp": 6, "udp": 17, "icmp": 1, "gre": 47}
NORMAL_LABELS = {"normal", "benign"}


def _is_text(series):
    return series.dtype == object or str(series.dtype) in ("string", "str")


def _encode_protocol(series):
    if _is_text(series):
        return series.str.lower().map(PROTOCOL_CODES).fillna(0).astype(int)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _binarise_label(series):
    if _is_text(series):
        return (~series.str.lower().isin(NORMAL_LABELS)).astype(int)
    return (series != 0).astype(int)


def _missing(df, column):
    return column not in df.columns or df[column].isna().all()


def _process_dataset1(path):
    print("  Processing Dataset 1 (Cybersecurity Intrusion) ...")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    label_col = next((c for c in ["AttackLabel", "attack_label", "Label", "label", "Attack"]
                      if c in df.columns), None)
    if label_col is None:
        raise KeyError(f"No label column found in dataset1. Columns: {list(df.columns)}")
    df[config.LABEL_COLUMN] = _binarise_label(df[label_col])
    if "Protocol" in df.columns:
        df["Protocol"] = _encode_protocol(df["Protocol"])
    for col in config.FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if _missing(df, "ByteRatio"):
        df["ByteRatio"] = df["BytesSent"] / df["BytesReceived"]
    if _missing(df, "TimeGap"):
        # no per-packet timestamps in this source, so the gap is approximated (Chapter 4, 4.2.2)
        df["TimeGap"] = df["ActiveTime"] + df["IdleTime"]
    if _missing(df, "FlowRate"):
        df["FlowRate"] = df["BytesSent"] / df["FlowDuration"] * 1000
    if "FlagCount" in df.columns:
        df.loc[df["Protocol"] != 6, "FlagCount"] = 0
    return df


def _process_dataset2(path):
    """LUFlow 2022 subset: flow-level fields renamed and derived features added."""
    print("  Processing Dataset 2 (LUFlow 2022) ...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    rename = {
        "PROTOCOL": "Protocol", "protocol": "Protocol", "proto": "Protocol",
        "IN_BYTES": "BytesSent", "bytes_in": "BytesSent",
        "OUT_BYTES": "BytesReceived", "bytes_out": "BytesReceived",
        "IN_PKTS": "NumPackets", "num_pkts_in": "NumPackets",
        "FLOW_DURATION_MILLISECONDS": "FlowDuration", "duration": "FlowDuration",
        "L4_SRC_PORT": "SrcPort", "src_port": "SrcPort",
        "L4_DST_PORT": "DstPort", "dest_port": "DstPort",
        "TCP_FLAGS": "FlagCount",
        "entropy": "Entropy", "total_entropy": "Entropy",
        "Label": config.LABEL_COLUMN, "label": config.LABEL_COLUMN,
    }
    mapping = {}
    for old, new in rename.items():
        if old in df.columns and new not in df.columns and new not in mapping.values():
            mapping[old] = new
    df.rename(columns=mapping, inplace=True)

    df["Protocol"] = _encode_protocol(df["Protocol"])
    df[config.LABEL_COLUMN] = _binarise_label(df[config.LABEL_COLUMN])
    for col in config.FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -1 marks "no port" for ICMP/GRE
    portless = ~df["Protocol"].isin([6, 17])
    for col in ("SrcPort", "DstPort"):
        if col in df.columns:
            df.loc[portless, col] = -1
    if "FlagCount" in df.columns:
        df.loc[df["Protocol"] != 6, "FlagCount"] = 0

    if _missing(df, "PacketSize"):
        df["PacketSize"] = (df["BytesSent"] + df["BytesReceived"]) / df["NumPackets"]
    if _missing(df, "ResponseTime"):
        df["ResponseTime"] = df["FlowDuration"] / df["NumPackets"]
    if _missing(df, "PacketRate"):
        df["PacketRate"] = df["NumPackets"] / df["FlowDuration"] * 1000
    if _missing(df, "PayloadSize"):
        df["PayloadSize"] = df["PacketSize"] * 0.8
    if _missing(df, "FlowRate"):
        df["FlowRate"] = df["BytesSent"] / df["FlowDuration"] * 1000
    if _missing(df, "ByteRatio"):
        df["ByteRatio"] = df["BytesSent"] / df["BytesReceived"]
    if _missing(df, "ActiveTime"):
        df["ActiveTime"] = (df["NumPackets"].clip(lower=1) * df["ResponseTime"]).clip(upper=df["FlowDuration"])
    if _missing(df, "IdleTime"):
        df["IdleTime"] = (df["FlowDuration"] - df["ActiveTime"]).clip(lower=0)
    if _missing(df, "TimeGap"):
        df["TimeGap"] = df["FlowDuration"] / df["NumPackets"]
    return df


def _process_dataset3(path):
    print("  Processing Dataset 3 (Testbed) ...")
    return pd.read_csv(path)


def preprocess_all():
    sources = [
        ("dataset1_intrusion.csv", _process_dataset1),
        ("dataset2_luflow.csv", _process_dataset2),
        ("dataset3_testbed.csv", _process_dataset3),
    ]
    keep = config.FEATURE_COLUMNS + [config.LABEL_COLUMN]
    frames = []
    for i, (filename, process) in enumerate(sources, 1):
        path = config.RAW_DIR / filename
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        df = process(path).reindex(columns=keep)
        frames.append(df)
        print(f"  Dataset {i}: {len(df)} rows, {df[config.LABEL_COLUMN].sum():.0f} attacks")
    if not frames:
        raise RuntimeError("No datasets found. Run download_data.py first.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Raw combined: {len(combined)} rows")
    # a division by zero upstream becomes inf here, then NaN, then the column median
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing_count = combined[config.FEATURE_COLUMNS].isna().sum(axis=1)
    before = len(combined)
    combined = combined[missing_count <= config.NUM_FEATURES * 0.5].reset_index(drop=True)
    print(f"  Dropped {before - len(combined)} rows with >50% missing features")

    before = len(combined)
    combined.drop_duplicates(subset=config.FEATURE_COLUMNS, inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"  Dropped {before - len(combined)} duplicate rows")

    combined["FlagCount"] = combined["FlagCount"].fillna(0)
    for col in config.FEATURE_COLUMNS:
        median = combined[col].median()
        combined[col] = combined[col].fillna(0 if pd.isna(median) else median).astype(np.float32)
    combined[config.LABEL_COLUMN] = combined[config.LABEL_COLUMN].astype(int)
    print(f"Combined: {len(combined)} rows, {combined[config.LABEL_COLUMN].value_counts().to_dict()}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "combined_dataset.csv"
    combined.to_csv(out, index=False)
    print(f"  Saved {len(combined)} rows to {out}\n")


if __name__ == "__main__":
    preprocess_all()
