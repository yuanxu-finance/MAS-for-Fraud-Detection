"""Convert user-downloaded Sparkov/IEEE-CIS Parquet files to the expected CSV format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sparkov", "ieee-cis"), required=True)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True, help="Downloaded input files, in source shard order")
    parser.add_argument("--datasets-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    import pandas as pd

    schema = json.loads(Path(__file__).with_name("csv_schema.json").read_text(encoding="utf-8"))[args.dataset]
    frames = []
    for path in args.parquet:
        frame = pd.read_parquet(path)
        missing = set(schema) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}. Use the linked prepared dataset, not a raw alternative.")
        frames.append(frame[list(schema)])
    frame = pd.concat(frames, ignore_index=True)
    for name, spec in schema.items():
        dtype = spec["dtype"]
        if dtype.startswith("datetime"):
            frame[name] = pd.to_datetime(frame[name]).astype(dtype)
        elif dtype == "category":
            unknown = set(frame[name].dropna().unique()) - set(spec["categories"])
            if unknown:
                raise ValueError(f"Unrecognised categories in {name}")
            frame[name] = pd.Categorical(frame[name], categories=spec["categories"], ordered=spec["ordered"])
        else:
            frame[name] = frame[name].astype(dtype)
        if dtype in ("object", "string", "category") and frame[name].eq("__MAS_NULL__").any():
            raise ValueError(f"Reserved CSV null marker occurs in {name}")
    filename = "transactions_full.csv" if args.dataset == "sparkov" else "ieee_cis_fraud_features.csv"
    folder = args.datasets_dir / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    # Exclusive creation protects an existing research data export.
    with target.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, na_rep="__MAS_NULL__", lineterminator="\n")
    print(f"Exported {len(frame):,} records to {target.resolve()}")
    print("The reference manifest records the original CSV bytes; a re-export may have a different hash.")


if __name__ == "__main__":
    main()
