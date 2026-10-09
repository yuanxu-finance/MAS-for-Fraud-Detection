"""Command-line entry for the SERA fraud-detection pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DATASETS = ("banksim", "sparkov", "ieee-cis", "ibm-aml", "ibm-aml-medium", "ibm-aml-li")


def verify_data(root: Path, dataset: str) -> bool:
    manifest = json.loads(
        (ROOT / "dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    passed = True
    for entry in manifest["files"]:
        if entry["path"].split("/")[0] != dataset:
            continue
        path = root / entry["path"]
        if not path.is_file():
            print(
                f"{'MISSING' if entry['required'] else 'OPTIONAL, absent'}: {entry['path']}"
            )
            passed &= not entry["required"]
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream))
        exact = (
            digest.hexdigest() == entry["sha256"]
            and path.stat().st_size == entry["bytes"]
        )
        schema_ok = columns == entry["columns"]
        print(
            f"{entry['path']}: reference hash={'MATCH' if exact else 'DIFFERS'}, columns={'MATCH' if schema_ok else 'DIFFER'}"
        )
        passed &= exact and schema_ok
    return passed


def json_value(value):
    """Convert scientific values into strict JSON, including non-finite diagnostics."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SERA: schema, history and graph evidence with held-out risk calibration"
    )
    parser.add_argument("--dataset", choices=DATASETS, default="banksim")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(os.environ.get("MAS_DATASET_DIR", ROOT / "dataset")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("MAS_OUTPUT_DIR", ROOT / "outputs")),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check imports and configuration; does not read data or train",
    )
    parser.add_argument(
        "--verify-data",
        action="store_true",
        help="Check the selected dataset against the reference SHA-256 manifest",
    )
    parser.add_argument(
        "--window-rows",
        type=int,
        help="Override the evaluation window for a smoke run; changes the research protocol",
    )
    args_list = sys.argv[1:] if argv is None else argv
    if not args_list:
        parser.print_help()
        return 0
    args = parser.parse_args(args_list)
    if args.window_rows is not None and args.window_rows <= 0:
        parser.error("--window-rows must be positive")
    if args.list_datasets:
        print("\n".join(DATASETS))
        return 0
    if args.verify_data:
        return 0 if verify_data(args.datasets_dir.resolve(), args.dataset) else 1

    from training.pipeline import load_config, run
    from model.sera.config import validate

    cfg = load_config(args.dataset, args.seed)
    if args.window_rows is not None:
        cfg["window_rows"] = args.window_rows
    validate(cfg)
    if args.check_only:
        print(
            json.dumps(
                {
                    "model": "SERA",
                    "dataset": args.dataset,
                    "imports": "OK",
                    "config": "OK",
                    "window_rows": cfg["window_rows"],
                }
            )
        )
        return 0
    result = run(
        args.dataset, args.seed, args.datasets_dir, args.output, args.window_rows
    )
    result = json_value({k: v for k, v in result.items() if not k.startswith("_")})
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"{args.dataset}_full_seed{args.seed}.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
