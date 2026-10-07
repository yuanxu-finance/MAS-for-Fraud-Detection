# Data preparation

Raw transaction data is downloaded separately. This directory provides expected filenames, sources, conversion schemas, and reference checksums. Follow each source's access and redistribution terms.

## BankSim

Download [BankSim](https://www.kaggle.com/datasets/ealaxi/banksim1), extract `bs140513_032310.csv`, and place it at:

```text
dataset/banksim/bs140513_032310.csv
```

## IBM-AML

The [IBM AML-Data repository](https://github.com/IBM/AML-Data) describes the synthetic data and links to the [IBM transactions dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). Download and extract these three files:

```text
dataset/ibm-aml/HI-Small_Trans.csv
dataset/ibm-aml-medium/HI-Medium_Trans.csv
dataset/ibm-aml-li/LI-Small_Trans.csv
```

The manifest also records `ibm-aml/pattern_labels.csv`, an optional sidecar in the original local research package. It is not needed by the released full-model CLI and need not be recreated.

## Sparkov

Use [Nooha/cc_fraud_detection_dataset](https://huggingface.co/datasets/Nooha/cc_fraud_detection_dataset), a prepared Sparkov dataset. The source generator is [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation).

Download both Parquet shards in their listed order:

- [train-00000-of-00002.parquet](https://huggingface.co/datasets/Nooha/cc_fraud_detection_dataset/resolve/a74d09b3fdcef5e29b081982bd2513e96214af59/data/train-00000-of-00002.parquet)
- [train-00001-of-00002.parquet](https://huggingface.co/datasets/Nooha/cc_fraud_detection_dataset/resolve/a74d09b3fdcef5e29b081982bd2513e96214af59/data/train-00001-of-00002.parquet)

From the repository root, convert them to the expected CSV:

```bash
python dataset/prepare.py --dataset sparkov --parquet /path/to/train-00000-of-00002.parquet /path/to/train-00001-of-00002.parquet
```

The output is `dataset/sparkov/transactions_full.csv`.

## IEEE-CIS

Use the prepared feature export [Kshitijbhatt1998/ieee-fraud-detection-pipeline-features](https://huggingface.co/datasets/Kshitijbhatt1998/ieee-fraud-detection-pipeline-features). Download [fraud_features.parquet](https://huggingface.co/datasets/Kshitijbhatt1998/ieee-fraud-detection-pipeline-features/resolve/6e9f80920753c5ab24af9c68e0d198a356994f5a/fraud_features.parquet), then run:

```bash
python dataset/prepare.py --dataset ieee-cis --parquet /path/to/fraud_features.parquet
```

The output is `dataset/ieee-cis/ieee_cis_fraud_features.csv`. The original [IEEE-CIS competition files](https://www.kaggle.com/c/ieee-fraud-detection/data) require feature preparation and cannot simply be renamed to this file: the loader expects the prepared feature schema.

## Export format and verification

The conversion script preserves source row order, applies the recorded [CSV schema](csv_schema.json), and writes missing values as `__MAS_NULL__`. It refuses to overwrite an existing CSV. Sparkov and IEEE-CIS loaders restore the recorded dtypes and categories before feature construction. Conversion and CSV loading materialise data in memory, so allow sufficient RAM for the selected source.

The [manifest](manifest.json) records each original local CSV's filename, byte size, SHA-256, and column order. Check one dataset with:

```bash
python main.py --dataset banksim --verify-data
python main.py --dataset ieee-cis --datasets-dir /path/to/dataset --verify-data
```

The command exits with status 1 for missing required files or a reference mismatch. CSV re-exports can differ in line endings or numeric formatting while representing equivalent records, so a mismatch calls for inspection; matching headers alone do not establish equivalent data. The pinned download links identify the upstream revisions inspected when preparing this release, but the original local exports do not carry upstream revision metadata, so those links are not a byte-for-byte provenance guarantee.

To use another data location, pass `--datasets-dir` to both preparation and model commands. Preserve the six dataset subdirectory names. Raw data, converted CSVs, caches, and run outputs are excluded by `.gitignore`.
