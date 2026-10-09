# MAS for Fraud Detection

SERA is a multi-agent framework for payment fraud detection that coordinates schema-aware representations, admissible historical context, typed temporal graph evidence, and risk-constrained alert calibration.

This repository provides the **SERA model pipeline** for the research **Multi-Agent Semantic–Relational Evidence Coordination and Low-False-Positive Risk Calibration for Payment Fraud**.

## Overview

Transaction fraud can depend on the meaning of individual fields, earlier activity, and relationships between accounts and counterparties. SERA brings these sources of evidence into a common scoring pipeline and uses a separate calibration stage to select an alert threshold under a false-positive budget.

## Method

### Schema-Aware Field Representation

Field specifications encode attribute meanings, units, entity roles, and admissible values. The semantic module converts these specifications and transaction values into a common representation across heterogeneous transaction schemas.

### Admissible Historical Context

The context module admits earlier observations according to temporal and business constraints. It learns weights over the admitted history and aggregates that evidence for the current transaction.

### Typed Temporal Graph Evidence

Typed relations connect entities and counterparties over time. The graph module builds evidence from transaction streams and temporal paths, supplying relational features and inspectable pattern summaries to the detector.

### Detection and Risk Calibration

An XGBoost detector combines base, semantic, historical, and graph features. Training and model selection precede threshold calibration on held-out negative examples. The calibration stage uses the configured false-positive target and violation-probability budget; the test report records both measured performance and constraint checks. TreeSHAP provides feature-level explanations.

## Datasets

The evaluation uses six transaction datasets. **Download data separately** using the [data preparation guide](dataset/README.md). The repository includes expected directories, CSV schemas, and a [SHA-256 reference manifest](dataset/manifest.json).

| Selector | Dataset | Expected file under `dataset/` |
|---|---|---|
| `banksim` | BankSim | `banksim/bs140513_032310.csv` |
| `sparkov` | Sparkov | `sparkov/transactions_full.csv` |
| `ieee-cis` | IEEE-CIS prepared features | `ieee-cis/ieee_cis_fraud_features.csv` |
| `ibm-aml` | IBM-AML HI-Small | `ibm-aml/HI-Small_Trans.csv` |
| `ibm-aml-medium` | IBM-AML HI-Medium | `ibm-aml-medium/HI-Medium_Trans.csv` |
| `ibm-aml-li` | IBM-AML LI-Small | `ibm-aml-li/LI-Small_Trans.csv` |

The supplied configurations use a time-ordered window of 400,000 records for BankSim, Sparkov, and IEEE-CIS; 600,000 for IBM-AML HI-Small; and 2,000,000 for IBM-AML HI-Medium and LI-Small. Each window is split chronologically into training, tuning, calibration, and test segments with target proportions of 60:15:10:15. Boundaries keep equal timestamps together, so actual segment sizes can differ. The loaders select the central chronological window; preprocessing is fitted on the training segment.

## Metrics

The pipeline reports Precision, Recall, F1, Average Precision (AP), AUC-ROC, Accuracy, and false-positive rate (FPR), together with calibration, capacity, and contract diagnostics.

## Main Results

The manuscript reports the following SERA results. These are research reference values, not results generated during repository packaging.

| Dataset | Accuracy | Precision | Recall | AUC-ROC |
|---|---:|---:|---:|---:|
| IBM-AML HI-Small | 0.9913 | 0.0409 | 0.5000 | 0.9537 |
| Sparkov | 0.9950 | 0.4272 | 0.9865 | 0.9950 |
| IEEE-CIS | 0.9689 | 0.5957 | 0.3097 | 0.8197 |
| IBM-AML HI-Medium | 0.9897 | 0.0225 | 0.4012 | 0.9266 |
| IBM-AML LI-Small | 0.9858 | 0.0187 | 0.4247 | 0.9536 |
| BankSim | 0.9954 | 0.7856 | 0.7961 | 0.9965 |

Reproduction depends on the data export, window, configuration, seed, and software versions. The release retains the supplied full-model configurations. Baseline implementations, manuscript assets, and experiment outputs are not distributed here.

## Ablation and Mechanism Analysis

The research examines field encoding, admitted history, graph evidence, and threshold calibration through ablations and mechanism diagnostics. The public entry point runs the full SERA pipeline; separate comparison and ablation runners are outside this release.

## Sensitivity

The research studies history count, lookback window, target FPR, and calibration risk budget. Dataset-specific full-model settings are stored in `model/config/datasets.json` and recorded in each run's JSON output.

## Code Structure

- `model/SERA.py`: public `load_config` and `run` entry points.
- `model/sera/semantic.py`, `context.py`, and `graph.py`: field, historical, and relational evidence.
- `model/sera/detector.py` and `constraint.py`: fusion scoring and risk calibration.
- `model/sera/pipeline.py`: training, selection, calibration, and evaluation.
- `model/sera/contracts.py`, `datasets.py`, `features.py`, and `metrics.py`: checks, data handling, base features, and metrics.
- `model/config/`: dataset configurations and CSV schemas.

## Implementation Notes

Use Python 3.10 in a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

[requirements-tested.txt](requirements-tested.txt) records the versions used for release validation. Choose a PyTorch build appropriate for the local CPU/CUDA environment. This executable pipeline uses local numerical models and does not require an LLM API key. External agent-system orchestration and its TSR/ASR experiments are not part of this entry point.

List datasets and check imports/configuration without training:

```bash
python main.py --list-datasets
python main.py --dataset banksim --check-only
```

After preparing data, verify the selected CSV against the supplied reference:

```bash
python main.py --dataset banksim --verify-data
```

Train and evaluate the complete model:

```bash
python main.py --dataset banksim --seed 2026 --output outputs/sera
```

Use an external data directory without copying files:

```bash
python main.py --dataset banksim --datasets-dir /path/to/dataset --output outputs/sera
```

Run a smaller functional check:

```bash
python main.py --dataset banksim --window-rows 20000 --output outputs/smoke
```

Changing `--window-rows` changes the evaluation protocol and is not a reproduction of the reported table. `--check-only` verifies imports and configuration; `--verify-data` reads the CSV and checks its header, size, and SHA-256. Neither option trains a model.

With no arguments, `main.py` prints usage. Default data and output locations are relative to the repository; `--datasets-dir` and `--output` override them. The environment variables `MAS_DATASET_DIR` and `MAS_OUTPUT_DIR` are also supported. Results are written to `<output>/<dataset>_full_seed<seed>.json`; cached windows remain local under `<output>/cache/`. Delete the corresponding local cache after replacing data in place.

Release validation covered imports and configurations for all six datasets, checksums for the original local CSVs, and an end-to-end BankSim run on 20,000 records. The smoke run's metrics, calibration, model selection, and contract diagnostics matched the supplied implementation exactly. This functional check does not establish full reproduction of the manuscript results.

## Repository Layout

```text
MAS-for-Fraud-Detection/
|-- dataset/                  download instructions, schemas, and checksums
|   |-- banksim/
|   |-- sparkov/
|   |-- ieee-cis/
|   |-- ibm-aml/
|   |-- ibm-aml-medium/
|   |-- ibm-aml-li/
|   |-- prepare.py            convert downloaded Parquet files to CSV
|   |-- csv_schema.json
|   `-- manifest.json
|-- model/
|   `-- SERA.py               full SERA pipeline and dataset configurations
|-- main.py                   training, evaluation, and data verification
|-- requirements.txt
|-- requirements-tested.txt
`-- README.md
```

The layout follows the research-repository organization of [MAGNET](https://github.com/AI-FST/MAGNET) and [AI-FST](https://github.com/yuanxu-finance/AI-FST).
