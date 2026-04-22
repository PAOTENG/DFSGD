# Spam Group Detection System — Quick Start

Python interpreter: `D:\Environment\Anaconda\envs\SBA\python.exe`
All experiments are conducted on Amazon datasets, using reviews collected from January to June 2013.
---

## Basic Usage

```bash
# Run the full pipeline (modules 1–8) with the default Electronics dataset
python spam_group_detection.py

# Specify a dataset
python spam_group_detection.py --dataset DataSet/Cell_Phones_and_Accessorie.db
python spam_group_detection.py --dataset DataSet/Electronics_2013_1.6.db
```

---

## Resume from a Specific Module

```bash
# Start from module 6 (cached results from modules 1–5 will be reused)
python spam_group_detection.py --dataset DataSet/Electronics_2013_1.6.db --start_module 6

# Run module 8 only (validation and output)
python spam_group_detection.py --dataset DataSet/Electronics_2013_1.6.db --start_module 8 --end_module 8
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `DataSet/Electronics_2013_1.6.db` | Path to the SQLite database |
| `--start_module` | `1` | First module to run (1–8) |
| `--end_module` | `8` | Last module to run (1–8) |
| `--attraction_threshold` | `0.92` | Attraction graph similarity threshold |
| `--repulsion_threshold` | `0.60` | Repulsion graph similarity threshold |
| `--lambda_factor` | `0.5` | Adjacency matrix enhancement factor λ |
| `--group_threshold` | `0.7` | Spam group classification threshold δ_g |

---

## Examples

```bash
# Retrain the GAT model on the Cell Phones dataset
python spam_group_detection.py --dataset DataSet/Cell_Phones_and_Accessorie.db --retrain
```
