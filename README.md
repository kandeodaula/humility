# Humility: Network Traffic Classification

This repository provides advanced, reproducible pipelines for feature selection, preprocessing, and deep learning model training on large-scale network intrusion datasets, including CICIDS 2017/2018 and LITNET-2020. The code is optimized for maximum resource utilization (L4 GPU + 53GB RAM) and supports both linear and nonlinear feature selection, aggressive data cleaning, and neural network architectures (BiLSTM, Deep Dense, Hybrid).

## Table of Contents
- Features
- Datasets
- Setup & Installation
- Usage
- Project Structure
- Advanced Options
- References
- Troubleshooting
- License

---

## Features
- Aggressive Data Preprocessing: Handles missing, duplicate, and infinite values in parallel.
- Feature Selection: Mutual Information (MIQ), Pearson Correlation, MRMR, and more.
- Balanced Sampling: Ensures equal benign/attack samples for robust training.
- Normalization: MinMaxScaler and RobustScaler for optimal GPU training.
- Neural Architectures: BiLSTM, Deep Dense, Hybrid models with hyperparameter optimization (Optuna).
- GPU/CPU Parallelization: Designed for L4 GPU and multi-core CPUs.
- Reproducible Splits: Consistent train/test splits and feature sets.

---

## Datasets
- CICIDS 2017/2018: Download from [CICIDS](http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip) or [Kaggle](https://www.kaggle.com/chethuhn/network-intrusion-dataset).
- LITNET-2020: Download from [GitHub](https://github.com/Grigaliunas/electronics9050800/raw/refs/heads/main/dataset/ALLinONE.zip).

> **Note:** Place extracted CSV files in the appropriate folders as described below.

---

## Setup & Installation

### 1. Clone the Repository
```powershell
git clone https://github.com/kandeodaula/humility.git
cd humility_code
```

### 2. Install Python Dependencies
Recommended: Python 3.8+

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:
```powershell
pip install pandas numpy scikit-learn tensorflow optuna dcor joblib psutil
```

### 3. Download and Prepare Datasets
- CICIDS 2017/2018: Place extracted CSVs in `CIC-IDS-2017/TrafficLabelling/`
- LITNET-2020: Place `allFlows.csv` in `LITNET-2020/`

---

## Usage

### CICIDS 2017/2018 Pipeline
Run the main script for preprocessing, feature selection, and model training:
```powershell
python cicds_2017_2018.py
```
- Outputs processed data to `cic_2017_processed/` (train/test splits, feature names, summary).
- Supports advanced feature selection and neural architectures.

### LITNET-2020 Pipeline
Run the script for LITNET-2020 preprocessing and training:
```powershell
python compare_linear_and_nonlinear_features.py
```
- Outputs processed data to `litnet_processed_data/` (train/test splits, feature names, metadata).

### Jupyter Notebooks
- cicids_2017_2018.ipnb and compare_linear_and_nonlinear_features.ipnb provide step-by-step, interactive versions of the pipelines.

---

## Project Structure
```
├── cicds_2017_2018.py                # CICIDS 2017/2018 pipeline
├── compare_linear_and_nonlinear_features.py  # LITNET-2020 pipeline
├── cicids_2017_2018.ipnb             # Notebook version
├── compare_linear_and_nonlinear_features.ipnb # Notebook version
├── README.md                         # This file
├── requirements.txt                  # Python dependencies (recommended)
├── cic_2017_processed/               # Output: processed CICIDS data
├── litnet_processed_data/            # Output: processed LITNET data
```

---

## Advanced Options
- GPU Acceleration: Scripts auto-detect and utilize GPU (CuPy, cuDF) if available.
- Hyperparameter Optimization: Enable Optuna for neural network tuning.
- Custom Feature Selection: Change `k` in pipeline scripts to select top-k features.
- Model Architecture: Choose between BiLSTM, Deep Dense, Hybrid in script arguments.

---

## References
- [CICIDS 2017 Dataset](https://www.unb.ca/cic/datasets/malmem-2017.html)
- [LITNET-2020 Dataset](https://github.com/Grigaliunas/electronics9050800)
- [Optuna Hyperparameter Optimization](https://optuna.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Scikit-learn](https://scikit-learn.org/)

---

## Troubleshooting
- Memory Errors: Reduce sample size or chunk size in scripts for lower-resource systems.
- Missing Files: Ensure datasets are downloaded and placed in correct folders.
- GPU Issues: Scripts fall back to CPU if GPU libraries are unavailable.

---

## License
This project is released under the MIT License.

---