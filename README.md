# Credit Card Fraud Detection

A machine learning project that trains a Multi-Layer Perceptron (MLP) on real-world credit card transaction data to classify transactions as legitimate or fraudulent.

---

## Dataset

The dataset is sourced from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the Machine Learning Group at ULB (Université Libre de Bruxelles).

### Download Instructions

1. Install the Kaggle CLI if you haven't already:
   ```bash
   pip install kaggle
   ```

2. Download and unzip the dataset:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud
   unzip creditcardfraud.zip
   ```

4. Place `creditcard.csv` in the root of this project directory.

### Dataset Overview

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (~0.17%) |
| Features | 31 |

**Features:**
- `Time` — seconds elapsed since the first transaction in the dataset
- `V1`–`V28` — anonymized features, likely the result of PCA dimensionality reduction applied to protect user identity and sensitive information
- `Amount` — transaction amount in USD
- `Class` — label: `0` = legitimate, `1` = fraudulent

---

## Project Overview

This project addresses the challenge of detecting credit card fraud in a heavily **class-imbalanced** dataset where fewer than 0.2% of transactions are fraudulent. A naive model that labels everything as legitimate would achieve ~99.8% accuracy while being useless in practice. To combat this, the project uses several techniques:

- **Stratified train/test split** — ensures both sets contain a representative proportion of fraud cases
- **Weighted loss function** — assigns a higher penalty (`pos_weight=100`) to missed fraud predictions so the model does not ignore the minority class
- **Threshold tuning** — instead of the default 0.5 decision threshold, the optimal threshold is selected by maximizing the fraud F1-score on a precision-recall curve

### Models

**FraudMLP (v1)** — Baseline 3-layer perceptron with Dropout regularization:
```
Linear(30 → 64) → ReLU → Dropout(0.3)
Linear(64 → 32) → ReLU → Dropout(0.3)
Linear(32 → 1)
```

**FraudMLP_v2** — Improved version adding Batch Normalization after each linear layer for faster, more stable convergence:
```
Linear(30 → 64) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(64 → 32) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(32 → 1)
```

Both models use:
- **Loss:** `BCEWithLogitsLoss` (numerically stable binary cross-entropy)
- **Optimizer:** Adam (`lr=1e-3`)
- **Scheduler:** `ReduceLROnPlateau` — halves the learning rate when validation loss plateaus

### Results

| Model | ROC-AUC | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---|---|---|---|---|
| FraudMLP v1 (threshold=0.5) | 0.9752 | 99.82% | 0.48 | 0.88 | 0.62 |
| FraudMLP v2 (tuned threshold) | 0.9789 | 99.94% | 0.83 | 0.81 | **0.82** |

Accuracy alone is a misleading metric for this dataset. **ROC-AUC** and **Fraud F1** are the primary indicators of model quality.

---

## Dependencies

Install all required packages with:

```bash
pip install pandas matplotlib seaborn scikit-learn torch numpy
```

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Exploratory data analysis and visualization |
| `scikit-learn` | Preprocessing, train/test split, and evaluation metrics |
| `torch` (PyTorch) | Model definition, training loop, and inference |

---

## Usage

1. Complete the dataset download steps above so `creditcard.csv` is in the project root.
2. Open `fraud.ipynb` in Jupyter or VS Code.
3. Run all cells from top to bottom.

The notebook is organized into the following sections:
1. **Exploratory Analysis** — class distribution, time/amount distributions, null checks
2. **Preprocessing** — scaling `Time` and `Amount`, stratified split, class weighting
3. **Model v1** — baseline MLP training and evaluation
4. **Model v2** — BatchNorm MLP with precision-recall threshold tuning and final evaluation
