## AI Anti-Phishing Mini-Experiment (PhiUSIIL Dataset)

This project provides a **reproducible mini-experiment** for a Bachelor thesis on AI-based anti-phishing detection.
It uses the **UCI PhiUSIIL Phishing URL Dataset** (CSV file) and trains:

- **Baseline model**: Logistic Regression
- **ML model**: Random Forest

Only **numeric features** are used; all text columns (including `FILENAME`, `URL`, `Domain`, `TLD`, `Title`, etc.) are dropped.

### 1. Project structure

- `README.md` – this file, with run instructions
- `requirements.txt` – Python dependencies
- `src/run_experiment.py` – main script (entry point)
- `src/utils.py` – helper functions (data loading, metrics, etc.)
- `outputs/` – folder where all results are saved:
  - `outputs/results_metrics.csv` – metrics for all model/threshold combinations
  - `outputs/feature_importance_top15.csv` – top-15 feature importances for Random Forest

### 2. Python version & environment

Recommended:

- **Python 3.10+**

Create and activate a virtual environment (optional but recommended):

```bash
cd /Users/daniilkukhto/Desktop/Bakalar_prace
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Input data: PhiUSIIL CSV

The script expects a CSV file:

- `PhiUSIIL_Phishing_URL_Dataset.csv`

You must pass the CSV path via the `--csv_path` argument.

Example:

```bash
python -m src.run_experiment --csv_path "PhiUSIIL_Phishing_URL_Dataset.csv"
```

### 4.1. Downloading the full PhiUSIIL dataset

The original PhiUSIIL Phishing URL Dataset is not included in this repository because of its size.

To run the experiment on the full data:

1. Open the UCI Machine Learning Repository page for the dataset:  
   `http://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset`
2. Download the CSV file **`PhiUSIIL_Phishing_URL_Dataset.csv`** from that page.
3. Place the downloaded file in the **project root** (the same folder where `README.md` and `requirements.txt` are located).
4. Run the experiment and point to this file via the `--csv_path` argument, for example:

   ```bash
   python -m src.run_experiment --csv_path "PhiUSIIL_Phishing_URL_Dataset.csv"
   ```

If you want to use a smaller sample for quick testing, you can create a reduced CSV (e.g., the first 10,000 rows) and pass its path instead of the full dataset.

### 5. Experiment design

- **Target variable**:
  - Original column: `label` (1 = legitimate, 0 = phishing)
  - Used in experiment: `y_phish = 1 - label` (so **1 = phishing (positive)**, 0 = legitimate)
- **Features**:
  - Use **only numeric columns**
  - Drop `FILENAME` and **all text/object columns**
- **Train/test split**:
  - Stratified 80/20 split
  - `random_state=42`

### 6. Models

- **Baseline: Logistic Regression**
  - `StandardScaler` on numeric features
  - `LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=200)`

- **ML model: RandomForestClassifier**
  - `RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )`

Both models are trained on the same training set and evaluated on the same test set.

### 7. Evaluation & thresholds

On the **test set**, the script computes:

- **Metrics**:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC (using predicted probabilities)
- **Confusion matrix**:
  - TN, FP, FN, TP
  - Emphasis on **FN** (missed attacks) and **FP** (extra alerts).

Three threshold regimes are evaluated for each model:

1. **Default**: `threshold = 0.50`
2. **Best F1**: threshold that **maximizes F1** over a grid `0.01 .. 0.99`
3. **High recall**:
   - Minimal threshold where **recall == 1.0** (if achievable) on the same grid
   - The script reports how many **FP** are needed to reach this.

All these results are saved to:

- `outputs/results_metrics.csv`

### 8. Feature importance

For the Random Forest model, the script:

- Extracts `feature_importances_`
- Sorts features by importance (descending)
- Saves top-15 into:

- `outputs/feature_importance_top15.csv`

### 9. Console output / logging

On run, the script prints:

- Dataset size (rows, columns)
- Total number of missing values
- **Class balance** for the **phishing** target (positive class):
  - Number and proportion of phishing vs. legitimate
- List of **dropped columns** (e.g., text columns, `FILENAME`)
- Final **summary table** of metrics for:
  - Baseline (LogReg) and Random Forest
  - For each of the 3 threshold types

### 10. Error handling

The script checks and fails with clear messages if:

- The CSV file does not exist
- The file is not a CSV file
- The `label` column is missing or contains values other than 0 and 1
- No numeric feature columns are available after dropping text columns

### 11. Reproducibility

The experiment uses fixed random seeds:

- `train_test_split(..., random_state=42)`
- `RandomForestClassifier(..., random_state=42)`

Running the same command again with the same data will reproduce identical results (up to library version differences).

