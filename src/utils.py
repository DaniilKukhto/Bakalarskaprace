from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class DataLoadingError(Exception):
    """Custom exception for data loading related errors."""


@dataclass
class DatasetInfo:
    n_rows: int
    n_cols: int
    n_missing: int
    class_counts: Dict[str, int]
    class_proportions: Dict[str, float]
    dropped_columns: List[str]
    feature_names: List[str]


def load_dataset_from_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, DatasetInfo]:
    """
    Load PhiUSIIL dataset from a CSV file.
    
    Construct y_phish = 1 - label and return numeric feature matrix and target.
    """
    csv_file_path = Path(csv_path)
    if not csv_file_path.exists():
        raise DataLoadingError(f"CSV file not found: {csv_file_path}")
    
    if not csv_file_path.suffix.lower() == ".csv":
        raise DataLoadingError(f"File is not a CSV: {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        raise DataLoadingError(f"Failed to read CSV file: {csv_file_path}") from e

    if "label" not in df.columns:
        raise DataLoadingError("Column 'label' not found in CSV.")

    unique_labels = sorted(df["label"].dropna().unique().tolist())
    if not set(unique_labels).issubset({0, 1}):
        raise DataLoadingError(f"Unexpected values in 'label' column: {unique_labels}. Expected only 0/1.")

    # Define target: phishing = 1, legitimate = 0
    y_phish = 1 - df["label"]

    # Determine columns to drop: non-numeric + explicit text identifiers
    dropped_columns: List[str] = []
    candidate_drop = {"FILENAME", "URL", "Domain", "TLD", "Title"}

    for col in df.columns:
        if col == "label":
            continue
        if col in candidate_drop:
            dropped_columns.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            dropped_columns.append(col)

    df_numeric = df.drop(columns=dropped_columns + ["label"], errors="ignore")

    if df_numeric.shape[1] == 0:
        raise DataLoadingError(
            "No numeric feature columns available after dropping text columns. "
            "Check the dataset schema."
        )

    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())

    # Class distribution for phishing target
    class_counts = {
        "phishing_1": int((y_phish == 1).sum()),
        "legitimate_0": int((y_phish == 0).sum()),
    }
    total = len(y_phish)
    class_proportions = {
        "phishing_1": class_counts["phishing_1"] / total,
        "legitimate_0": class_counts["legitimate_0"] / total,
    }

    info = DatasetInfo(
        n_rows=n_rows,
        n_cols=n_cols,
        n_missing=n_missing,
        class_counts=class_counts,
        class_proportions=class_proportions,
        dropped_columns=dropped_columns,
        feature_names=list(df_numeric.columns),
    )

    return df_numeric, y_phish, info


@dataclass
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """Compute metrics at a specific probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return ThresholdMetrics(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> ThresholdMetrics:
    """
    Find threshold that maximizes F1-score over a grid.
    Returns ThresholdMetrics for the best threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    best_metrics: Optional[ThresholdMetrics] = None

    for t in thresholds:
        m = compute_metrics_at_threshold(y_true, y_proba, t)
        if best_metrics is None or m.f1 > best_metrics.f1:
            best_metrics = m

    if best_metrics is None:
        raise ValueError("Unable to compute best F1 threshold.")

    return best_metrics


def find_high_recall_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    target_recall: float = 1.0,
    atol: float = 1e-8,
) -> Optional[ThresholdMetrics]:
    """
    Find minimal threshold where recall == target_recall (within tolerance).
    Returns ThresholdMetrics or None if unreachable.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    for t in thresholds:
        m = compute_metrics_at_threshold(y_true, y_proba, t)
        if abs(m.recall - target_recall) <= atol:
            return m

    return None

