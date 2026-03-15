import argparse
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import (
    load_dataset_from_csv,
    compute_metrics_at_threshold,
    find_best_f1_threshold,
    find_high_recall_threshold,
    DataLoadingError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini-experiment for AI anti-phishing using PhiUSIIL dataset.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file (PhiUSIIL_Phishing_URL_Dataset.csv)",
    )
    return parser.parse_args()


def ensure_outputs_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def build_models(feature_names: List[str]):
    numeric_features = feature_names

    numeric_transformer_logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_transformer_rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor_logreg = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer_logreg, numeric_features),
        ]
    )

    preprocessor_rf = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer_rf, numeric_features),
        ]
    )

    logreg_clf = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=200,
    )

    rf_clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    logreg_model = Pipeline(
        steps=[
            ("preprocess", preprocessor_logreg),
            ("clf", logreg_clf),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocessor_rf),
            ("clf", rf_clf),
        ]
    )

    return logreg_model, rf_model


def main() -> int:
    args = parse_args()

    try:
        X, y, info = load_dataset_from_csv(args.csv_path)
    except DataLoadingError as e:
        print(f"[ERROR] Data loading failed: {e}", file=sys.stderr)
        return 1

    # Logging dataset info
    print("=== Dataset Info ===")
    print(f"Rows: {info.n_rows}, Columns: {info.n_cols}")
    print(f"Total missing values: {info.n_missing}")
    print("Class balance (phishing target y_phish):")
    print(
        f"  Phishing (1): {info.class_counts['phishing_1']} "
        f"({info.class_proportions['phishing_1']:.3f})"
    )
    print(
        f"  Legitimate (0): {info.class_counts['legitimate_0']} "
        f"({info.class_proportions['legitimate_0']:.3f})"
    )
    print("Dropped columns:")
    if info.dropped_columns:
        for col in info.dropped_columns:
            print(f"  - {col}")
    else:
        print("  (none)")
    print("====================\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    logreg_model, rf_model = build_models(info.feature_names)

    print("Training Logistic Regression (baseline)...")
    logreg_model.fit(X_train, y_train)
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    print("Training completed.\n")

    # Probabilities on test set
    y_proba_logreg = logreg_model.predict_proba(X_test)[:, 1]
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    thresholds_grid = np.arange(0.01, 1.00, 0.01)

    # Metrics container
    results_rows: List[Dict] = []

    def add_result_row(
        model_name: str,
        threshold_type: str,
        metrics,
    ):
        results_rows.append(
            {
                "model": model_name,
                "threshold_type": threshold_type,
                "threshold": metrics.threshold,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "roc_auc": metrics.roc_auc,
                "tn": metrics.tn,
                "fp": metrics.fp,
                "fn": metrics.fn,
                "tp": metrics.tp,
            }
        )

    # 1) Default threshold = 0.50
    print("Evaluating models at threshold = 0.50 ...")
    default_threshold = 0.50
    m_logreg_default = compute_metrics_at_threshold(y_test.values, y_proba_logreg, default_threshold)
    m_rf_default = compute_metrics_at_threshold(y_test.values, y_proba_rf, default_threshold)

    add_result_row("LogisticRegression", "fixed_0.5", m_logreg_default)
    add_result_row("RandomForest", "fixed_0.5", m_rf_default)

    # 2) Best F1 threshold
    print("Searching for best F1 thresholds...")
    m_logreg_best_f1 = find_best_f1_threshold(y_test.values, y_proba_logreg, thresholds_grid)
    m_rf_best_f1 = find_best_f1_threshold(y_test.values, y_proba_rf, thresholds_grid)

    add_result_row("LogisticRegression", "best_f1", m_logreg_best_f1)
    add_result_row("RandomForest", "best_f1", m_rf_best_f1)

    # 3) High recall threshold (recall == 1.0 if achievable)
    print("Searching for high-recall thresholds (recall == 1.0)...")
    m_logreg_high_recall = find_high_recall_threshold(
        y_test.values,
        y_proba_logreg,
        thresholds_grid,
        target_recall=1.0,
    )
    m_rf_high_recall = find_high_recall_threshold(
        y_test.values,
        y_proba_rf,
        thresholds_grid,
        target_recall=1.0,
    )

    if m_logreg_high_recall is not None:
        add_result_row("LogisticRegression", "high_recall_recall1.0", m_logreg_high_recall)
        print(
            f"LogisticRegression high-recall threshold: {m_logreg_high_recall.threshold:.2f}, "
            f"FP={m_logreg_high_recall.fp}, FN={m_logreg_high_recall.fn}"
        )
    else:
        print("LogisticRegression: recall==1.0 not achievable on grid 0.01..0.99.")

    if m_rf_high_recall is not None:
        add_result_row("RandomForest", "high_recall_recall1.0", m_rf_high_recall)
        print(
            f"RandomForest high-recall threshold: {m_rf_high_recall.threshold:.2f}, "
            f"FP={m_rf_high_recall.fp}, FN={m_rf_high_recall.fn}"
        )
    else:
        print("RandomForest: recall==1.0 not achievable on grid 0.01..0.99.")

    # Save metrics to CSV
    outputs_dir = ensure_outputs_dir()
    results_df = pd.DataFrame(results_rows)
    results_csv_path = outputs_dir / "results_metrics.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved metrics to: {results_csv_path}")

    # Feature importance for RandomForest (top 15)
    rf_clf = rf_model.named_steps["clf"]
    importances = rf_clf.feature_importances_
    feature_importance_df = pd.DataFrame(
        {
            "feature": info.feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    top15 = feature_importance_df.head(15)
    fi_csv_path = outputs_dir / "feature_importance_top15.csv"
    top15.to_csv(fi_csv_path, index=False)
    print(f"Saved RandomForest top-15 feature importances to: {fi_csv_path}")

    # Pretty-print summary table
    print("\n=== Summary of results (test set) ===")
    summary_df = results_df.copy()
    summary_df["precision"] = summary_df["precision"].round(3)
    summary_df["recall"] = summary_df["recall"].round(3)
    summary_df["f1"] = summary_df["f1"].round(3)
    summary_df["roc_auc"] = summary_df["roc_auc"].round(3)
    summary_df["threshold"] = summary_df["threshold"].round(3)

    # Order columns for display
    display_cols = [
        "model",
        "threshold_type",
        "threshold",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    print(
        tabulate(
            summary_df[display_cols],
            headers="keys",
            tablefmt="github",
            showindex=False,
        )
    )
    print("Note: FN = missed attacks, FP = extra alerts.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

