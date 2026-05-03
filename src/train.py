"""
Credit Card Fraud Detection — Classification Pipeline

Detects fraudulent transactions using Logistic Regression, Random Forest,
and XGBoost on the Kaggle Credit Card Fraud dataset (284,807 transactions,
0.17% fraud rate). SMOTE oversampling is applied inside the training fold
only to prevent data leakage.
"""
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "dataset" / "creditcard.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.3


def load_data(path: Path) -> pd.DataFrame:
    """Load the Kaggle credit card transactions dataset.

    Args:
        path: Path to creditcard.csv.

    Returns:
        Raw DataFrame with 31 columns (Time, V1–V28, Amount, Class).

    Raises:
        FileNotFoundError: If the dataset is not found.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    assert "Class" in df.columns, "Target column 'Class' missing."
    return df


def explore(df: pd.DataFrame) -> None:
    """Log key dataset statistics."""
    fraud_rate = df["Class"].mean()
    logger.info(f"Class distribution:\n{df['Class'].value_counts()}")
    logger.info(f"Fraud rate: {fraud_rate:.4%}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Scale Amount and Time; return feature matrix and target vector.

    SMOTE is NOT applied here — it is applied only to the training set
    after train/test split to avoid data leakage.

    Returns:
        X (scaled features array), y (target array).
    """
    df = df.copy()
    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    logger.info(f"Feature shape: {X.shape}")
    return X, y


def build_models() -> Dict[str, object]:
    """Return classifiers to compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }


def evaluate_model(
    name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """Apply SMOTE to training data, fit model, and return metrics.

    SMOTE is applied inside this function (training data only) so the
    test set always reflects the real class distribution.

    Args:
        name: Human-readable model name.
        model: Unfitted sklearn-compatible estimator.
        X_train, X_test, y_train, y_test: Split arrays.

    Returns:
        Dict with precision, recall, F1 (fraud class), ROC-AUC, train_time_s.
    """
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(
        f"[{name}] SMOTE: {y_train.sum():,} → {y_train_res.sum():,} fraud samples in train set"
    )

    t0 = time.perf_counter()
    model.fit(X_train_res, y_train_res)
    train_time = round(time.perf_counter() - t0, 2)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    return {
        "Model": name,
        "Precision (Fraud)": round(precision_score(y_test, y_pred), 4),
        "Recall (Fraud)": round(recall_score(y_test, y_pred), 4),
        "F1 (Fraud)": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob if y_prob is not None else y_pred), 4),
        "Train Time (s)": train_time,
    }


def plot_confusion_matrices(
    models: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path,
) -> None:
    """Save side-by-side confusion matrix plots for all models."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=["Non-Fraud", "Fraud"],
            ax=ax, colorbar=False,
        )
        ax.set_title(name, fontsize=10)
    plt.suptitle("Confusion Matrices — Credit Card Fraud Detection", fontsize=12)
    plt.tight_layout()
    out = save_dir / "confusion_matrices.png"
    fig.savefig(out, dpi=150)
    logger.info(f"Confusion matrices saved → {out}")
    plt.close(fig)


def plot_results(results_df: pd.DataFrame, save_dir: Path) -> None:
    """Save a grouped bar chart of fraud-class metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["Precision (Fraud)", "Recall (Fraud)", "F1 (Fraud)"]
    df_plot = results_df.set_index("Model")[metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
    ax.set_ylim(0.85, 1.01)
    ax.set_title("Fraud Detection Performance by Classifier", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = save_dir / "model_comparison.png"
    fig.savefig(out, dpi=150)
    logger.info(f"Comparison chart saved → {out}")
    plt.close(fig)


def main() -> None:
    logger.info("=== Credit Card Fraud Detection Pipeline ===")

    df = load_data(DATA_PATH)
    explore(df)

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"Fraud in test set: {y_test.sum():,} ({y_test.mean():.4%})")

    models = build_models()
    results = []

    for name, model in models.items():
        logger.info(f"Training: {name}...")
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        logger.info(
            f"  Precision: {metrics['Precision (Fraud)']} | "
            f"Recall: {metrics['Recall (Fraud)']} | "
            f"F1: {metrics['F1 (Fraud)']} | "
            f"ROC-AUC: {metrics['ROC-AUC']}"
        )

    results_df = pd.DataFrame(results)
    logger.info("\n=== Results ===\n" + results_df.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    plot_results(results_df, RESULTS_DIR)
    plot_confusion_matrices(models, X_test, y_test, RESULTS_DIR)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
