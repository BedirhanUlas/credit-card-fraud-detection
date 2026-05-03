# Credit Card Fraud Detection

End-to-end machine learning system for detecting fraudulent credit card transactions. Addresses the extreme class imbalance challenge (0.17% fraud rate) using SMOTE oversampling and compares Logistic Regression, Random Forest, and XGBoost models.

## Business Problem

Credit card fraud costs the global financial industry over $30 billion annually. A high-recall fraud detection model directly reduces financial losses — missing a fraud case (false negative) is far more costly than a false alarm. This project optimizes for **recall on the fraud class** while maintaining high precision.

## Dataset

| Attribute | Value |
|---|---|
| Source | Kaggle — European cardholders (September 2013) |
| Transactions | 284,807 |
| Fraudulent | 492 (0.17%) |
| Features | 28 PCA-transformed features + Amount + Time |
| Challenge | Severe class imbalance |

## Results

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 97.4% | 0.06 | 0.92 | 0.11 | 0.97 |
| Random Forest | **100%** | **1.00** | **1.00** | **1.00** | **1.00** |
| XGBoost | **100%** | **1.00** | **1.00** | **1.00** | **1.00** |

Random Forest and XGBoost achieved perfect classification after SMOTE balancing, demonstrating the power of ensemble methods on this problem.

## Methodology

### Class Imbalance — SMOTE
The dataset has a 577:1 class ratio. SMOTE (Synthetic Minority Oversampling Technique) generates synthetic fraud samples in feature space to create a balanced training set, preventing the model from simply predicting "not fraud" every time.

### Model Pipeline
```
Raw Data → EDA → StandardScaler → Train/Test Split
       → SMOTE (train only) → Model Training → Evaluation
```

### Evaluation Strategy
- Prioritize **Recall** over Accuracy (minimize missed fraud)
- Use **ROC-AUC** and **Precision-Recall curve** for imbalanced evaluation
- Confusion matrix analysis per model

## Key Findings

1. **Logistic Regression** provides a strong baseline (92% fraud recall) and is interpretable
2. **Random Forest & XGBoost** both achieve perfect scores — the PCA features provide clean separability after SMOTE balancing
3. **SMOTE is critical** — without it, models achieve 99.8% accuracy by predicting everything as legitimate
4. Feature importance analysis (Random Forest): V17, V14, V12, V10 are the top fraud predictors

## Project Structure

```
credit-card-fraud-detection/
├── Credit Card Fraud Detection.ipynb   # Full modeling notebook
└── Capstone Project_Final Report.pdf   # Detailed technical report
```

## Quick Start

```bash
git clone https://github.com/BedirhanUlas/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Download dataset from Kaggle
# kaggle datasets download -d mlg-ulb/creditcardfraud

pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyter
jupyter notebook "Credit Card Fraud Detection.ipynb"
```

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Related

- **Part 1 — EDA & Baseline:** [credit-card-fraud-detection-eda](https://github.com/BedirhanUlas/credit-card-fraud-detection-eda) — Exploratory analysis and logistic regression baseline

## Future Work

- Deploy best model as a real-time scoring API (FastAPI + Redis caching)
- Implement streaming fraud detection with Apache Kafka
- Add model drift monitoring with evidently.ai
- Threshold tuning for cost-sensitive classification

## License

MIT
