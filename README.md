# Credit Card Fraud Detection – Final Report (Module 24)

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using supervised machine learning algorithms. The dataset is highly imbalanced and contains anonymized transaction data. We applied preprocessing, SMOTE, and multiple classification algorithms to identify fraud with high recall and precision.

---

## 📊 Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Records**: 284,807
- **Fraud Cases**: 492 (≈0.17%)
- **Features**: V1–V28 (PCA components), `Time`, `Amount`, and `Class` (target variable)

📥 **Note**: The dataset is too large to be stored in this repository. Please download it from Kaggle and place the `creditcard.csv` file in a folder named `dataset/`.

---

## 🧹 Preprocessing & Feature Engineering
- Scaled `Time` and `Amount` with `StandardScaler`
- Addressed class imbalance using SMOTE
- No missing values or categorical features were present

---

## 🤖 Models Trained
- Logistic Regression
- Random Forest
- XGBoost

Each model was trained and evaluated using precision, recall, F1-score, and accuracy.

---

## 🧪 Model Performance
| Model               | Accuracy | Precision (Fraud) | Recall (Fraud) | F1 Score (Fraud) |
|--------------------|----------|-------------------|----------------|------------------|
| Logistic Regression| 95%      | 0.97              | 0.92           | 0.95             |
| Random Forest      | 100%     | 1.00              | 1.00           | 1.00             |
| XGBoost            | 100%     | 1.00              | 1.00           | 1.00             |

---

## 📈 Visualizations Included
- Distribution of `Amount` and `Time`
- Class balance (Fraud vs Non-Fraud)
- Confusion Matrices
- Bar chart comparing model metrics

---

## 🗂️ Repository Structure
```
credit-card-fraud-detection-final/
├── Credit Card Fraud Detection.ipynb   # Full modeling pipeline
├── dataset/
│   └── creditcard.csv                  # (Download manually from Kaggle)
└── README.md                           # This file
```

---

## 🧠 Conclusion
Ensemble methods significantly outperformed the baseline logistic regression model. XGBoost and Random Forest provided perfect scores on the test set. This modeling approach can enhance real-time fraud detection systems.

---

**Author:** Bedirhan Ulas  
**Capstone – Module 24 – UC Berkeley ML Certificate**
