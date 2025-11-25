Credit Card Fraud Detection — MLE Core Project

A complete machine learning engineering workflow for detecting fraudulent credit card transactions.
The project focuses on handling extreme class imbalance, comparing multiple models, and building production-ready ML components (pipelines, models, inference scripts).

Project Structure
fraud-detection/
├── data/                 # Raw dataset (creditcard.csv)
├── notebook/             # EDA, model training, evaluation
│   ├── eda.ipynb
│   ├── main.ipynb                      # Baseline logistic regression
│   ├── model_logistic_regularized.ipynb
│   ├── model_xgboost.ipynb
│   └── model_compare.ipynb
├── models/               # Saved models (.pkl)
│   ├── logistic_baseline.pkl
│   ├── logreg_l1.pkl
│   ├── logreg_l2.pkl
│   └── xgb_model.pkl
├── results/              # Metrics comparison, ROC/PR curves, result CSVs
├── src/                  # Deployable code (predict.py, utils)
└── README.md

Project Goal

Credit card fraud detection is a highly imbalanced binary classification task
(fraud cases < 0.2% of all transactions).

This project aims to:

Build robust machine learning models for rare-event detection

Evaluate linear vs. tree-based models

Handle class imbalance (class weights, scale_pos_weight)

Produce clean, reproducible, MLE-style artifacts

Prepare for deployment (FastAPI, Docker)

Models Implemented
1. Logistic Regression (Baseline Pipeline)

Notebook: main.ipynb

Pipeline(StandardScaler → LogisticRegression)

Uses class_weight='balanced'

Serves as a simple interpretable baseline

Baseline Performance

(Using default decision threshold 0.5)

Metric	Fraud Class (1)
Precision	~0.06
Recall	~0.03
F1	~0.04
ROC AUC	~0.97

➡ Although AUC looks high, precision/recall are extremely poor due to class imbalance.

2. Regularized Logistic Regression (L1 & L2)

Notebook: model_logistic_regularized.ipynb

Explicit regularization significantly improves performance:



3. XGBoost (Tree-based Gradient Boosting)

Notebook: model_xgboost.ipynb

Why XGBoost?

Handles non-linear fraud patterns

Works extremely well on imbalanced datasets

Built-in parameter: scale_pos_weight = (#neg / #pos)

Best performance across all metrics

Saved model: models/xgb_model.pkl

📈 Overall Model Comparison

Generated in: model_compare.ipynb

Model	AUC	PR-AUC	Recall	Precision	F1
Logistic-Baseline	0.97	0.08	0.03	0.06	0.04
Logistic-L2	…	…	…	…	…
Logistic-L1	…	…	…	…	…
XGBoost	…	…	…	…	…

(CSV saved as results/model_results_all.csv)

Key Insight:
👉 PR-AUC and recall matter more than ROC-AUC in fraud detection.

📝 What We Learned
✓ ROC-AUC alone is misleading for imbalanced problems

A model can achieve 0.97 AUC and still miss 97% of fraud cases.

✓ Regularization (L1/L2) improves logistic models

Stronger signal extraction → better fraud recall.

✓ XGBoost is the strongest performer

Captures complex fraud patterns
Handles imbalance effectively
Best PR-AUC & recall combination