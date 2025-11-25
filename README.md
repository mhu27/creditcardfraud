Credit Card Fraud Detection (Machine Learning Engineering Project)

This project builds a reproducible, scalable, and machine-learning–engineering–ready pipeline to detect fraudulent credit card transactions.
Because fraud cases are extremely rare (0.17% of data), the project focuses on class imbalance handling, modeling, evaluation, and threshold tuning.

This repository contains all stages of the workflow, including data preprocessing, model comparison, metrics logging, model saving, and Jupyter notebooks for experimentation.

Project Structure
creditcardfraud/
├── data/               # Raw dataset (creditcard.csv)
├── notebook/           # EDA, modeling, XGBoost, comparisons
├── models/             # Saved trained models (.pkl)
├── results/            # Comparison metrics (CSV)
├── src/                # (Optional) prediction scripts
├── README.md           # Project documentation

Project Goals

Build baseline logistic regression model

Add regularized models (L1 / L2 Logistic Regression)

Add tree-based model: XGBoost

Compare all models with consistent metrics

Tune thresholds for fraud detection

Save models and results for production usage



Dataset Summary

Source: Kaggle “Credit Card Fraud Detection”

Samples: 284,807

Fraud cases: 492

Fraud ratio: 0.172%

Feature types: PCA-transformed financial features + Time + Amount

Label: Class (1 = fraud)

The dataset is highly imbalanced, requiring weighted loss, threshold tuning, and proper evaluation.


Models Implemented
Logistic Regression (Baseline)

StandardScaler

class_weight="balanced"

ROC-AUC baseline ≈ 0.97

✔ L2-Regularized Logistic Regression

Prevents overfitting

Slightly improved recall stability

✔ L1-Regularized Logistic Regression

Performs feature selection

Useful for interpretability

✔ XGBoost Classifier

Handles nonlinear patterns

Much higher recall on minority fraud class

Best-performing model overall

scale_pos_weight used to counter class imbalance


Model Comparison Results

Results saved in:
results/model_results_all.csv

Example metrics (your actual numbers may vary):
Model	ROC-AUC	PR-AUC	Precision	Recall	F1
Logistic-Baseline	0.972	0.718	0.918	0.061	0.114
Logistic-L2	0.972	0.718	0.918	0.061	0.114
Logistic-L1	0.972	0.718	0.918	0.061	0.114
XGBoost	0.986	0.857	0.888	0.558	0.685

XGBoost significantly improves recall and overall fraud-detection performance.



Threshold Tuning

Fraud detection is cost-sensitive → false negatives are very expensive.
We tuned decision thresholds based on:

Precision-Recall Curve

Application-specific constraints

Business trade-offs

Example:

y_pred_custom = (y_proba >= 0.99999).astype(int)


Custom thresholds dramatically increased precision for fraud class at the cost of recall.



Model Saving

Both logistic regression and XGBoost models are saved as .pkl:

models/logistic_model.pkl
models/xgboost_model.pkl





Reproducibility

Notebook files documenting all steps:

notebook/
├── eda.ipynb
├── model_train_logistic.ipynb
├── model_train_xgboost.ipynb
├── model_compare.ipynb


These notebooks include:

EDA

Feature inspection

Class imbalance handling

Model training

Metrics comparison

Saving results