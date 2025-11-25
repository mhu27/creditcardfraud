#  Credit Card Fraud Detection (MLE Core Project)

## Project Overview

This project builds a machine learning model to detect fraudulent credit card transactions. Since fraud cases are extremely rare compared to legitimate ones, the primary challenge lies in handling class imbalance and optimizing the decision threshold to maximize fraud detection performance.

The goal is to create a reproducible, scalable, and well-documented workflow as part of a transition into Machine Learning Engineering (MLE).

---

##  Project Structure
fraud-detection/
├── data/ # Raw dataset (.csv)
├── notebook/ # Exploratory analysis, training and evaluation (Jupyter notebooks)
├── model/ # Saved model pipeline (.pkl)
├── src/ # Custom prediction logic (predict.py)
├── README.md # Project documentation

## 🛠 Model Summary

- **Model**: Logistic Regression with `class_weight='balanced'`
- **Preprocessing**: StandardScaler for feature normalization
- **Tooling**: `Pipeline` to unify preprocessing and model
- **Train/Test split**: Stratified sampling based on label (`Class`)
- **Custom Threshold**: `y_score >= 0.9999999999` selected to balance precision and recall

---

##  Evaluation Metrics (after threshold tuning)

| Metric     | Class 0 (Non-Fraud) | Class 1 (Fraud) |
|------------|---------------------|------------------|
| Precision  | 1.00                | 0.85             |
| Recall     | 1.00                | 0.76             |
| F1-score   | 1.00                | 0.80             |
| ROC AUC    | \> 0.97             | (Excellent separation ability) |

> With the new threshold, the model catches ~76% of fraud cases with high precision (85%), while maintaining nearly perfect performance on non-fraud cases.

---

##  Key Takeaways

- AUC and ROC analysis helped diagnose true model potential despite class imbalance.
- Custom thresholding allowed a major improvement in fraud class precision (from 0.06 → 0.85).
- Built using reproducible components (Pipeline, joblib, sklearn APIs), ready for production wrapping.

---

##  Next Steps

- [ ] Compare against Random Forest or XGBoost baselines
- [ ] Visualize Precision-Recall curve to support threshold selection
- [ ] Save PR / ROC plots for presentation
- [ ] Build and expose a minimal FastAPI service for inference
- [ ] Dockerize and test locally

---

##  Model Inference (Quick Usage)

```python
import joblib
model = joblib.load('model/fraud_model.pkl')
y_proba = model.predict_proba(X_new)[:, 1]
y_pred = (y_proba >= 0.99).astype(int)