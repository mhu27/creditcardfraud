import joblib
import numpy as np

model=joblib.load('model/fraud_model.pkl')

def predict_fraud(X_new, threshold=0.9999999999):
    proba = model.predict_proba(X_new)[:, 1]
    return (proba >= threshold).astype(int)

if __name__ == "__main__":
    test_sample = np.random.rand(1, 30)  # 形状必须匹配模型输入
    result = predict_fraud(test_sample)
    print("Fraud Prediction:", result)