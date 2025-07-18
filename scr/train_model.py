from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train(X_train, y_train):
    model = Pipeline([
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model