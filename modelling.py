# modeling.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(X, y, params=None):
    if params is None:
        params = {"n_estimators": 100, "max_depth": 5}
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model

def predict_model(model, X_test):
    return model.predict(X_test)
