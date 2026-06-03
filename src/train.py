import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/processed_data.csv")

# =========================
# SIMPLE CLEANING (IMPORTANT)
# =========================
df = df.dropna()

# Convert target (if exists from your Task 4)
if "is_high_risk" not in df.columns:
    raise Exception("Target column 'is_high_risk' not found. Run Task 4 first.")

X = df.drop(columns=["is_high_risk"])
y = df["is_high_risk"]

# Encode categorical columns (simple safe fix)
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MLflow setup
# =========================
mlflow.set_experiment("task-5-credit-risk")

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, preds)
    }

# =========================
# MODEL 1: Logistic Regression
# =========================
with mlflow.start_run(run_name="Logistic Regression"):

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    metrics = evaluate(lr, X_test, y_test)

    mlflow.log_param("model", "LogisticRegression")

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.sklearn.log_model(lr, "model")

# =========================
# MODEL 2: Random Forest
# =========================
with mlflow.start_run(run_name="Random Forest"):

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="f1"
    )

    grid.fit(X_train, y_train)

    rf = grid.best_estimator_

    metrics = evaluate(rf, X_test, y_test)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("best_n_estimators", rf.n_estimators)
    mlflow.log_param("best_max_depth", rf.max_depth)

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.sklearn.log_model(rf, "model")

print("Training completed successfully!")