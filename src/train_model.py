import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/processed/processed.csv")

print("Dataset shape:", df.shape)

# =========================
# TARGET
# =========================
target = "is_high_risk"

# =========================
# DROP HIGH CARDINALITY COLUMNS (IMPORTANT FIX)
# =========================
drop_cols = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "TransactionStartTime"
]

df = df.drop(columns=drop_cols, errors="ignore")

# =========================
# SPLIT
# =========================
X = df.drop(columns=[target])
y = df[target]

# =========================
# ENCODING (SAFE NOW)
# =========================
X = pd.get_dummies(X)
X = X.fillna(0)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

# =========================
# SAVE MODEL
# =========================
os.makedirs("src/models", exist_ok=True)

joblib.dump(model, "src/models/model.pkl")

print("Model saved at src/models/model.pkl")