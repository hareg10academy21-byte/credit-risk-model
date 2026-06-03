import pandas as pd
import scorecardpy as sc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==================================================
# Load Data
# ==================================================
df = pd.read_csv("data/data.csv")

TARGET = "FraudResult"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ==================================================
# Feature Engineering Transformer
# ==================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # -------------------------------
        # Datetime Features
        # -------------------------------
        X["TransactionStartTime"] = pd.to_datetime(
            X["TransactionStartTime"]
        )

        X["TransactionHour"] = (
            X["TransactionStartTime"].dt.hour
        )

        X["TransactionDay"] = (
            X["TransactionStartTime"].dt.day
        )

        X["TransactionMonth"] = (
            X["TransactionStartTime"].dt.month
        )

        X["TransactionYear"] = (
            X["TransactionStartTime"].dt.year
        )

        # -------------------------------
        # Aggregate Features
        # -------------------------------
        agg = (
            X.groupby("CustomerId")
            .agg(
                TotalTransactionAmount=("Amount", "sum"),
                AverageTransactionAmount=("Amount", "mean"),
                TransactionCount=("Amount", "count"),
                StdTransactionAmount=("Amount", "std")
            )
            .reset_index()
        )

        X = X.merge(
            agg,
            on="CustomerId",
            how="left"
        )

        X["StdTransactionAmount"] = (
            X["StdTransactionAmount"]
            .fillna(0)
        )

        # -------------------------------
        # Drop Unnecessary Columns
        # -------------------------------
        X.drop(
            columns=[
                "TransactionStartTime",
                "TransactionId",
                "BatchId",
                "AccountId",
                "SubscriptionId",
                "CustomerId"
            ],
            errors="ignore",
            inplace=True
        )

        return X


# ==================================================
# Feature Engineering
# ==================================================
feature_engineer = FeatureEngineer()

X_features = feature_engineer.fit_transform(X)


# ==================================================
# Detect Column Types
# ==================================================
numeric_features = (
    X_features
    .select_dtypes(include=["int64", "float64"])
    .columns
    .tolist()
)

categorical_features = (
    X_features
    .select_dtypes(include=["object", "string"])
    .columns
    .tolist()
)


# ==================================================
# Numeric Pipeline
# ==================================================
numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ==================================================
# Categorical Pipeline
# ==================================================
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# ==================================================
# Preprocessor
# ==================================================
preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])


# ==================================================
# Full Pipeline
# ==================================================
pipeline = Pipeline([
    (
        "feature_engineering",
        feature_engineer
    ),
    (
        "preprocessing",
        preprocessor
    )
])


# ==================================================
# Run Pipeline
# ==================================================
X_processed = pipeline.fit_transform(X)

print("Pipeline executed successfully")
print("Processed Shape:", X_processed.shape)


# ==================================================
# Information Value
# ==================================================
print("\nCalculating WoE...")

woe_df_input = df.drop(
    columns=[
        "TransactionStartTime",
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId"
    ],
    errors="ignore"
)

bins = sc.woebin(
    woe_df_input,
    y=TARGET
)

woe_df = sc.woebin_ply(
    woe_df_input,
    bins
)

print(woe_df.head())
# ==================================================
# Display Columns
# ==================================================
print("\nDataset Columns:")
print(df.columns.tolist())