import pandas as pd
import scorecardpy as sc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

# ==================================================
# Load Data
# ==================================================
df = pd.read_csv("data/data.csv")


# ==================================================
# RFM FEATURE ENGINEERING
# ==================================================

df["TransactionStartTime"] = pd.to_datetime(
    df["TransactionStartTime"]
)

snapshot_date = (
    df["TransactionStartTime"].max()
    + pd.Timedelta(days=1)
)

rfm = (
    df.groupby("CustomerId")
    .agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (
                snapshot_date - x.max()
            ).days
        ),
        Frequency=(
            "TransactionId",
            "count"
        ),
        Monetary=(
            "Amount",
            "sum"
        )
    )
    .reset_index()
)

rfm_features = rfm[
    ["Recency", "Frequency", "Monetary"]
]

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm_features
)


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    rfm_scaled
) 

cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

print(cluster_summary)


high_risk_cluster = rfm.groupby("Cluster")["Monetary"].mean().idxmin()

rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)
# print(df.columns.tolist())


df = df.merge(
    rfm[
        ["CustomerId", "is_high_risk"]
    ],
    on="CustomerId",
    how="left"
)  

TARGET = "is_high_risk"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nHigh Risk Distribution:")
print(df["is_high_risk"].value_counts())

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
iv_df = sc.iv(woe_df, y="is_high_risk")

print("\nInformation Value:")
print(iv_df.sort_values(by="info_value", ascending=False))
# ==================================================
# Display Columns
# ==================================================
print("\nDataset Columns:")
print(df.columns.tolist())