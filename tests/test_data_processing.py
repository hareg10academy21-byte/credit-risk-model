import pandas as pd

df = pd.read_csv("data/processed_data.csv")


def test_basic():
    assert 1 + 1 == 2


def test_columns_exist():
    required_cols = ["CustomerId", "Amount", "TransactionId"]

    for col in required_cols:
        assert col in df.columns


def test_target_column_exists():
    assert "is_high_risk" in df.columns