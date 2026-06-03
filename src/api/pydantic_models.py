from pydantic import BaseModel


class CustomerRequest(BaseModel):
    CustomerId: int
    Amount: float
    TransactionId: int


class PredictionResponse(BaseModel):
    customer_id: int
    risk_probability: float
    risk_label: str