# Credit Risk Probability Model for Alternative Data

## Project Overview

This project aims to build an end-to-end credit risk scoring system for Bati Bank using transaction data from an eCommerce platform. The solution will generate customer risk probabilities, credit scores, and support buy-now-pay-later decisions.

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

Basel II requires financial institutions to measure, monitor, and manage credit risk using transparent and reliable methodologies. Because credit decisions directly affect customers and regulatory compliance, models must be interpretable and well documented. An interpretable model allows risk analysts, auditors, and regulators to understand how predictions are generated and whether they align with business and regulatory expectations.

Proper documentation ensures that model assumptions, data sources, feature engineering steps, validation procedures, and limitations are clearly recorded. This improves accountability, supports model governance, and enables continuous monitoring and auditing throughout the model lifecycle.

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The available transaction dataset does not contain a direct indicator showing whether a customer eventually defaulted on a loan. Since supervised machine learning requires a target variable, a proxy variable must be created to approximate customer risk behavior.

An RFM (Recency, Frequency, Monetary) based segmentation approach can be used to identify customers with behavioral patterns that may indicate higher or lower credit risk. These segments can then serve as proxy labels for model training.

However, proxy variables introduce several risks. The proxy may not perfectly represent actual default behavior, resulting in inaccurate predictions. Customers may be incorrectly classified as high-risk or low-risk, potentially causing lost revenue, increased credit losses, or unfair lending decisions. Therefore, the methodology used to create the proxy variable must be carefully justified and documented.

### 3. What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

Simple models such as Logistic Regression combined with Weight of Evidence (WoE) transformations are highly interpretable. They provide clear explanations for predictions, making them easier to validate, audit, and justify to regulators. Their transparency supports compliance and stakeholder trust.

More advanced models such as Gradient Boosting often achieve higher predictive accuracy by capturing complex non-linear relationships in the data. However, they are generally less interpretable and more difficult to explain. This can create challenges during regulatory reviews and model governance processes.

In a regulated financial environment, organizations must balance predictive performance with explainability. While high-performance models may improve risk prediction accuracy, interpretable models often provide stronger regulatory acceptance and easier operational monitoring.
