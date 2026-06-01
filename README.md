## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord requires financial institutions to measure, monitor, and manage credit risk using transparent and reliable methodologies. Since lending decisions directly affect a bank's capital requirements and regulatory compliance, models used for credit scoring must be interpretable and well documented.

An interpretable model allows risk analysts, auditors, regulators, and business stakeholders to understand how different variables influence credit decisions. Well-documented models also support model validation, regulatory reviews, and ongoing monitoring. In regulated financial environments, organizations must be able to explain why a customer was classified as high-risk or low-risk and demonstrate that the model is fair, consistent, and compliant with regulatory standards.

Therefore, Basel II encourages the use of models whose predictions can be explained, justified, and audited throughout their lifecycle.

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

In many real-world datasets, a direct default indicator may not be available. In such situations, a proxy variable is created to approximate credit risk. Examples include prolonged delinquency, missed payments, low account activity, or other behaviors believed to be associated with default.

A proxy variable enables model development when actual default outcomes are unavailable. However, this approach introduces several business risks:

* The proxy may not accurately represent true default behavior.
* Customers classified as risky by the proxy may actually be good borrowers.
* Some truly risky customers may not be captured by the proxy.
* Model performance may deteriorate when applied to real lending decisions.
* Incorrect risk estimates can lead to higher default losses or missed lending opportunities.

As a result, proxy-based models should be carefully validated and continuously monitored to ensure that the proxy remains a reasonable representation of actual credit risk.

### 3. What are the key trade-offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high-performance model (e.g., Gradient Boosting) in a regulated financial context?

There is often a trade-off between model interpretability and predictive performance.

#### Logistic Regression with Weight of Evidence (WoE)

Advantages:

* Easy to understand and explain.
* Coefficients have clear business meaning.
* Simpler to validate and audit.
* Widely accepted by regulators.
* Supports transparent decision-making.

Disadvantages:

* May not capture complex nonlinear relationships.
* Often produces lower predictive accuracy than advanced machine learning models.

#### Gradient Boosting Models

Advantages:

* Typically achieve higher predictive accuracy.
* Capture nonlinear patterns and feature interactions automatically.
* Often improve risk prediction and portfolio performance.

Disadvantages:

* More difficult to interpret.
* Harder to explain individual predictions.
* Require additional model governance and explainability techniques.
* May face greater regulatory scrutiny.

In regulated financial environments, institutions often balance these factors by using interpretable models for regulatory compliance and governance, while evaluating advanced machine learning models when higher predictive performance provides significant business value. Modern explainability techniques such as SHAP values can help improve the transparency of complex models, but they may still not be as straightforward as traditional scorecard-based approaches.
