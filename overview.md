# Credit Default Prediction Using the Home Credit Default Risk Dataset

## Problem Statement

Credit default occurs when a borrower fails to meet the legal obligations of a debt contract — typically by missing scheduled payments on a loan. Predicting which applicants are likely to default is one of the most consequential problems in consumer lending:

- **For lenders**: Accurate default prediction drives loan approval decisions, pricing (interest rates), and portfolio risk management. Mispricing default risk directly erodes profitability and can threaten institutional solvency.
- **For borrowers**: Overly conservative models deny credit to creditworthy individuals, disproportionately affecting underbanked populations. Overly permissive models lead to unsustainable debt burdens.
- **For regulators**: Default prediction models sit at the intersection of financial stability and consumer protection, making them subject to intense regulatory scrutiny.

This project builds an end-to-end credit default prediction system using the Home Credit Default Risk dataset, demonstrating technical depth in ML/AI, awareness of industry constraints, and systems thinking around productionizing models at scale.

## Dataset Overview

**Source**: [Home Credit Default Risk — Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)

Home Credit serves borrowers with little or no traditional credit history. The dataset captures a rich, multi-table view of each applicant:

| Table | Description | Approximate Size |
|---|---|---|
| `application_train/test` | Main application data (target variable `TARGET`: 1 = default) | ~307K train / ~49K test rows, 122 features |
| `bureau` | Prior credits from other financial institutions (reported to credit bureau) | ~1.7M rows |
| `bureau_balance` | Monthly balance snapshots for bureau credits | ~27M rows |
| `previous_application` | Prior Home Credit loan applications | ~1.7M rows |
| `POS_CASH_balance` | Monthly POS/cash loan snapshots | ~10M rows |
| `credit_card_balance` | Monthly credit card balance snapshots | ~3.8M rows |
| `installments_payments` | Repayment history for prior Home Credit loans | ~13.6M rows |

**Key characteristics**: High dimensionality, severe class imbalance (~8% default rate), mixed data types (numeric, categorical, temporal), and multi-table relational structure requiring aggregation-based feature engineering.

## End-to-End Analysis Pipeline

### 1. Data Ingestion & Storage
- Load raw CSV files; profile schema, types, and sizes.
- Store in a format suitable for iterative analysis (Parquet for columnar efficiency).

### 2. Exploratory Data Analysis (EDA)
- Target distribution and class imbalance quantification.
- Univariate and bivariate analysis of key features (income, credit amount, loan type, age, employment).
- Missing value patterns across tables.
- Correlation analysis and multicollinearity detection.

### 3. Feature Engineering
- **Aggregation features**: Summarize relational tables (bureau, previous applications, payment history) per applicant — e.g., mean/max/min of prior loan amounts, count of late payments, credit utilization ratios.
- **Temporal features**: Recency of last bureau inquiry, trend in payment behavior over time.
- **Domain features**: Debt-to-income ratio, loan-to-value proxies, credit history length.
- **Encoding**: Target encoding for high-cardinality categoricals; one-hot for low-cardinality.
- **Imputation strategy**: Justify choices (median, mode, indicator columns for missingness-as-signal).

### 4. Modeling
- **Baseline**: Logistic regression for interpretability and regulatory baseline.
- **Gradient boosting**: LightGBM / XGBoost as primary models — strong performance on tabular data with built-in handling of missing values and categoricals.
- **Ensemble**: Blending or stacking to combine model strengths.
- **Hyperparameter tuning**: Bayesian optimization (Optuna) with stratified k-fold cross-validation.

### 5. Evaluation
- **Primary metric**: AUC-ROC (discrimination ability across thresholds).
- **Secondary metrics**: AUC-PR (precision-recall, better for imbalanced data), KS statistic, Brier score (calibration).
- **Threshold analysis**: Optimize decision threshold based on business cost matrix (cost of false negative vs. false positive).
- **Calibration curves**: Ensure predicted probabilities reflect true default rates.

### 6. Explainability & Fairness
- **SHAP values**: Global and local feature importance; identify which factors drive individual predictions.
- **Partial dependence plots**: Understand marginal effect of key features.
- **Fairness audit**: Analyze model performance across protected groups (gender, age brackets) to check for disparate impact.
- **Adverse action reasons**: Demonstrate how top SHAP contributors can generate applicant-facing explanations (regulatory requirement under ECOA).

## Industry Context

### Basel III & Capital Requirements
Banks must hold capital proportional to the credit risk in their portfolios. Internal Ratings-Based (IRB) approaches under Basel III allow institutions to use internal models to estimate Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). Accurate PD models directly reduce required capital buffers, freeing capital for lending.

### Fair Lending & ECOA
The Equal Credit Opportunity Act (ECOA) and Regulation B prohibit discrimination in credit decisions. Models must be auditable for disparate impact across protected classes. The shift toward ML models in underwriting has intensified regulatory focus on explainability — black-box models that cannot provide adverse action reasons face legal risk.

### Model Risk Management (SR 11-7 / OCC 2011-12)
Federal guidance requires banks to maintain rigorous model risk management frameworks: independent model validation, ongoing performance monitoring, and documentation of model limitations. This project's pipeline mirrors production MRM expectations.

## Systems & Scalability Angle

### Spark Pipelines
- The multi-table aggregation step (especially `bureau_balance` at ~27M rows and `installments_payments` at ~13.6M rows) motivates distributed processing.
- Demonstrate PySpark for feature engineering at scale; compare runtime and memory footprint against pandas.

### Model Serving
- Package the trained model behind a REST API (FastAPI) for real-time scoring.
- Since we work with a static Kaggle dataset, the API serves as a **proof-of-concept**: the trained model is loaded once, and individual test-set rows are sent as JSON requests to simulate new loan applications arriving from a loan origination system. The endpoint returns a default probability and SHAP-based adverse action reasons.
- Discuss batch vs. real-time inference trade-offs in production lending systems.

### Drift Monitoring
- Define a monitoring strategy: track input feature distributions (PSI — Population Stability Index) and model output distributions over time.
- Alert on concept drift (changing relationship between features and default) vs. data drift (shifting input distributions).


