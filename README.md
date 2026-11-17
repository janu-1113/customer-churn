# Customer Churn Prediction

Predicting which telecom customers are likely to leave, using the Telco Customer Churn dataset.

## Overview

Churn is the number of customers who cancel or unsubscribe from a service. This project analyzes past customer data to understand the characteristics and behavior of customers before they churned, and prepares the data for building models that can flag customers who are likely to leave in the future.

## Dataset

- **Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Size:** 7,043 customers (rows) and 21 columns (features)
- **Target:** `Churn` (Yes / No)
- **Class balance:** about 26.5% churned, 73.5% retained

| Column | Description |
|---|---|
| `customerID` | Unique customer ID |
| `gender` | Male / Female |
| `SeniorCitizen` | Whether the customer is a senior citizen (1, 0) |
| `Partner` | Whether the customer has a partner (Yes, No) |
| `Dependents` | Whether the customer has dependents (Yes, No) |
| `tenure` | Number of months the customer has stayed with the company |
| `PhoneService` | Whether the customer has phone service (Yes, No) |
| `MultipleLines` | Yes, No, No phone service |
| `InternetService` | DSL, Fiber optic, No |
| `OnlineSecurity` | Yes, No, No internet service |
| `OnlineBackup` | Yes, No, No internet service |
| `DeviceProtection` | Yes, No, No internet service |
| `TechSupport` | Yes, No, No internet service |
| `StreamingTV` | Yes, No, No internet service |
| `StreamingMovies` | Yes, No, No internet service |
| `Contract` | Contract term (month-to-month, one year, two year) |
| `PaperlessBilling` | Whether the customer uses paperless billing |
| `PaymentMethod` | Customer's payment method |
| `MonthlyCharges` | Amount charged monthly |
| `TotalCharges` | Total amount charged |
| `Churn` | Whether the customer churned (target) |

## Project Structure

```
.
├── customer_churn_prediction.ipynb
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

Download the CSV from Kaggle and place it in the `data/` folder before running the notebook.

## Workflow

1. **Exploratory data analysis**
   - Inspect shape, columns, data types, and class balance
   - Convert `TotalCharges` to numeric (it loads as text)
   - Drop the 11 rows with missing `TotalCharges` (7,032 rows remain)
   - Visualize churn share, gender split, and tenure vs. churn (histograms, violin plot)
2. **Preprocessing and feature engineering**
   - Drop `customerID` (unique, not informative)
   - Encode categorical features with `LabelEncoder`
   - Pearson correlation analysis with a heatmap
   - Drop `TotalCharges` to reduce multicollinearity with `tenure` and `MonthlyCharges`
3. **Feature selection**
   - Stratified 80/20 train/test split (`random_state=0`)
   - Standardize features with `StandardScaler`
   - Rank features with Random Forest feature importance
   - Check redundancy with Recursive Feature Elimination (`RFECV` with L1 Logistic Regression)
4. **Modeling (planned)**
   - Random Forest
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbors
   - Gaussian Naive Bayes

## Key Findings

- Average tenure is about 32 months; average monthly charges are about 64.8 and average total charges about 2,283.
- Churn is concentrated in the first few months of service (roughly 0 to 8 months). Customers who stay past that point tend to become long-term, loyal customers.
- Gender is split almost evenly and shows no meaningful relationship with churn.
- `MonthlyCharges` and `tenure` are the most important features by a clear margin (about 0.24 and 0.23 importance), followed by `Contract`, `PaymentMethod`, `TechSupport`, and `OnlineSecurity`.
- `PhoneService` is the least important feature.
- `RFECV` selected 16 of 18 features, which is close to the full set, so all 18 features are kept for modeling.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook or JupyterLab

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Usage

```bash
git clone <your-repo-url>
cd <your-repo-folder>
jupyter notebook customer_churn_prediction.ipynb
```

Run the cells from top to bottom.

## Next Steps

- Train and compare the five models listed above
- Evaluate with accuracy, precision, recall, F1, and ROC-AUC (recall on the churn class matters most for retention use cases)
- Handle class imbalance (for example class weights or SMOTE)
- Tune hyperparameters for the best model

## Acknowledgements

Dataset provided by IBM and shared on Kaggle by [BlastChar](https://www.kaggle.com/blastchar/telco-customer-churn).
