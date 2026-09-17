# Taiwan Credit Card Default Prediction

## Project Overview

This project predicts whether a credit card customer is likely to default on their payment using machine learning.

The project uses the Taiwan Credit Card Default dataset and compares machine learning approaches before selecting a tuned Random Forest model for the final prediction system.

## Final Model

The final model is a tuned Random Forest Classifier optimized using GridSearchCV with ROC-AUC as the evaluation metric.

### Model Performance

- Accuracy: 78.72%
- Precision: 51.72%
- Recall: 56.52%
- F1-Score: 54.02%
- ROC-AUC: 77.38%

## Features

The model uses 23 customer-related financial and demographic features, including:

- Credit limit
- Sex
- Education
- Marriage status
- Age
- Payment history
- Bill amounts
- Previous payment amounts

## Deployment

The trained Random Forest model is saved using Joblib and integrated into a Streamlit application.

The application allows users to enter customer information and receive:

- Default prediction
- Default probability

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Project Structure

```text
Taiwan-Credit-Card-Default/
│
├── streamlit_app.py
├── app.py
├── credit_default_model.joblib
├── requirements.txt
└── README.md