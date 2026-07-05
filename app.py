from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="A production-ready FastAPI service that predicts customer churn using a fine-tuned XGBoost model.",
    version="1.0.0"
)

# Load pre-trained serialization artifacts (transformers and models)
loaded_bin_enc = joblib.load('models/telco_churn_encoder.pkl')
loaded_nom_enc = joblib.load('models/telco_churn_encoder2.pkl')
loaded_model = joblib.load('models/telco_churn_xgb_model.pkl')

# Debug prints to verify expected pipeline shapes on startup
print("--- BINARY ENCODER EXPECTED FEATURES ---", loaded_bin_enc.feature_names_in_)
print("--- NOMINAL ENCODER EXPECTED FEATURES ---", loaded_nom_enc.feature_names_in_)

# Pydantic schema for strict input data validation and type enforcement
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict", summary="Predict Customer Churn Probability")
def predict_churn(customer: CustomerData):
    # 1. Convert the incoming Pydantic object into a native Python dictionary
    data_dict = customer.dict()

    # 2. Wrap the dictionary in a list to construct a single-row Pandas DataFrame
    input_df = pd.DataFrame([data_dict])

    # 3. Feature Engineering Steps
    # Feature 1: Total Services Diversity Score (Count how many add-on services the customer has)
    services = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_df['Total_Services'] = (input_df[services] == 'Yes').sum(axis=1)

    # Feature 2: High-Risk Billing Behavioral Flag (Combines month-to-month contracts with electronic checks)
    input_df['is_high_risk_billing'] = (
                (input_df['Contract'] == 'Month-to-month') & (input_df['PaymentMethod'] == 'Electronic check')).astype(int)

    # Feature 3: Temporal Scaling (Convert contract tenure from months to years)
    input_df['tenure_year'] = input_df['tenure'] / 12

    # Define categorical grouping sets based on training phase splits
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    nominal_cols = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']

    # 4. Encoding Transformations
    # Process binary columns through the pre-trained One-Hot Encoder matrix
    bin_mapped = loaded_bin_enc.transform(input_df[binary_cols])
    bin_df = pd.DataFrame(bin_mapped, columns=loaded_bin_enc.get_feature_names_out())

    # Sklearn Single-Row Matrix Bug Bypass for Nominal Data:
    # Dynamically map incoming text values using the historical category index orders
    # to eliminate NumPy object array casting or single-row type coercion errors.
    for i, col in enumerate(nominal_cols):
        category_mapping = {val: idx for idx, val in enumerate(loaded_nom_enc.categories_[i])}
        input_df[col] = input_df[col].map(category_mapping)

    # 5. Matrix Concatenation
    # Combine the transformed binary features with the unified numerical/nominal input data side-by-side (axis=1)
    final_df = pd.concat([input_df, bin_df], axis=1)

    # 6. Strict Column Feature Alignment
    # Enforce the exact spatial sequence and feature naming convention required by the pre-trained XGBoost model
    model_columns = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'is_high_risk_billing', 'Total_Services',
        'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
        'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes', 'PaperlessBilling_Yes',
        'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod', 'tenure_year'
    ]
    final_features = final_df[model_columns]

    # 7. Inference Phase
    # Extract the prediction class (0 or 1) and calculate probability metrics for churn risk
    prediction = int(loaded_model.predict(final_features)[0])
    probability = float(loaded_model.predict_proba(final_features)[0][1])

    # 8. Return standardized JSON response object
    return {
        "churn_prediction": prediction,
        "probability": round(probability, 2)
    }

# To run the server locally, execute the following command in your terminal:
# uvicorn app:app --reload