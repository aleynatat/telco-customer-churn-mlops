# Telco Customer Churn Prediction & MLOps Pipeline

An end-to-end Machine Learning and MLOps project that predicts telecommunication customer churn using an optimized XGBoost model and serves it via a production-ready FastAPI web service.

## 🚀 Project Overview
Customer churn is a critical metric for telecom companies. This project shifts the paradigm from a purely theoretical model inside a Jupyter Notebook to a live, production-grade microservice (API) capable of transforming raw customer inputs into real-time risk probabilities in milliseconds.

## 🛠️ Tech Stack & Skills Demonstrated
- **Data & Analytics:** Pandas, NumPy, Scikit-Learn
- **Modeling:** XGBoost, Hyperparameter Tuning (`GridSearchCV`, `RandomizedSearchCV`)
- **Backend & MLOps:** FastAPI, Pydantic, Uvicorn, Joblib
- **Software Engineering:** Vectorization over loops, Single-row matrix bug-bypassing, English code documentation

## 🧬 Data-Centric Feature Engineering
To maximize the predictive power of the model, three advanced features were engineered directly into the pipeline:
1. **Total Services Diversity (`Total_Services`):** Quantifies customer dependency by checking across 7 add-on services simultaneously using vectorized matrix summation instead of slow Python loops.
2. **High-Risk Billing Flag (`is_high_risk_billing`):** An behavioral indicator that captures the deadly combination of `Month-to-month` contracts and `Electronic check` payment methods.
3. **Temporal Scaling (`tenure_year`):** Converts customer contract lifetimes from months into normalized yearly distributions.

## 📊 Model Optimization & Performance
Initially, the base model suffered from a low customer capture rate (**Recall: 41%**). By realigning the optimization goal towards the minority class using `scale_pos_weight: 2` and fine-tuning via `GridSearchCV`, the pipeline achieved balanced, production-ready metrics:

- **Accuracy:** 78%
- **Churn Recall (Capture Rate):** **74%** (Drastically reducing missed high-risk customers)
- **F1-Score:** 0.64

## ⚙️ Production Pipeline & Single-Row Inference Fix
During the deployment phase using FastAPI, we encountered and bypassed a critical industry bug related to Scikit-Learn's `OrdinalEncoder`. When a single-row of mixed datatypes (strings & floats) enters an API request, NumPy coerces the matrix into an `object` array, causing `np.isnan` validation to fail inside Sklearn. 

**Solution:** We engineered a custom dictionary-mapping bypass utilizing the pre-trained encoder's internal categories (`categories_`), securing type safety without altering input schemas.

---

## 🏃‍♂️ How to Run Locally

### 1. Clone the repository and navigate into it:
```bash
git clone [https://github.com/YOUR_USERNAME/telco-customer-churn-mlops.git](https://github.com/YOUR_USERNAME/telco-customer-churn-mlops.git)
cd telco-customer-churn-mlops
