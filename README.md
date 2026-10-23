
# Customer Churn Prediction - End-to-End MLOps

An end-to-end machine learning project that predicts customer churn using Python, Scikit-learn, MLflow, FastAPI, Docker, and GitHub Actions.

This project demonstrates the complete ML engineering workflow: data preprocessing, exploratory analysis, model training, experiment tracking, API development, containerization, and automated testing.

## Business Problem

Customer churn occurs when customers stop using a company's services.

Predicting churn can help businesses identify customers who may leave and prioritize retention strategies.

This project uses the IBM Telco Customer Churn dataset to estimate each customer's probability of churning.

## Technology Stack

- **Programming:** Python 3.12
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Experiment Tracking:** MLflow
- **API:** FastAPI, Pydantic, Uvicorn
- **Containerization:** Docker
- **Testing:** Pytest
- **CI:** GitHub Actions
- **Version Control:** Git and GitHub

## Machine Learning Workflow

1. Load and clean the Telco customer churn dataset.
2. Convert TotalCharges to numeric values.
3. Remove duplicate customer records and identifiers.
4. Split data into stratified training and test sets.
5. Handle missing values using Scikit-learn imputers.
6. Standardize numeric features and one-hot encode categorical features.
7. Train Logistic Regression and Random Forest models.
8. Evaluate performance on held-out test data.
9. Perform five-fold cross-validation on training data.
10. Track experiments and model metrics with MLflow.
11. Serve predictions using FastAPI.
12. Package and test the application with Docker.

Preprocessing and classification are combined into Scikit-learn pipelines to reduce data leakage.

## Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains 7,043 customer records, with approximately 26.5% of customers labeled as churned.

The model uses 19 input features, including tenure, contract type, monthly charges, internet service, payment method, and customer demographics.

## Model Evaluation

Performance on the held-out test set:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8055 | 0.8041 |
| Precision | 0.6572 | 0.6667 |
| Recall | 0.5588 | 0.5241 |
| F1 Score | 0.6040 | 0.5868 |
| ROC-AUC | 0.8419 | 0.8431 |

**Selected deployment model: Logistic Regression**

Logistic Regression was selected because it achieved higher recall and F1 score while producing a smaller model artifact.

The model is a portfolio demonstration and has not been validated for production business decisions.

## MLflow Experiment Tracking

MLflow is used to log model parameters, evaluation metrics, and trained pipelines.

The project includes experiment runs for Logistic Regression and Random Forest.

MLflow allows comparisons between experiments and supports loading logged models for inference.

## FastAPI Prediction Service

The prediction API provides three endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Application information |
| `/health` | GET | Application and model readiness |
| `/predict` | POST | Predict customer churn |

The `/predict` endpoint accepts customer information as JSON and returns the predicted label and churn probability.

Example response:

```json
{
  "prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.6118
}
```

The example probability is from a sample customer, not an average model performance metric.

FastAPI automatically generates interactive API documentation at `/docs`.

## Run Locally

Create a Python 3.12 virtual environment and install the project dependencies.

```bash
python -m venv .venv
```

Activate the virtual environment and install dependencies.

```bash
pip install -r requirements.txt
```

Train the selected deployment model:

```bash
python -m src.train_deployment
```

Start the prediction API:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open the API documentation:

http://127.0.0.1:8000/docs

## Run with Docker

The deployment Dockerfile trains and packages Logistic Regression during the image build.

Build the deployment image:

```bash
docker build -f Dockerfile.deploy -t customer-churn-api:v2 .
```

Run the container:

```bash
docker run --rm -d --name churn-api-v2 -p 8002:8000 customer-churn-api:v2
```

Open:

http://127.0.0.1:8002/docs

Check model readiness:

```bash
curl http://127.0.0.1:8002/health
```

The deployment image contains its trained model and does not require a model volume mount.

## Automated Testing and CI

Automated tests cover:

- Data preprocessing
- Machine learning pipelines
- MLflow configuration and tracking logic
- FastAPI health checks
- Prediction response structure
- Request validation
- Invalid and missing customer features

The project currently has **20 passing automated tests**.

Run tests locally:

```bash
python -m pytest -v
```

GitHub Actions automatically runs the test suite on pushes and pull requests to the main branch.

## Reproducibility

The training pipeline uses:

- Stratified 80/20 train/test split
- Random state of 42
- Preprocessing fitted on training data
- Explicit model hyperparameters
- Pinned deployment dependency versions
- A deployment Dockerfile that trains the model during the build

## Current Deployment Status

- Local FastAPI application: verified
- Docker v1 with mounted model: verified
- Docker v2 with packaged model: verified
- Local and Docker sample predictions: matched
- GitHub Actions: passing

The Docker application has been verified locally. Public cloud hosting has not yet been completed.

## Future Improvements

- Deploy the API to a public cloud hosting platform
- Add structured logging and monitoring
- Add authentication, request limits, and production security controls
- Track model drift and prediction performance over time
- Implement automated model retraining
- Improve classification threshold selection and probability calibration
- Add a user-friendly prediction frontend

## Repository

https://github.com/Mounika636/customer-churn-mlops
