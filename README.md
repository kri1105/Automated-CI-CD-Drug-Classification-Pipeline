# CI/CD Enabled Drug Recommendation System

## 📌 Project Overview

This project is an **end-to-end Machine Learning system** that predicts the most suitable drug for a patient based on medical attributes. The system is built with **production-quality practices**, including automated preprocessing, model training, evaluation, and **CI/CD validation using GitHub Actions**.

Unlike a simple ML notebook, this project demonstrates how machine learning models are **built, tested, and validated automatically** in real-world engineering workflows.

---

## 🚀 Key Features

* End-to-end **scikit-learn Pipeline** for preprocessing and training
* Handles **mixed data types** (numerical + categorical)
* Automated **model training and evaluation**
* **CI/CD pipeline** using GitHub Actions
* Sanity testing of trained model during CI
* Serialized ML pipeline for reproducible inference
* Resume & industry-ready project structure

---

## 🧠 Dataset Description

**Dataset:** `drug200.csv`

The dataset contains patient attributes and the corresponding drug prescribed.

### Features:

| Column Name | Description                          |
| ----------- | ------------------------------------ |
| Age         | Patient age                          |
| Sex         | Gender (M/F)                         |
| BP          | Blood Pressure (LOW / NORMAL / HIGH) |
| Cholesterol | Cholesterol level                    |
| Na_to_K     | Sodium to Potassium ratio            |

### Target:

* **Drug** (Multi-class classification: DrugA, DrugB, DrugC, DrugX, DrugY)

---

## 🏗️ Project Structure

```
project-root/
│
├── data/
│   └── drug200.csv
│
├── model/
│   └── drug_pipeline.pkl
│
├── train.py
├── test_pipeline.py
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## ⚙️ ML Pipeline Architecture

```
Raw Data
   ↓
ColumnTransformer
   ├── Numerical → StandardScaler
   └── Categorical → OneHotEncoder
   ↓
Logistic Regression Classifier
   ↓
Evaluation & Serialization
```

This ensures **no data leakage** and consistent preprocessing during both training and inference.

---

## 🧪 CI/CD Workflow

On every **push or pull request**, GitHub Actions automatically:

1. Installs dependencies
2. Trains the ML pipeline
3. Saves the trained model (`.pkl`)
4. Loads the model
5. Runs a test prediction
6. Fails the build if any step breaks

This guarantees model reliability and reproducibility.

---

## ▶️ How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the pipeline

```bash
python train.py
```

### 3. Test the trained model

```bash
python test_pipeline.py
```

---

## 🔍 Sample Prediction Code

```python
import joblib
import pandas as pd

pipeline = joblib.load("model/drug_pipeline.pkl")

sample = pd.DataFrame([
    {
        "Age": 45,
        "Sex": "M",
        "BP": "HIGH",
        "Cholesterol": "NORMAL",
        "Na_to_K": 15.5
    }
])

print(pipeline.predict(sample))
```

---

## 📊 Model Performance

* Accuracy: ~98–99%
* Evaluation Metric: Accuracy Score
* Model: Logistic Regression (baseline)

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **Joblib**
* **GitHub Actions (CI/CD)**

---

## 📈 Future Enhancements

* Add accuracy threshold quality gate in CI
* Integrate FastAPI for real-time inference
* Add model versioning
* Track experiments using metrics logging
* Replace baseline model with ensemble methods

---

## 🎯 Resume Highlights

* Built an end-to-end ML pipeline with automated preprocessing and training
* Implemented CI/CD validation for machine learning workflows
* Ensured reproducibility through pipeline serialization
* Designed industry-ready project structure following MLOps principles

---

## 📎 License

This project is intended for educational and demonstration pu

