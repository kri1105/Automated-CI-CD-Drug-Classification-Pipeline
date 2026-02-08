import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/drug200.csv"
TARGET = "Drug"
MODEL_PATH = "model/drug_pipeline.pkl"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -----------------------------
# Feature groups
# -----------------------------
categorical_features = ["Sex", "BP", "Cholesterol"]
numerical_features = ["Age", "Na_to_K"]

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# -----------------------------
# Pipeline
# -----------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

# -----------------------------
# Save model
# -----------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"Pipeline saved at: {MODEL_PATH}")
