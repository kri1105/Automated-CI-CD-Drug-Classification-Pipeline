import joblib
import pandas as pd
import os

MODEL_PATH = "model/drug_pipeline.pkl"

# Check if model exists
assert os.path.exists(MODEL_PATH), "Model file not found!"

# Load pipeline
pipeline = joblib.load(MODEL_PATH)

# Dummy test input
test_data = pd.DataFrame([
    {
        "Age": 30,
        "Sex": "F",
        "BP": "LOW",
        "Cholesterol": "NORMAL",
        "Na_to_K": 12.3
    }
])

# Predict
prediction = pipeline.predict(test_data)

# Basic sanity check
assert len(prediction) == 1, "Prediction failed!"
assert isinstance(prediction[0], str), "Prediction output is not a class label"

print("✅ CI Test Passed: Pipeline loaded and prediction successful")
