import joblib
import numpy as np
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "../models/trained_model.pkl")
model = joblib.load(model_path)

def predict(data, gender="Female"):
    """
    Predict diagnosis using the trained model.
    
    Parameters:
    data (list): 12 features in order:
        [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
         DiabetesPedigreeFunction, Age, HeartRate, Weight, BodyTemperature, OxygenSaturation]
    gender (str): "Male" or "Female". Sets Pregnancies=0 for males.
    
    Returns:
    int: 1 for Positive, 0 for Negative
    """

    data = np.array(data, dtype=float).reshape(1, -1)

    # Handle male pregnancies
    if gender.lower() == "male":
        data[0, 0] = 0

    # Check feature count
    if data.shape[1] != model.n_features_in_:
        raise ValueError(f"Input data must have {model.n_features_in_} features, got {data.shape[1]}")

    return int(model.predict(data)[0])
