import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ----------------------------
# Paths (relative to this file)
# ----------------------------
# Use "../data" because this script is inside "src"
data_path = os.path.join(os.path.dirname(__file__), "../data/medical_dataset.csv")
model_path = os.path.join(os.path.dirname(__file__), "../models/trained_model.pkl")

# Make sure models folder exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# ----------------------------
# Load dataset
# ----------------------------
data = pd.read_csv(data_path)

# ----------------------------
# Features and target
# ----------------------------
features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'HeartRate', 'Weight',
    'BodyTemperature', 'OxygenSaturation'
]

# Check if all features exist
missing = set(features + ['Outcome']) - set(data.columns)
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

X = data[features]
y = data['Outcome']

print(f"Number of features: {X.shape[1]}")  # should print 12

# ----------------------------
# Split data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Evaluate model
# ----------------------------
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print("Model trained successfully!")
print("Accuracy:", accuracy)

# ----------------------------
# Save trained model
# ----------------------------
joblib.dump(model, model_path)
print(f"Trained model saved at: {model_path}")
