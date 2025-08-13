from flask import Flask, render_template, request
import sys, os

# Include src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def prediction():
    try:
        gender = request.form.get("Gender", "Female")
        pregnancies = 0 if gender.lower() == "male" else float(request.form.get("Pregnancies", 0))

        # Features in order
        feature_names = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI",
                         "DiabetesPedigreeFunction","Age","HeartRate","Weight",
                         "BodyTemperature","OxygenSaturation"]

        features = [pregnancies]
        for name in feature_names:
            value = request.form.get(name)
            if value is None or value.strip() == "":
                raise ValueError(f"Missing value for {name}")
            features.append(float(value))

        result = predict(features, gender)
        diagnosis = "Positive" if result == 1 else "Negative"

        return render_template("index.html", result=diagnosis)

    except ValueError as ve:
        return render_template("index.html", result=f"Error: {ve}")
    except Exception as e:
        return render_template("index.html", result=f"Unexpected error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
