from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs
        hours = float(request.form['hours'])
        attendance = float(request.form['attendance'])
        prevmarks = float(request.form['prevmarks'])
        assignments = float(request.form['assignments'])
        sleep = float(request.form['sleep'])

        # Prepare input for prediction
        input_data = np.array([[hours, attendance, prevmarks, assignments, sleep]])
        prediction = model.predict(input_data)[0]

        # Optional: confidence
        proba = model.predict_proba(input_data)
        confidence = round(np.max(proba) * 100, 2)

        return render_template("index.html",
                               result=prediction,
                               confidence=confidence,
                               hours=hours,
                               attendance=attendance,
                               prevmarks=prevmarks,
                               assignments=assignments,
                               sleep=sleep)
    except Exception as e:
        return render_template("index.html", error=f"Invalid input! ({e})")

if __name__ == "__main__":
    app.run(debug=True)