from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load all models, imputer and MSE scores
with open("model.pkl", "rb") as f:
    models, imputer, mse_scores = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def predict():
    error = None
    prediction = None

    if request.method == "POST":
        try:
            data = [
                float(request.form["voltage"]),
                float(request.form["current"]),
                float(request.form["temperature"]),
                float(request.form["soc"]),
                int(request.form["cycles"]),
                float(request.form["capacity"])
            ]
            input_data = imputer.transform([data])
            prediction = models["RandomForest"].predict(input_data)[0]
            prediction = round(prediction, 2)

        except ValueError:
            error = "Please enter valid numeric values in all fields."
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"

    return render_template("index2.html", prediction=prediction, error=error, mse_scores=mse_scores)

if __name__ == "__main__":
    app.run(debug=True)
