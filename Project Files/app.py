from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="Flask/static"
)

# ------------------ Load Model ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Training", "HDI.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model Loaded Successfully!")
print("Features:", model.feature_names_in_)

# ------------------ Home ------------------
@app.route("/")
def home():
    return render_template("indexnew.html")


@app.route("/Home")
def Home():
    return render_template("home.html")


@app.route("/Prediction")
def Prediction():
    return render_template("indexnew.html")


# ------------------ Prediction ------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        life = float(request.form["life"])
        expected = float(request.form["expected"])
        mean = float(request.form["mean"])
        gni = float(request.form["gni"])

        X = pd.DataFrame({
            "Life Expectancy at Birth (2021)": [life],
            "Expected Years of Schooling (2021)": [expected],
            "Mean Years of Schooling (2021)": [mean],
            "Gross National Income Per Capita (2021)": [gni]
        })

        prediction = float(model.predict(X)[0])
        prediction = round(prediction, 3)

        if prediction < 0.4:
            category = "Low HDI"
        elif prediction < 0.7:
            category = "Medium HDI"
        elif prediction < 0.8:
            category = "High HDI"
        else:
            category = "Very High HDI"

        prediction_text = f"""
        <h2>{category}</h2>
        <h3>Predicted HDI Score : {prediction}</h3>
        """

        return render_template(
            "resultnew.html",
            prediction_text=prediction_text,
            is_error=False
        )

    except Exception as e:
        return render_template(
            "resultnew.html",
            prediction_text=f"Error : {e}",
            is_error=True
        )


if __name__ == "__main__":
    app.run(debug=True)