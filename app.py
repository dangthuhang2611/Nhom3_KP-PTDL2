# ============================
# app.py (version dùng Pipeline)
# ============================

from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Chỉ load 1 file: pipeline.pkl
pipeline = pickle.load(open("model.pkl", "rb"))

# Load dataset để lấy danh sách input
df = pd.read_csv("cleaned_stock_dataset.csv")

input_features = df.drop(columns=[
    "Annual_Return",
    "Excess_Return",
    "Systematic_Risk"
]).columns.tolist()

@app.route("/")
def index():
    return render_template("index.html", input_features=input_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy dữ liệu từ form
        user_input = [float(request.form[col]) for col in input_features]

        # Không scale nữa — pipeline đã lo toàn bộ
        pred = pipeline.predict([user_input])[0]

        results = {
            "Annual_Return": float(pred)
        }

        return render_template(
            "index.html",
            input_features=input_features,
            results=results
        )

    except Exception as e:
        return f"Prediction error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
