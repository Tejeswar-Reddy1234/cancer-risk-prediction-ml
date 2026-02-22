from flask import Flask, request, jsonify
import joblib
import numpy as np
import config

app = Flask(__name__)

model = joblib.load(config.MODEL_PATH)

@app.route("/")
def home():
    return "Cancer Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["features"]
    data = np.array(data).reshape(1,-1)

    prediction = model.predict(data)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)