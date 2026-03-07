from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    claim_amount = float(request.form['claim_amount'])

    # create dummy input with 37 features
    features = [0]*37

    # put claim amount as one feature
    features[0] = claim_amount

    prediction = model.predict([features])

    result = "Fraud Claim" if prediction[0] == 1 else "Genuine Claim"

    return f"Prediction Result: {result}"


if __name__ == "__main__":
    app.run(debug=True)