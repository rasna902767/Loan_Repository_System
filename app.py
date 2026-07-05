from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Model accuracy
MODEL_ACCURACY = 84.0


@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=MODEL_ACCURACY
    )


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["Age"])
    income = float(request.form["Income"])
    loan = float(request.form["LoanAmount"])
    credit = float(request.form["CreditScore"])

    input_data = pd.DataFrame(
        [[age, income, loan, credit]],
        columns=[
            "Age",
            "Income",
            "LoanAmount",
            "CreditScore"
        ]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        accuracy=MODEL_ACCURACY
    )


if __name__ == "__main__":
    app.run(debug=True)