from flask import Flask, request, jsonify, render_template
from joblib import load

app = Flask(__name__)

# Load the model at the start
model = load("optimized_rf_model.joblib")

# Serve the HTML form
@app.route("/")
def index():
    return render_template("index.html")

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting JSON payload
        features = [data[feature] for feature in ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
        prediction = model.predict([features])[0]
        prediction_in_dollars = prediction * 1000
        formatted_prediction = f"${prediction_in_dollars:,.2f}"
        return jsonify({"prediction": formatted_prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)