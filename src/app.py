from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# Load model
model = mlflow.sklearn.load_model("models:/model/production")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame(data)
    predictions = model.predict(input_data)
    return jsonify(predictions.tolist())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
