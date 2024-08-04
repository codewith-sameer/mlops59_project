import mlflow.sklearn
import pandas as pd

# Load model
model = mlflow.sklearn.load_model("models:/model/production")


def predict(input_data):
    input_df = pd.DataFrame(input_data)
    predictions = model.predict(input_df)
    return predictions
