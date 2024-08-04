import pytest
from flask.testing import FlaskClient
import json
import sys
import os
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../src")))

from app import app  # noqa: E402


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


@patch("mlflow.sklearn.load_model")
def test_home(mock_load_model, client: FlaskClient):
    """Test the home route"""
    mock_load_model.return_value = MagicMock()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome to the ML model prediction service!" in response.data


@patch("mlflow.sklearn.load_model")
def test_predict(mock_load_model, client: FlaskClient):
    """Test the predict route"""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1]
    mock_load_model.return_value = mock_model

    data = {
        "sepal length (cm)": [5.1, 5.9],
        "sepal width (cm)": [3.5, 3.0],
        "petal length (cm)": [1.4, 4.2],
        "petal width (cm)": [0.2, 1.5],
    }
    response = client.post(
        "/predict", data=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert isinstance(json_data, list)  # Ensure the prediction is a list
    assert json_data == [0, 1]  # Ensure the mocked prediction is returned


if __name__ == "__main__":
    pytest.main()
