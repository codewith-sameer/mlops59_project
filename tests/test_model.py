import pytest
from flask.testing import FlaskClient
import json
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "../src")))

from app import app  # noqa: E402


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_home(client: FlaskClient):
    """Test the home route"""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome to the ML model prediction service!" in response.data


def test_predict(client: FlaskClient):
    """Test the predict route"""
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


if __name__ == "__main__":
    pytest.main()
